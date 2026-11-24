"""Run the PvP bot against the live game.

  python scripts/run_bot.py               # you play; click L3+R3 to hand over to the bot
  python scripts/run_bot.py --dry-run     # read memory and print decisions, no controller
  python scripts/run_bot.py --bot-only    # no handoff: the bot drives from the start
  python scripts/run_bot.py --red-sign    # while the bot drives, keep a red sign down

Handoff (default): the bot's virtual pad is the only controller the game reads, and
your real controller is passed through it. Click L3 + R3 (both sticks, within 0.4 s) to let the
bot play; touch anything to take control back. The bot only fights player characters
(humans, hollows, red phantoms), never NPCs.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ptde_ai.adapt import Adapter, Profiles               # noqa: E402
from ptde_ai.brain import Brain, Config                    # noqa: E402
from ptde_ai import geometry as G                          # noqa: E402
from ptde_ai.character import CharacterTracker, fightable, pick_target  # noqa: E402
from ptde_ai.escape import (ARRIVAL_LOOK_MS, DONE, TRAVERSAL_TRINKET,  # noqa: E402
                            DugoutEscape, should_queue_on_arrival)
from ptde_ai.focus import FocusWatch                       # noqa: E402
from ptde_ai.freeze import FreezeWatch                     # noqa: E402
from ptde_ai.gamepad import DDOWN                         # noqa: E402
from ptde_ai.greet import Greeter                          # noqa: E402
from ptde_ai.heal import Healer, between_fights            # noqa: E402
from ptde_ai.character import Sub                          # noqa: E402
from ptde_ai.fightlog import FightLog                      # noqa: E402
from ptde_ai import players                                # noqa: E402
from ptde_ai.handoff import BOT, MANUAL, Handoff           # noqa: E402
from ptde_ai.memory import ProcMem                         # noqa: E402
from ptde_ai.nets import MODEL_DIR, MLP                    # noqa: E402
from ptde_ai.offsets import HOME_CHR_TYPES, PLAYER_CHR_TYPES, npc_target  # noqa: E402
from ptde_ai.process import find_game_pid                  # noqa: E402
from ptde_ai.recorder import Recorder                       # noqa: E402
from ptde_ai.redsign import RedSign, soapstone_ready       # noqa: E402
from ptde_ai.retrain import Retrainer                      # noqa: E402
from ptde_ai.state import StateReader                      # noqa: E402
from ptde_ai.steer import CAL_PATH, Steering               # noqa: E402


ROOT = Path(__file__).resolve().parent.parent


def code_mtime() -> float:
    files = list((ROOT / "ptde_ai").glob("*.py")) + [Path(__file__).resolve()]
    return max(f.stat().st_mtime for f in files)


CONTROL = ROOT / "control" / "press"


def read_press_script() -> list[tuple[int, int, float]]:
    """Test hook: control/press holds lines like "SELECT 120", "DDOWN 100", "WAIT 500".
    Returns [(buttons, hat, ms)] and deletes the file. Only used for development."""
    from ptde_ai import gamepad as gp
    steps = []
    try:
        text = CONTROL.read_text()
        CONTROL.unlink()
    except OSError:
        return steps
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        name, ms = parts[0].upper(), float(parts[1]) if len(parts) > 1 else 120.0
        if name == "WAIT":
            steps.append((0, gp.DCENTER, ms))
        elif name == "FWD":                       # push the left stick forward
            steps.append((-1, gp.DCENTER, ms))
            steps.append((0, gp.DCENTER, 80.0))
        elif name in ("DUP", "DDOWN", "DLEFT", "DRIGHT"):
            steps.append((0, getattr(gp, name), ms))
            steps.append((0, gp.DCENTER, 80.0))
        elif hasattr(gp, name):
            steps.append((getattr(gp, name), gp.DCENTER, ms))
            steps.append((0, gp.DCENTER, 80.0))
    return steps


def code_imports() -> tuple[bool, str]:
    """Does the code on disk import? (checked in a separate process before a hot reload)"""
    import subprocess
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); "
                        "import ptde_ai.brain, ptde_ai.escape, ptde_ai.redsign, ptde_ai.hud, "
                        "ptde_ai.freeze, ptde_ai.threat, ptde_ai.adapt, ptde_ai.retrain; "
                        "import py_compile; py_compile.compile(sys.argv[2], doraise=True)",
                        str(ROOT), str(Path(__file__).resolve())],
                       capture_output=True, text=True, timeout=30)
    return r.returncode == 0, (r.stderr.strip().splitlines() or ["?"])[-1]


class Reload(Exception):
    """Code changed while idle at home: restart in place to pick it up."""


class NullPad:
    """Stands in for the virtual pad in --dry-run."""
    def __init__(self):
        self.reset()

    def reset(self):
        from ptde_ai.gamepad import DCENTER, MIDDLE
        self.lx = self.ly = self.rx = self.ry = MIDDLE
        self.buttons, self.hat = 0, DCENTER

    def send(self):
        pass

    def close(self):
        pass


def load_net(name: str):
    path = MODEL_DIR / f"{name}.npz"
    if not path.exists():
        print(f"warning: {path.name} missing, run tools/train_nets.py ({name} net disabled)")
        return None
    return MLP.load(path)


def claim_slot_zero():
    """Create the bot pad so it owns XInput slot 0, moving a connected Bluetooth
    controller behind it. Returns (pad, real_mac or None)."""
    from ptde_ai.gamepad import VirtualPad
    from ptde_ai.realpad import (bluetooth_connect, bluetooth_disconnect, bluetooth_mac,
                                 find_real_controller)
    real = find_real_controller()
    mac = bluetooth_mac(real) if real else None
    if real and not mac:
        print(f"note: {real.name} looks wired. Unplug it now and plug it back in after "
              "the bot starts, or the game will keep reading it instead of the bot.")
    if mac:
        print(f"moving {real.name} behind the bot's pad (Bluetooth reconnect)...")
        bluetooth_disconnect(mac)
        time.sleep(1.5)
    pad = VirtualPad()
    time.sleep(1.5)            # let Wine register it in the free slot 0
    if mac:
        if bluetooth_connect(mac):
            print("controller reconnected")
        else:
            print("couldn't reconnect it automatically: press the Xbox button on your controller")
    return pad, mac


def release_slot_zero(pad, mac):
    """Remove the bot pad and bounce the real controller so it gets slot 0 back."""
    from ptde_ai.realpad import bluetooth_connect, bluetooth_disconnect
    pad.close()
    if mac:
        bluetooth_disconnect(mac)
        time.sleep(1.5)
        if not bluetooth_connect(mac):
            print("press the Xbox button on your controller to reconnect it")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print decisions, don't press anything")
    ap.add_argument("--bot-only", action="store_true", help="no handoff: the bot drives from the start")
    ap.add_argument("--hz", type=float, default=120.0, help="decision rate (subroutine windows are 30-50 ms)")
    ap.add_argument("--start-as-bot", action="store_true",
                    help="with handoff: the bot starts in control (touch your controller to take over)")
    ap.add_argument("--red-sign", action="store_true",
                    help="while the bot is in control at home, place Red Sign Soapstone signs")
    ap.add_argument("--no-record", action="store_true", help="don't save fight recordings")
    ap.add_argument("--no-greet", action="store_true", help="don't bow to the host after being summoned")
    ap.add_argument("--sign-down", action="store_true",
                    help="a red sign is already placed (set automatically on hot reload)")
    ap.add_argument("--no-reload", action="store_true",
                    help="don't restart automatically when the code changes")
    ap.add_argument("--trinket-selected", action="store_true",
                    help="arena: the Traversal Trinket is the selected quick item (e.g. stopped "
                         "mid-escape); walk to the centre, switch to the soapstone and sign")
    ap.add_argument("--allow-r1", action="store_true",
                    help="earlier style: pokes and R1 punishes (default build is backstab-only)")
    ap.add_argument("--no-attack-net", action="store_true")
    ap.add_argument("--no-defense-net", action="store_true")
    a = ap.parse_args()

    if not CAL_PATH.exists() and not a.dry_run:
        sys.exit("No calibration.json yet: run `python scripts/probe.py --calibrate-steering` first.")

    mem = ProcMem(find_game_pid())
    reader = StateReader(mem)
    cfg = Config(use_attack_net=not a.no_attack_net, use_defense_net=not a.no_defense_net,
                 backstab_only=not a.allow_r1)
    from ptde_ai.hitmemory import HitMemory
    hits = HitMemory()
    if hits.table:
        print(f"hit damage memory: {len(hits.table)} attacks known")
    brain = Brain(Steering.load(), load_net("defense"), load_net("attack"), cfg, hits=hits)
    if (MODEL_DIR / "threat.npz").exists():
        brain.threat_net = MLP.load(MODEL_DIR / "threat.npz")
        print("backstab threat net: trained on the bot's own fights (replaces the old defense net)")
    from ptde_ai.arena_map import ArenaMap
    from ptde_ai.escape import load_arena
    brain.arena = ArenaMap.from_arena(load_arena())
    if brain.arena is not None:
        print(f"arena walls: {len(brain.arena.polygon)}-sided walkable area, {brain.arena.area:.0f} m^2 "
              "(back to the wall = cover; no rolling into walls)")
    me = CharacterTracker(is_player=True, weapon_range=cfg.player_weapon_range)
    them = CharacterTracker(is_player=False, weapon_range=cfg.enemy_weapon_range)
    them.timings = hits                 # learned hit times for attacks missing from the table

    redsign = None
    if a.red_sign:
        ok, why = soapstone_ready(reader.quick_items())
        print(f"red sign: {why}")
        if ok:
            redsign = RedSign()
            from ptde_ai.hud import sign_visible
            redsign.sign_visible = sign_visible
            if a.sign_down:
                redsign.sign_down, redsign.last_sign_ms = True, time.perf_counter() * 1000.0
                print("red sign: already down (carried over from the previous run)")
            if reader.self_types().values.get("chr_type") == 2:
                # started while already summoned (restarted mid-fight): no arrival - no bow,
                # no dugout queue, just carry on fighting
                redsign.was_away = True
                print("red sign: started while summoned: carrying on the fight")
        else:
            print("red sign disabled")

    # Arena mod: after dying the bot reloads in a dugout; it uses the Traversal Trinket,
    # walks to the arena centre and signs again (positions in arena.json).
    quick = [i for i in reader.quick_items() if i > 0]
    escape = None
    if TRAVERSAL_TRINKET in quick:
        escape = DugoutEscape(n_items=len(quick))
        # where each item sits in the d-pad cycle, counted from the soapstone (quick-slot order)
        from ptde_ai.offsets import RED_SIGN_SOAPSTONE
        from ptde_ai.redsign import POISON_KNIFE
        if RED_SIGN_SOAPSTONE in quick:
            s0 = quick.index(RED_SIGN_SOAPSTONE)
            escape.trinket_at = (quick.index(TRAVERSAL_TRINKET) - s0) % len(quick)
            if POISON_KNIFE in quick:
                escape.knife_at = (quick.index(POISON_KNIFE) - s0) % len(quick)
        print(f"quick items: {quick} -> trinket {escape.trinket_at} press(es) from the soapstone"
              + (f", poison knives {escape.knife_at}" if escape.knife_at is not None else ""))
        brain.items = escape                  # the brain throws knives: it shares the item count
        from ptde_ai.hud import HudReader, grab
        escape.hud = HudReader()
        seen = grab()
        print(f"quick-item screen check: {'sees the ' + seen if seen else 'CAN NOT read the HUD (the bot will not use items in the dugout until it can)'}")
        if escape.ready and a.trinket_selected:
            from ptde_ai.escape import WALK
            escape.cycles = escape.trinket_at
            p0 = CharacterTracker(is_player=True, weapon_range=cfg.player_weapon_range).update(reader.player())
            if p0 is not None and escape.in_dugout(p0):
                print("dugout escape: on; trinket selected and we're in the dugout: waiting our turn, "
                      "then using it.")
            else:
                escape.state, escape.t_state = WALK, time.perf_counter() * 1000.0
                print("dugout escape: on; trinket selected, so walking to the centre and switching "
                      "back to the soapstone first.")
        elif escape.ready:
            print("dugout escape: on. Start with the SOAPSTONE selected (the bot counts its own "
                  "d-pad presses; don't change quick items while it drives).")
        else:
            print("dugout escape: off (arena.json needs 'dugout' and 'center')")
            escape = None

    mac, real, handoff = None, None, None
    if a.dry_run:
        pad = NullPad()
    else:
        pad, mac = claim_slot_zero()
        print(f"bot pad at {pad.ui.device.path}")
        if not a.bot_only:
            from ptde_ai.realpad import RealPad
            real = RealPad()
            handoff = Handoff(start_as_bot=a.start_as_bot)
            who = "the bot is" if a.start_as_bot else "you're"
            print(f"{who} in control. Click L3+R3 (both sticks, within 0.4 s) to let the bot play;"
                  " touch anything to take back over.")

    # Dark Souls ignores the controller while its window is in the background.
    focus = None if a.dry_run else FocusWatch()
    focus_noted = None

    rec = None if a.no_record else Recorder()
    if rec is not None:
        print(f"recording fights to {rec.path.relative_to(rec.path.parent.parent)}")

    def record(mode):
        if rec is not None and p is not None and e is not None and fightable(e, self_type)[0]:
            rec.tick(now_ms, mode, self_type, p, e, pad, brain if mode == "bot" else None)

    period = 1.0 / a.hz
    start_mtime, last_code_check, reloading = code_mtime(), 0.0, False
    press_steps, press_until, last_press_check = [], 0.0, 0.0
    greeter = Greeter()
    target_slot, last_target_check = None, -1e9
    others_in_arena = others_in_world = None
    user_hat = 0
    bad_mtime = None
    freeze = FreezeWatch()
    arrived_t, arrival_decided, greet_attacked = None, True, False
    npcs_seen: set = set()
    healer, heal_max = Healer(), 0
    fights = FightLog()
    profiles = Profiles()
    adapter = Adapter(profiles)
    retrainer = Retrainer()
    if retrainer.running:
        print(f"retrain: still running from before the reload (pid {retrainer.pid})")
    brain_err_t = -1e9
    idle_reported = None
    dry_sign_noted = False
    print("running; Ctrl+C to stop")
    try:
        while True:
            t0 = time.perf_counter()
            now_ms = t0 * 1000.0
            p = me.update(reader.player(), brain.history, now_ms)
            cam = reader.camera()
            self_type = reader.self_types().values.get("chr_type")
            if p is not None and now_ms - last_target_check > 200:
                last_target_check = now_ms
                arena = brain.arena if brain.arena is not None and brain.arena.here(p) else None
                inside = (lambda x, y: arena.wall_distance(x, y) > -0.7) if arena else None
                slots = reader.slots()
                # other players (not NPCs) anywhere in this world: one-on-one = no dugout queue
                # fighters: players, plus the Havelmom summon (the one NPC that's an opponent)
                fighter = lambda sl: sl["chr_type"] in PLAYER_CHR_TYPES or npc_target(sl["chr_type"], sl["entity_id"])
                others_in_world = sum(1 for sl in slots if fighter(sl))
                if brain.arena is not None:
                    # players inside the walls, at ANY HP: for the dugout queue. Live 2026-11-24
                    # one of the two dropped to 1 HP (round over) but was still standing in the
                    # arena, and the bot left the dugout 1.7 s after arriving. It's our turn
                    # when someone actually leaves.
                    others_in_arena = sum(1 for sl in slots if fighter(sl)
                                          and sl["hp"] > 0
                                          and brain.arena.wall_distance(sl["x"], sl["y"]) > -0.7)
                # log each NPC that's inside the arena once: the Havelmom summon (entity 1269420)
                # is the only NPC the bot may fight; this shows what else is there
                for sl in slots:
                    if sl["chr_type"] in PLAYER_CHR_TYPES or (inside and not inside(sl["x"], sl["y"])):
                        continue
                    key = (sl["slot"], sl["chr_type"], sl["team_type"], sl["entity_id"])
                    if key not in npcs_seen and sl["hp"] > 0:
                        npcs_seen.add(key)
                        tgt = npc_target(sl["chr_type"], sl["entity_id"])
                        line = (f"NPC in the arena: entity {sl['entity_id']} slot {sl['slot']} type {sl['chr_type']} team "
                                f"{sl['team_type']} hp {sl['hp']} weapons {sl['r_weapon']}/{sl['l_weapon']}"
                                f" -> {'fighting it (' + tgt + ')' if tgt else 'leaving it alone'}")
                        print(f"[{now_ms / 1000:10.3f}] {line}")
                        if rec is not None:
                            rec.event(now_ms, line)
                pick = pick_target(slots, self_type, (p.loc_x, p.loc_y), target_slot, in_arena=inside)
                if pick is None and target_slot is None:
                    # nobody in the arena yet: watch whoever we'd fight (the bot waits in the middle)
                    pick = pick_target(slots, self_type, (p.loc_x, p.loc_y), None)
                if pick is not None and pick != target_slot:
                    if target_slot is not None:
                        print(f"[{now_ms / 1000:10.3f}] new target: character slot {pick}")
                    target_slot = pick
                    them = CharacterTracker(is_player=False, weapon_range=cfg.enemy_weapon_range)
                    them.timings = hits
                    brain.reset_routines()
                    fights.retarget()
                    adapter.abort()
                    adapter.sid, adapter.name = None, "?"   # don't credit them until identified
                    if rec is not None:
                        rec.event(now_ms, f"new target: character slot {pick}")
            e = them.update(reader.enemy(target_slot or 1), brain.history, now_ms)
            if redsign is not None:
                msg = redsign.update_location(self_type, now_ms)
                if msg and "summoned" in msg and not a.no_greet:
                    greeter.start(now_ms)
                if msg and "summoned" in msg:
                    arrived_t, arrival_decided, greet_attacked = now_ms, False, False
                if msg:
                    npcs_seen.clear()                 # new world: new NPCs
                if msg and "back in our world" in msg:
                    greeter.state = "done"        # a greeting cut short (you took over) ends here
                    arrived_t, arrival_decided = None, True
                if msg:
                    freeze.arrived(now_ms)
                    brain.reset_routines()
                    fights.new_fight()
                    adapter.abort()
                    adapter.sid, adapter.name = None, "?"
                    print(f"[{now_ms / 1000:10.3f}] {msg}")
                    if rec is not None:
                        rec.event(now_ms, msg)

            # who we're fighting (Steam name / ID) and how it ends -> records/fights.jsonl
            if p is not None and e is not None and fightable(e, self_type)[0]:
                if fights.want_lookup(now_ms):
                    try:
                        who = players.identify(mem, players.enemy_chr_ins(mem, target_slot or 1))
                    except (OSError, ValueError):
                        who = None
                    msg = fights.set_opponent(who)
                    if msg:
                        print(f"[{now_ms / 1000:10.3f}] {msg}")
                        if rec is not None:
                            rec.event(now_ms, msg)
                        line = adapter.set_opponent(who.steam_id, who.steam_name)
                        print(f"[{now_ms / 1000:10.3f}] {line}")
                result = fights.tick(p.hp, e.hp, now_ms)
                if result:
                    o = fights.opponent
                    msg = (f"{'WON against' if result == 'won' else 'lost to'} "
                           f"{o.steam_name if o else 'an unidentified player'} (records/fights.jsonl)")
                    print(f"[{now_ms / 1000:10.3f}] {msg}")
                    if rec is not None:
                        rec.event(now_ms, msg)

            # background retraining of the backstab net: only while idle, frozen during fights
            if not a.dry_run:
                idle_now = (p is not None and self_type in HOME_CHR_TYPES
                            and (handoff is None or handoff.mode == BOT)
                            and not (e is not None and fightable(e, self_type)[0])
                            and not (escape is not None and escape.active))
                for line in retrainer.tick(now_ms, idle_now):
                    print(f"[{now_ms / 1000:10.3f}] {line}")

            if handoff is not None:
                s = real.snapshot()
                changed = handoff.update(s, now_ms)
                if changed:
                    brain.reset_routines()
                    print(f"[{now_ms / 1000:10.3f}] control -> {changed.upper()}")
                if handoff.mode == MANUAL:
                    if escape is not None:        # your quick-item presses (d-pad down) count too
                        h = s.hat()
                        if h == DDOWN and user_hat != DDOWN:
                            escape.user_pressed_ddown()
                        user_hat = h
                    s.apply_to(pad)
                    pad.send()
                    record("you")
                    if p is not None and e is not None:
                        brain.observe(p, e, now_ms)
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                    continue

            # Bot is driving (or --bot-only / --dry-run).
            if not a.dry_run and now_ms - last_press_check > 250:
                last_press_check = now_ms
                greet_file = CONTROL.with_name("greet")
                if greet_file.exists():           # test hook: run the greeting now
                    greet_file.unlink(missing_ok=True)
                    greeter.start(now_ms, force=True)
                    print(f"[{now_ms / 1000:10.3f}] test hook: greeting")
                if CONTROL.exists():
                    press_steps = read_press_script()
                    press_until = now_ms
                    print(f"[{now_ms / 1000:10.3f}] test hook: {len(press_steps)} steps")
            if press_steps or now_ms < press_until:
                if now_ms >= press_until and press_steps:
                    btn, hat, ms = press_steps.pop(0)
                    pad.reset()
                    if btn == -1:
                        pad.ly = 1                        # stick forward
                    else:
                        pad.buttons, pad.hat = btn, hat
                        if escape is not None and hat == DDOWN:
                            escape.user_pressed_ddown()
                    press_until = now_ms + ms
                pad.send()
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                continue
            game_focused = focus.focused if focus is not None else None
            if game_focused is not None and game_focused != focus_noted:
                if not game_focused:
                    print(f"[{now_ms / 1000:10.3f}] game window isn't focused: Dark Souls ignores the"
                          " controller until you click its window (red signs paused)")
                elif focus_noted is False:
                    print(f"[{now_ms / 1000:10.3f}] game window focused again")
                focus_noted = game_focused
            # Heal between fights with a Divine Blessing (user 2026-11-24), also between 1 HP
            # rounds: only between fights (heal.between_fights: waiting in the dugout, round over,
            # or no opponent near) and nobody's swinging at us. Not while trinketing/walking (the dugout wait
            # is fine), and at 1 HP with 2+ others around the dugout exit goes first.
            if p is not None and not a.dry_run and game_focused is not False:
                heal_max = max(heal_max, p.hp)
                d_e = G.distance(p, e) if e is not None else 99.0
                fight_on = not between_fights(
                    escape is not None and escape.state in ("settle", "wait-turn"), p.hp,
                    e is not None and fightable(e, self_type)[0], e.hp if e is not None else 0, d_e)
                swinging = (e is not None and d_e < e.weapon_range + 1.0 and e.sub in (
                    Sub.WINDUP, Sub.WINDUP_CLOSING, Sub.WINDUP_GHOST_HIT, Sub.ACTIVE_DURING_HURTBOX))
                esc_ok = escape is None or escape.state in ("idle", "done", "settle", "wait-turn")
                leaving = (escape is not None and escape.state in ("idle", "done") and p.hp <= 1
                           and (others_in_world or 0) >= 2)
                safe = (p.hp > 0 and not fight_on and not swinging and esc_ok and not leaving
                        and not greeter.active and (handoff is None or handoff.mode == BOT))
                if healer.active:
                    if not safe:
                        healer.abort(now_ms, pad)
                    pad.reset()
                    msg = healer.update(p.hp, p.animation_type, now_ms, pad)
                    if msg:
                        print(f"[{now_ms / 1000:10.3f}] {msg}")
                        if rec is not None:
                            rec.event(now_ms, msg)
                    pad.send()
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                    continue
                if safe and Healer.wants(p.hp, heal_max) and (handoff is None or handoff.mode == BOT):
                    msg = healer.start(p.hp, now_ms)
                    if msg:
                        print(f"[{now_ms / 1000:10.3f}] {msg} ({p.hp}/{heal_max} HP)")
                        if rec is not None:
                            rec.event(now_ms, msg)
                        pad.reset(); healer.update(p.hp, p.animation_type, now_ms, pad); pad.send()
                        time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                        continue

            # Summoned and two players are already in the arena: straight to the dugout, no
            # bow (user 2026-11-24). Decided once per summon, ARRIVAL_LOOK_MS after arriving.
            if (escape is not None and not a.dry_run and not arrival_decided and arrived_t is not None
                    and now_ms - arrived_t >= ARRIVAL_LOOK_MS and not escape.active and p is not None):
                arrival_decided = True
                if should_queue_on_arrival(self_type == 2, now_ms - arrived_t, others_in_arena, p.hp):
                    greeter.state = "done"            # no bow: just get out
                    msg = escape.queue(now_ms)
                    if msg:
                        print(f"[{now_ms / 1000:10.3f}] {msg}")
                        if rec is not None:
                            rec.event(now_ms, msg)
            if escape is not None and not a.dry_run and game_focused is not False:
                was_active = escape.active
                pad.reset()
                # Also when summoned: in 1 HP rounds the arena puts a "dead" red phantom in the
                # host world's dugout (same coordinates), and the bot walked into its walls for
                # 25 s (2026-11-23). Only signing (redsign) is home-only.
                msg = escape.update(p, self_type in HOME_CHR_TYPES or self_type == 2, now_ms, pad,
                                    brain.steer, cam.values.get("yaw"), others_in_arena,
                                    others_in_world, summoned=self_type == 2)
                if msg:
                    print(f"[{now_ms / 1000:10.3f}] {msg}")
                if escape.state == DONE and was_active and self_type == 2:
                    escape.done_at = None      # summoned: no sign press follows, so a quick return to
                                               # the dugout isn't "the sign press used the trinket"
                elif escape.state == DONE and was_active and redsign is not None:
                    redsign.last_sign_ms, redsign.returned_at = -1e12, None
                    redsign.retry_at, redsign.fails, redsign.sign_down = 0.0, 0, False
                if escape.active:
                    brain.reset_routines()
                    pad.send()
                    time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                    continue
            if greeter.active and not a.dry_run:
                pad.reset()
                host_ok = e is not None and fightable(e, self_type)[0]
                msg = greeter.tick(p, e, now_ms, pad, brain.steer, cam.values.get("yaw"), host_ok)
                if msg and "attacked" in msg:
                    greet_attacked = True         # someone came for us: it's our round now
                if msg:
                    print(f"[{now_ms / 1000:10.3f}] {msg}")
                    if rec is not None:
                        rec.event(now_ms, msg)
                pad.send()
                record("greet")
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
                continue
            idle_reason = None
            if p is None or e is None or "yaw" not in cam.values:
                idle_reason = "no opponent / camera readable"
            elif p.hp <= 0:
                idle_reason = "dead"
            else:
                ok, desc = fightable(e, self_type)
                if not ok:
                    idle_reason = f"not engaging: other character is {desc}"
            if idle_reason:
                for line in freeze.update(p, pad, cam.values, now_ms, engaged=False):
                    print(f"[{now_ms / 1000:10.3f}] {line}")
                if idle_reported != idle_reason:
                    print(idle_reason + " -> idling")
                    idle_reported = idle_reason
                pad.reset()
                if (redsign is not None and p is not None and game_focused is not False
                        and self_type in HOME_CHR_TYPES):
                    if escape is not None and escape.in_dugout(p):
                        pass                      # never sign from the dugout
                    elif (escape is not None and escape.n_items > 0 and escape.cycles % escape.n_items != 0
                          and not escape.active):
                        escape.restore_soapstone(now_ms, pad)   # knives left selected: soapstone first
                    elif a.dry_run:
                        if not dry_sign_noted:
                            print("red sign: would place a sign now (dry run, nothing pressed)")
                            dry_sign_noted = True
                    else:
                        msg = redsign.tick(p, now_ms, pad)
                        if msg:
                            print(f"[{now_ms / 1000:10.3f}] {msg}")
                pad.send()
                # Hot reload: only when nothing is in progress (home, alive, sign down,
                # no escape), so the item count and sign state carry over exactly.
                settled = (not a.dry_run and not a.no_reload and p is not None and p.hp > 0
                           and self_type in HOME_CHR_TYPES
                           and (redsign is None or (redsign.sign_down and not redsign.busy(now_ms)))
                           and (escape is None or (not escape.active and escape.cycles == 0
                                                   and not escape.in_dugout(p))))
                if settled and now_ms - last_code_check > 2000:
                    last_code_check = now_ms
                    if retrainer.model_updated() and (MODEL_DIR / "threat.npz").exists():
                        brain.threat_net = MLP.load(MODEL_DIR / "threat.npz")
                        print(f"[{now_ms / 1000:10.3f}] loaded the retrained backstab net")
                    m = code_mtime()
                    # only once the edits have settled: live 2026-11-23 a reload landed between
                    # two edits seconds apart and the half-edited brain crashed mid-duel
                    if m != start_mtime and time.time() - m > 5:
                        # live 2026-11-24 a reload landed on a half-edited import and the new
                        # process died at startup: only reload code that imports cleanly
                        ok, err = code_imports()
                        if ok:
                            raise Reload
                        if bad_mtime != m:
                            bad_mtime = m
                            print(f"[{now_ms / 1000:10.3f}] code changed but doesn't load, "
                                  f"keeping the running version: {err}")
            else:
                if idle_reported:
                    print(f"opponent: {fightable(e, self_type)[1]} -> engaging")
                idle_reported = None
                # per-opponent styles: score the last exchange / wake-up, pick the next style
                for line in adapter.update(now_ms, p, e, G.distance(p, e), brain.is_down(p)):
                    print(f"[{now_ms / 1000:10.3f}] {line}")
                    if rec is not None:
                        rec.event(now_ms, line)
                brain.style = adapter.style()
                try:
                    brain.tick(p, e, cam.values["yaw"], now_ms, pad)
                except Exception:                 # never leave the character standing idle
                    import traceback
                    if now_ms - brain_err_t > 10000:
                        brain_err_t = now_ms
                        print(f"[{now_ms / 1000:10.3f}] brain error (carrying on):")
                        traceback.print_exc()
                    brain.reset_routines()
                    pad.reset()
                if not a.dry_run:
                    for line in freeze.update(p, pad, cam.values, now_ms, engaged=True):
                        print(f"[{now_ms / 1000:10.3f}] {line}")
                        if rec is not None:
                            rec.event(now_ms, line)
                pad.send()
                record("bot")
                for line in brain.log:
                    print(f"[{now_ms / 1000:10.3f}] {line}")
                brain.log.clear()
            time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    except Reload:
        reloading = True
        print(f"[{now_ms / 1000:10.3f}] code changed: restarting to load it")
    finally:
        brain.hits.maybe_save(force=True)
        profiles.save(force=True)
        retrainer.release()
        if rec is not None:
            rec.close()
            print(f"recorded {rec.ticks} fight ticks to {rec.path.name}")
        if focus is not None:
            focus.close()
        if real is not None:
            if mac is None and real.device is not None:   # controller connected mid-session
                from ptde_ai.realpad import bluetooth_mac
                mac = bluetooth_mac(real.device)
            real.close()
        if isinstance(pad, NullPad) or reloading:
            pad.close()                      # reload: no Bluetooth bounce
        else:
            release_slot_zero(pad, mac)
            print("stopped, controller released")
    if reloading:
        argv = [x for x in sys.argv if x not in ("--trinket-selected", "--sign-down")]
        if redsign is not None and redsign.sign_down:
            argv.append("--sign-down")
        if handoff is not None and handoff.mode == BOT and "--start-as-bot" not in argv:
            argv.append("--start-as-bot")
        os.execv(sys.executable, [sys.executable] + argv)


if __name__ == "__main__":
    main()
