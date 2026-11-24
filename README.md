# [BOT]King: a Dark Souls PvP bot for Linux

[BOT]King plays PvP in **Dark Souls: Prepare to Die Edition** under Steam/Proton on Linux.
It reads the game's memory, decides what to do 120 times a second, and plays through a
virtual Xbox controller. It never writes to the game and never injects code.

It started as a Python port of metal-crow's Windows/C
[Dark-Souls-PvP-AI](https://github.com/metal-crow/Dark-Souls-PvP-AI) and has since grown well
past it: a backstab-focused build, nets retrained on its own fights, per-opponent adaptation,
and full arena-mod etiquette (red-signing, bowing, the dugout queue).

> **Disclaimer.** This is a hobby/research project. Using bots in online play may break the
> game's or Steam's terms of service and may get your account restricted, and other players may
> not want to fight a bot. You are responsible for how you use it. No warranty (see the license).

## What it does

**Fights.** By default it plays a Havel + Great Club backstab build: kick-input backstabs,
races to the opponent's back, run backstabs that punish forward movement, wake-up pressure,
poise trades instead of panic rolls, and mix-ups for turtles, campers and casters. It dodges
melee by animation timing, sidesteps projectiles, and gets out of fire whips, surges and
tempests. `--allow-r1` switches to the original's poke style.

**Learns.** Every fight is recorded. A backstab-threat net retrains itself in the background
while the bot is idle, and only installs a new net if it scores well on the newest fights it
wasn't trained on. Damage and reach are learned per weapon and per attack from hits taken.
Opponents are remembered by Steam ID, and play styles are picked per opponent and per weapon
by Thompson sampling.

**Behaves.** It puts down a red sign and re-signs after each duel, bows to the host on
arrival, fights only players (plus the arena's Havelmom summon, never other NPCs), waits its turn in the arena dugout when the arena
is busy, and heals between rounds.

**Shares the controller.** Your real controller is passed through the bot's virtual pad.
Click **L3 + R3** to hand control to the bot; touch anything to take it back.

Every behaviour rule is written down in [`linux-port/RULES.md`](linux-port/RULES.md), and each
one is enforced by an offline test.

## How it works

```
Dark Souls (Proton)                                                    Dark Souls (Proton)
        |                                                                       ^
  /proc/<pid>/mem       state.py           brain.py + nets        gamepad.py    |
  (read-only)  ----->   snapshot   ----->  decide (one   ----->   uinput Xbox --+
  memory.py             per tick           loop, 120 Hz)          360 pad
                                               |
                                               v
                                    recorder.py -> recordings/  -> retrain.py (when idle)
```

- **Memory reading** (`memory.py`, `offsets.py`, `state.py`): the original's pointer chains,
  read through `/proc/<pid>/mem` and re-resolved every tick, so loads and new phantoms never
  leave it reading stale addresses.
- **Steering** (`steer.py`): stick input is computed relative to the live camera yaw. The
  original needed Cheat Engine to freeze the camera and a compile flag per map; this needs one
  calibration.
- **Decisions** (`brain.py`, `character.py`): one deterministic single-threaded loop. A tick,
  both nets included, takes about 0.06 ms.
- **Nets** (`nets.py`, `threat.py`): small numpy MLPs with their input scaling stored next to
  the weights.

## Quick start

You need Linux with a writable `/dev/uinput`, Dark Souls: PTDE on Steam with Proton, and
Python 3.10+.

```bash
cd linux-port
python3 -m venv .venv && .venv/bin/pip install numpy evdev pillow
.venv/bin/python tests/test_brain.py && .venv/bin/python tests/test_handoff.py   # no game needed
.venv/bin/python scripts/probe.py --watch          # in a duel: check the memory reads
.venv/bin/python scripts/run_bot.py --dry-run      # print decisions, press nothing
.venv/bin/python scripts/run_bot.py                # play; L3+R3 hands control to the bot
.venv/bin/python scripts/run_bot.py --red-sign     # sign down and fight whoever summons it
```

Before the first real run, read the bring-up checklist in
[`linux-port/README.md`](linux-port/README.md). Dark Souls only reads controller slot 0, so the
bot's pad has to connect before your real controller. Steering also needs a one-time
calibration.

## Training data and models

The trained models are in [`linux-port/models/`](linux-port/models/):

| Model | What it does | Trained on |
|---|---|---|
| `threat.npz` | predicts an incoming backstab | the bot's own fights: 47 recordings, 1.26M ticks, 319 backstabs (held-out AUC 0.86) |
| `attack.npz`, `defense.npz` | the original's attack and backstab-detection nets, rebuilt | metal-crow's training files in [`Neural Nets/`](Neural%20Nets/) |
| `hit_damage.json` | damage and reach per weapon and attack | hits taken in recorded fights |

The raw fight recordings (`linux-port/recordings/`) and per-opponent records
(`linux-port/records/`) are not in this repository. They contain other players' Steam names and
IDs, so they stay on the machine that recorded them.

## Repository layout

| Path | What it is |
|---|---|
| [`linux-port/`](linux-port/) | the Linux bot: code, models, tools, tests and [full documentation](linux-port/README.md) |
| [`linux-port/RULES.md`](linux-port/RULES.md) | every behaviour rule and the test that enforces it |
| [`linux-port/NOTICE.md`](linux-port/NOTICE.md) | attribution and everything changed from the original |
| `Dark Souls AI C/`, `Dark Souls AI gui/`, `Neural Nets/`, `DARKSOULS_AI_Help.CT` | metal-crow's original Windows version |
| [`ORIGINAL_README.md`](ORIGINAL_README.md), [`WRITEUP.md`](WRITEUP.md) | metal-crow's original README and their writeup on how the Windows version was built |

## Credits and license

The original AI, its memory research, animation tables and training data are the work of
**metal-crow** (Giovanni Frank Crisanti):
[metal-crow/Dark-Souls-PvP-AI](https://github.com/metal-crow/Dark-Souls-PvP-AI). This fork
keeps its full history.

Both the original and the Linux port are licensed under the GNU GPL v2: see
[`LICENSE.md`](LICENSE.md) and [`linux-port/LICENSE`](linux-port/LICENSE).
