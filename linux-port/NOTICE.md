# Notice

ptde-pvp-bot is a fork and Python port of
[metal-crow/Dark-Souls-PvP-AI](https://github.com/metal-crow/Dark-Souls-PvP-AI) by metal-crow
(Giovanni Frank Crisanti), a Windows/C PvP AI for Dark Souls: Prepare to Die Edition.

The original is licensed under the GNU General Public License, version 2
(Copyright (c) 2015 Giovanni Frank Crisanti). This project is a derivative work and is
distributed under the same license: see [`LICENSE`](LICENSE). This port lives in a GitHub
fork of the original repository; the original source is the rest of this repository
([`../`](../)), unmodified, with its own [`LICENSE.md`](../LICENSE.md), and its full commit
history is kept. Several modules (`ptde_ai/offsets.py`, `brain.py`, `character.py`,
`geometry.py`, `history.py`, `redsign.py`) are direct ports of the original's C code, and
`ptde_ai/tables.py` is generated from its animation and weapon tables by
`tools/convert_tables.py`.

Python port and extensions: Copyright (c) 2026 the ptde-pvp-bot contributors.

## What changed from the original

**Platform**

- Runs natively on Linux against the game under Steam/Proton, in Python, instead of Windows/C.
- Game memory is read **read-only** through `/proc/<pid>/mem` (`memory.py`, `process.py`),
  using the original's pointer chains; nothing is written to the game, no code is injected,
  and Cheat Engine is not needed. Pointers are re-resolved every tick.
- The vJoy driver is replaced by a uinput virtual Xbox 360 pad (`gamepad.py`), and the user's
  real controller is passed through it (`realpad.py`), with an L3+R3 handoff between the
  player and the bot (`handoff.py`).
- The Cheat Engine camera lock is replaced by camera-relative steering from the live camera
  yaw (`steer.py`, calibrated once into `calibration.json`).
- One deterministic single-threaded loop instead of racing threads (`scripts/run_bot.py`),
  with fight recording (`recorder.py`) and hot reload between fights.

**Bug fixes in the ported logic** (details in `README.md`): the shipped attack and defense nets
were identical and took the wrong number of inputs; a misaligned training file; instinct
attacks, mid-swing dodges, omnistep and pivot backstabs not working as intended; distance
computed from absolute coordinates; shared dodge/attack history ids; and others.

**Models**

- The FANN nets are replaced by small numpy MLPs with their input scaling saved alongside the
  weights (`nets.py`, `tools/train_nets.py`), retrained from the original's training data.
- A new backstab-threat net (`threat.py`, `tools/train_threat.py`) uses bot-relative inputs and
  is trained on the bot's own recorded fights; it can retrain itself in the background while
  idle and only installs a new net that scores well on held-out recent fights (`retrain.py`).
- Per-weapon, per-attack damage and reach learned from hits taken (`hitmemory.py`,
  `models/hit_damage.json`).

**Behaviour** (specified rule by rule, each with a test, in [`RULES.md`](RULES.md))

- A backstab-focused build (kick-input backstabs, races to the back, run backstabs, wake-up
  pressure, turtle and caster handling, poise trades, projectile and fire-tempest dodging),
  with the original's poke style available via `--allow-r1`.
- Per-opponent style adaptation: opponents are identified by SteamID64 (`players.py`), fights are
  logged (`fightlog.py`), and styles are chosen by Thompson sampling over per-opponent,
  per-weapon and pooled records (`adapt.py`).
- Arena-mod etiquette: auto red-signing gated on memory state instead of blind sleeps
  (`redsign.py`), dugout escape with the Traversal Trinket and a dugout queue when the arena is
  busy (`escape.py`), the arena's walkable area for wall-aware positioning (`arena_map.py`,
  `arena.json`), bowing to the host on arrival (`greet.py`), and healing between rounds
  (`heal.py`).
- HUD screen checks to tell which quick item is selected and whether a sign prompt is showing
  (`hud.py`, `assets/hud/`), window-focus detection (`focus.py`), and a stuck-position detector
  (`freeze.py`).
- Offline test suites (`tests/test_brain.py`, `tests/test_handoff.py`) that need no game.
