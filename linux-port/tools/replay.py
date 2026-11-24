"""Replay a recording through the current brain (live config, nets on) and print its decisions.

  python tools/replay.py <recording> <t0_ms> <t1_ms> [print every Nth tick]
"""
import sys, math, random, json
sys.path[:0]=['.','tests']
from ptde_ai.recorder import load
from ptde_ai.state import Snapshot
from ptde_ai.brain import Brain, Config
from ptde_ai.character import CharacterTracker
from ptde_ai.hitmemory import HitMemory
from ptde_ai.steer import Steering
from ptde_ai.nets import MODEL_DIR, MLP
from ptde_ai.arena_map import ArenaMap
from test_brain import Pad
path, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 1
rows=[r for r in load(path) if 'mode' in r]
hits=HitMemory(path=None); hits.table=json.load(open('models/hit_damage.json'))
net=lambda n: MLP.load(MODEL_DIR/f"{n}.npz")
b=Brain(Steering(0.0,1.0), net("defense"), net("attack"), Config(), rng=random.Random(0), hits=hits)
b.arena=ArenaMap.from_arena(json.load(open('arena.json')))
me,them=CharacterTracker(True,2.5),CharacterTracker(False,6.0); them.timings=hits
n=0
for r in rows:
    if r['t']>t1: break
    e=them.update(Snapshot(values=r['e']), b.history, r['t']); p=me.update(Snapshot(values=r['p']), b.history, r['t'])
    pad=Pad(); b.tick(p,e,0.0,r['t'],pad)
    if r['t']>=t0:
        n+=1
        if n%every==0 or b.log:
            inst=b.last_instinct
            print(f"{r['t']/1000:.3f} e {e.sub.name[:12]:12s} d {math.hypot(p.loc_x-e.loc_x,p.loc_y-e.loc_y):.2f} inst {inst.priority.name}/{inst.defense.name}/{inst.attack.name} dodge {b.dodge_type.name if b.in_dodge() else '-'} atk {b.attack_type.name if b.in_attack() else '-'} p {p.sub.name[:8]} live {r.get('attack')}/{r.get('dodge')} {b.log}")
    b.log.clear()
