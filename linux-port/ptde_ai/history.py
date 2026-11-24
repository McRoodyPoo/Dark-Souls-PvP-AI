"""Rolling memories the bot keeps between ticks (port of Memory.c).

Index 0 is the newest entry, like the C arrays. Times are milliseconds.
"""
from __future__ import annotations

from collections import deque

from . import tables as T

SUBROUTINE_SELF_LEN = 20
ANIM_ID_ENEMY_LEN = 20
ANIM_TYPE_ENEMY_LEN = 100   # sampled every 100 ms -> 10 s of history
DISTANCE_LEN = 50           # sampled every 100 ms -> 5 s
AI_HP_LEN = 20              # sampled every 500 ms -> 10 s


class History:
    def __init__(self):
        # entries are (kind, id) with kind 'def' or 'atk'
        self.last_subroutines_self = deque([("none", 0)] * SUBROUTINE_SELF_LEN, maxlen=SUBROUTINE_SELF_LEN)
        self.last_anim_ids_enemy = deque([0] * ANIM_ID_ENEMY_LEN, maxlen=ANIM_ID_ENEMY_LEN)
        self.last_anim_types_enemy = deque([0] * ANIM_TYPE_ENEMY_LEN, maxlen=ANIM_TYPE_ENEMY_LEN)
        self.distances = deque([0.0] * DISTANCE_LEN, maxlen=DISTANCE_LEN)
        self.ai_hp = deque([0] * AI_HP_LEN, maxlen=AI_HP_LEN)
        self._t_anim_type = self._t_dist = self._t_hp = -1e9

    def push_subroutine_self(self, tagged: tuple[str, int]) -> None:
        kind, sub_id = tagged
        self.last_subroutines_self.appendleft((kind, int(sub_id)))

    def push_anim_id_enemy(self, aid: int) -> bool:
        """Returns True when this is a new animation (used to resync the timer)."""
        if aid != self.last_anim_ids_enemy[0]:
            self.last_anim_ids_enemy.appendleft(aid)
            return True
        return False

    def push_anim_type_enemy(self, anim_type: int, now_ms: float) -> None:
        if now_ms - self._t_anim_type >= 100 and (anim_type != self.last_anim_types_enemy[0] or anim_type == 0):
            self.last_anim_types_enemy.appendleft(anim_type)
            self._t_anim_type = now_ms

    def push_distance(self, d: float, now_ms: float) -> None:
        if now_ms - self._t_dist >= 100:
            self.distances.appendleft(d)
            self._t_dist = now_ms

    def push_ai_hp(self, hp: int, now_ms: float) -> None:
        if now_ms - self._t_hp >= 500:
            self.ai_hp.appendleft(hp)
            self._t_hp = now_ms

    def enemy_stamina_estimate(self, enemy) -> int:
        """Stamina isn't synced over the network, so estimate it from recent animations."""
        est = 192
        for at in reversed(self.last_anim_types_enemy):
            if at in (T.Anim.Backstep_1H, T.Anim.Backstep_2H):
                est -= 19
            elif at in T.DODGE_ANIM:
                est -= 28
            elif T.ATTACK_ANIM.get(at, 0):
                # Original passed the enemy's *current* animation here instead of `at`.
                est -= stamina_drain_for_attack(enemy.r_weapon, at)
            elif at in (T.Anim.Nothing, T.Anim.Shield_Held_Up, T.Anim.Shield_Held_Up_walking):
                est += enemy.stam_recovery // 10
            est = max(-40, min(192, est))
        return est


STAMINA_DRAIN_DEFAULT = 25  # original had no default (undefined behavior in C)


def stamina_drain_for_attack(weapon_id: int, anim_type: int) -> int:
    behavior = T.WEAPON_TYPE_TO_BEHAVIOR.get(weapon_id // 1000)
    if behavior is None:
        return STAMINA_DRAIN_DEFAULT
    return T.STAMINA_DRAIN.get(behavior * 1000 + anim_type, STAMINA_DRAIN_DEFAULT)


def poise_damage_for_attack(weapon_id: int, anim_type: int) -> float:
    anim_type = T.COMPRESS_ANIM.get(anim_type, anim_type)
    return T.POISE_DAMAGE.get((weapon_id // 1000) * 1000 + anim_type, 0.0)


def poise_damage_known(weapon_id: int, anim_type: int) -> float | None:
    """Like poise_damage_for_attack, but None when the table doesn't have the attack."""
    anim_type = T.COMPRESS_ANIM.get(anim_type, anim_type)
    return T.POISE_DAMAGE.get((weapon_id // 1000) * 1000 + anim_type)


def is_weapon_shield(weapon_id: int) -> bool:
    return weapon_id // 100000 in (14, 15, 90)
