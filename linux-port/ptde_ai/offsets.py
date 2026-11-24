"""PTDE memory layout, ported from metal-crow/Dark-Souls-PvP-AI (CharacterStruct.h,
HelperUtil.c, Handler.c). Chains use Cheat Engine semantics; see ProcMem.chain.

Those chains were found against the 2015-2021 Steamworks build. Run
`python scripts/probe.py` in a duel to confirm they still resolve here.

metal-crow's naming: "Enemy" = the opponent, "Player" = the character the AI drives.
x/y are the horizontal plane (the game calls them x/z); rotation is radians in
[-pi, pi] like atan2.
"""

MODULE_BASE = 0x400000

ENEMY_BASE = MODULE_BASE + 0xF7DC70
PLAYER_BASE = MODULE_BASE + 0xF7D644
CAMERA_BASE = MODULE_BASE + 0xF7D6DC   # ChrFollowCam pointer (see CAMERA)
SELECTED_ITEM_BASE = MODULE_BASE + 0xF7F8F4

# field -> (chain, struct type). Types follow the sizes metal-crow reads.
ENEMY = {
    # metal-crow's position chains read 0 on this build (verified 2026-11-22); these go
    # through the same map-data -> position struct DS-Gadget uses (X 0x10, Z 0x18).
    "loc_x":            ([0x4, 0x4, 0x28, 0x1C, 0x10], "f32"),
    "loc_y":            ([0x4, 0x4, 0x28, 0x1C, 0x18], "f32"),
    "rotation":         ([0x4, 0x4, 0x28, 0x54, 0x34], "f32"),
    "animation_type":   ([0x4, 0x4, 0x28, 0x54, 0x1EC], "u16"),
    "hp":               ([0x4, 0x4, 0x2D4], "u32"),
    "r_weapon":         ([0x4, 0x4, 0x34C, 0x654, 0x1F8], "u32"),
    "l_weapon":         ([0x4, 0x4, 0x34C, 0x654, 0x1B8], "u32"),
    "hurtbox_active":   ([0x4, 0x0, 0xC, 0x3C, 0xF], "u8"),
    "anim_timer":       ([0x4, 0x4, 0x28, 0x18, 0x4DC], "f32"),
    "anim_timer2":      ([0x4, 0x4, 0x28, 0x18, 0x440], "f32"),
    "anim_id":          ([0x4, 0x4, 0x28, 0x18, 0x444], "i32"),
    "anim_id2":         ([0x4, 0x4, 0x28, 0x18, 0x3A8], "i32"),
    "anim_id3":         ([0x4, 0x4, 0x65C, 0x268, 0x770], "i32"),
    "velocity":         ([0x4, 0x4, 0x658, 0x5C, 0x3BC], "f32"),
    "stam_recovery":    ([0x4, 0x4, 0x170, 0x34C, 0x408], "i32"),
    "poise":            ([0x4, 0x4, 0x60, 0x8, 0x1C0], "f32"),
    # ChrIns +0x70 / +0x74 (DS-Gadget). Values from the user's CE table
    # "kill+unkill bosses and clone character slots.CT"; see CHR_TYPE / TEAM_TYPE.
    "chr_type":         ([0x4, 0x4, 0x70], "i32"),
    "team_type":        ([0x4, 0x4, 0x74], "i32"),
    "entity_id":        ([0x4, 0x4, 0x208], "u32"),     # event entity ID (NPCs; see NPC_TARGETS)
}

# Our own position from the character table (same struct as the enemy's): updates
# every frame. The PLAYER chains below only refresh ~3x/s (measured live 2026-11-22),
# which made the bot steer from positions up to a metre old.
SELF_POS = {
    "loc_x":            ([0x4, 0x0, 0x28, 0x1C, 0x10], "f32"),
    "loc_h":            ([0x4, 0x0, 0x28, 0x1C, 0x14], "f32"),
    "loc_y":            ([0x4, 0x0, 0x28, 0x1C, 0x18], "f32"),
}

# Same table, slot 0 = the local character.
SELF = {
    "chr_type":         ([0x4, 0x0, 0x70], "i32"),
    "team_type":        ([0x4, 0x0, 0x74], "i32"),
}

CHR_TYPE = {0: "alive player", 1: "white phantom", 2: "black phantom", 8: "hollow player"}
TEAM_TYPE = {0: "none", 1: "alive", 2: "white ghost", 3: "black ghost", 4: "hollow host",
             5: "wander ghost", 6: "enemy", 7: "boss", 8: "friend", 9: "angry friend",
             10: "decoy enemy", 11: "blood child", 12: "battle friend"}

# Who the bot may fight depends on what it is (see character.fightable):
#   bot is a red/black phantom (2): the host (0/8) and the host's white phantoms (1)
#   bot is the host or a white phantom: red/black phantoms (2)
# Anything else (e.g. chr type 9, like Big Hat Logan) is an NPC and is never engaged.
HOME_CHR_TYPES = {0, 8}          # in your own world, human or hollow
# As a red phantom (2) in the arena the bot fights anyone there, other reds included (user,
# 2026-11-23: "he only wanted the host... he was afraid of the red"). Who counts as being
# "there" is decided by the arena walls (character.pick_target). Never NPCs (5, 7).
FIGHTABLE_BY_SELF = {2: {0, 1, 2, 8}, 0: {2}, 8: {2}, 1: {2}}
FIGHTABLE_UNKNOWN_SELF = {0, 2, 8}   # own type unreadable: original conservative set
PLAYER_CHR_TYPES = {0, 1, 2, 8}

# The one exception to "never NPCs" (user, 2026-11-24: "target the NPC Havelmom summon
# despite the rules"). Matched by event entity ID (ChrIns + 0x208, found 2026-11-24), from the
# arena's event script event/m12_00_00_00.emevd.js:
#   1269420  Havelmom Great Club summon: event 18019083 warps it into the arena ("grand
#            entrance"), sets it to the Invader team, makes it INVINCIBLE, and disables it
#            again after 30 s. The bot may fight it.
#   1031001  Havelmom merchant in the dugout: never.
# Every other NPC stays off-limits.
NPC_TARGETS = {1269420: "Havelmom Great Club summon"}
NPC_NEVER = {1031001: "Havelmom merchant"}
ENTITY_ID_OFFSET = 0x208


def npc_target(chr_type, entity_id) -> str | None:
    """Name of the allowed NPC opponent this character is, or None."""
    if chr_type in PLAYER_CHR_TYPES or not entity_id or entity_id in NPC_NEVER:
        return None
    return NPC_TARGETS.get(entity_id)

# Quick-item slots, from DS-Gadget's CharData2 (player game data): AOB
# "A1 ? ? ? ? 8B 40 34 53 32 DB 85 C0" +1 -> 0x1378700, struct = [[base]+8].
CHAR_DATA2_BASE = MODULE_BASE + 0xF78700
QUICK_ITEMS = {f"item{i + 1}": ([0x8, 0x288 + 4 * i], "i32") for i in range(5)}
RED_SIGN_SOAPSTONE = 101         # goods id (also metal-crow's RedSoapstone)

PLAYER = {
    "loc_x":            ([0x3C, 0x330, 0x4, 0x20C, 0x3C0], "f32"),
    "loc_y":            ([0x3C, 0x330, 0x4, 0x20C, 0x3C8], "f32"),
    "loc_h":            ([0x3C, 0x330, 0x4, 0x20C, 0x3C4], "f32"),   # height (verified 2026-11-22)
    "rotation":         ([0x3C, 0x28, 0x1C, 0x4], "f32"),
    "animation_type":   ([0x288, 0xC, 0xC, 0x10, 0x41C], "u16"),
    "hp":               ([0x288, 0xC, 0x330, 0x4, 0x2D4], "u32"),
    "stamina":          ([0x288, 0xC, 0x330, 0x4, 0x2E4], "i32"),
    "r_weapon":         ([0x3C, 0x30, 0xC, 0x654, 0x1F8], "u32"),
    "l_weapon":         ([0x3C, 0x30, 0xC, 0x654, 0x1B4], "u32"),
    "anim_timer":       ([0x28, 0x0, 0x148, 0x4C8, 0x4DC], "f32"),
    "anim_timer2":      ([0x28, 0x0, 0x148, 0x4C8, 0x440], "f32"),
    "anim_id":          ([0x288, 0xC, 0x618, 0x28, 0x7B0], "i32"),
    "anim_id2":         ([0x3C, 0x28, 0x18, 0x8C, 0x1D4], "i32"),
    "anim_id3":         ([0x3C, 0x10C], "i32"),
    "ready_state":      ([0x3C, 0x30, 0xC, 0x20C, 0x7D2], "u8"),
    "locked_on":        ([0x3C, 0x170, 0x2C, 0x390, 0x128], "u8"),
    "two_handing":      ([0x28, 0x0, 0x148, 0x4C8, 0x0], "u8"),
    "poise":            ([0x28, 0x18, 0xE0, 0xC, 0x1C0], "f32"),
    "bleed_status":     ([0x3C, 0x308], "i32"),
    # Used by metal-crow's auto red-sign loop (Handler.c).
    "visual_status":    ([0x28, 0x0, 0x30, 0xC, 0x70], "i32"),
}

CAMERA = {
    # metal-crow's camera base (0x135CDD4) is null on this build. DS-Gadget's
    # ChrFollowCam (AOB "D9 45 08 A1 ? ? ? ? 51 D9 1C 24 50", +4) resolves to
    # 0x137D6DC; struct = [[[base]+0x3C]+0x60]. 0x144 is the same yaw field
    # metal-crow locked with Cheat Engine.
    "yaw":       ([0x3C, 0x60, 0x144], "f32"),
    "pitch":     ([0x3C, 0x60, 0x140], "f32"),
    "pos_x":     ([0x3C, 0x60, 0x100], "f32"),
    "pos_z":     ([0x3C, 0x60, 0x108], "f32"),
}

SELECTED_ITEM = ([0x67C, 0xC, 0x18, 0x730, 0x2D4], "i32")
