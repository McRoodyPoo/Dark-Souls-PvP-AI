# [BOT]King: the rules

Everything the user has told the bot to do or not do, in their words where possible, with
the test that enforces each one. Run the tests before every deploy:

```bash
.venv/bin/python tests/test_brain.py && .venv/bin/python tests/test_handoff.py
```

If a change makes one of these tests fail, the change is wrong, not the test. A rule is only
changed or removed when the user says so, and then this file changes in the same edit.

Tests live in `tests/test_brain.py` unless marked `(handoff)` = `tests/test_handoff.py`.

## Hard limits (never)

| Rule | Test |
|---|---|
| Never attack NPCs, with one exception: the Havelmom Great Club summon (entity 1269420 in the arena's event script; the Havelmom merchant, 1031001, never) is an opponent like any player, and counts as one for the dugout queue. *(exception added 2026-11-24)* | `test_never_engages_npcs`, `test_fights_the_right_side`, `test_fights_the_havelmom_summon_but_no_other_npc` |
| Memory access is read-only: nothing writes to the game. | `test_memory_is_read_only` |
| Never put the bot's sign down in the dugout. In our own dugout, only press "use item" when the screen shows the Traversal Trinket. *(2026-11-24)* | `test_never_uses_the_soapstone_in_the_dugout`, `test_unreadable_screen_never_presses_in_our_dugout` |
| Never R1 on wake-up. | `test_never_r1_on_wake_up` |

## Manners and bookkeeping

| Rule | Test |
|---|---|
| Bow to the host after being summoned (and fight back if attacked during the greeting). Face the opponent first (unlocked, with the stick), and only bow once facing them; never bow facing away. Only a swing aimed at the bot counts as an attack: arriving while the host fights someone else isn't one. *(2026-11-24)* | `test_greeting_faces_then_bows_and_aborts_on_attack`, `test_arriving_into_someone_elses_fight_is_not_an_attack`, `test_faces_the_opponent_before_bowing` |
| Log the Steam ID of everyone it fights; `tools/defeated.py` lists who lost to it. | `test_fight_log_records_who_lost_to_the_bot` |
| 1 HP means the round is over (for the opponent: a win; for the bot: a loss), not just death. Missed 1 HP wins can be recovered from the recordings with `tools/recover_wins.py`. *(2026-11-24)* | `test_one_hp_ends_the_round` |
| Record all fights (`recordings/`). | not unit-tested; checked by `tools/replay_check.py` |
| Dismiss a "Failed to join session" box with A before trying the sign again. | `test_red_sign_flow` |
| Don't fiddle with the sign while waiting: if the "Check your Summon Sign" prompt is on screen, the sign is down, so press nothing (a dialog open when a summon arrives may make it fail). *(2026-11-24)* | `test_sign_check_looks_before_pressing` |

## Build: Havel + Great Club (`backstab_only`, the default)

| Rule | Test |
|---|---|
| No R1s. Backstabs only. Exceptions: jumping attacks and rolling ghost strikes against a turtle. | `test_havel_build_backstab_only`, `test_turtles_get_kicks_jumping_attacks_and_roll_ghosts` |
| Backstab with the kick input ("kick confirm": the kick input used for the backstab attempt, an option select). | `test_havel_build_backstab_only`, `test_havel_kick_input_is_reliable`, `test_lines_up_at_their_back_then_kicks` |
| If the kick lands but isn't a backstab, go for the back again. | `test_havel_build_backstab_only` |
| If a kick lands clean (took HP off them, their shield wasn't up), race for their back for the backstab: unlocked, block then sprint, even while they're still stunned. A kick into a raised shield doesn't count. *(2026-11-24)* | `test_clean_kick_starts_a_race_for_the_back` |
| Aim every kick at the opponent (unlock first: kicks aim freely unlocked), leading a moving target by 0.15 s. A kick backstab is one motion: kick > unlock > block and sprint together from the first instant (max speed ASAP) > let go of block > circle to their back > backstab. Every kick that isn't already a backstab chains into it. *(2026-11-24)* | `test_kicks_are_aimed_at_them_and_chain_into_the_race`, `test_kicks_lead_a_moving_target` |
| Turtles (standing still, shield up, facing you): mix up kicks (mostly), jumping attacks (stick at them + R2) and rolling ghost strikes, at random intervals; don't circle a turtle. *(2026-11-24)* | `test_turtles_get_kicks_jumping_attacks_and_roll_ghosts` |
| The no-cast loop (animation type 132, the failed-cast pose some players spam to turn with you): don't angle for the backstab, hit them directly with the turtle mix-up. *(2026-11-24)* | `test_no_cast_loop_gets_hit_not_circled` |
| A race for their back that gains nothing in 0.8 s means they just turn with you: stop racing them for 6 s and hit them with the turtle mix-up instead (except after a landed kick or with their back turned). *(2026-11-24)* | `test_stops_racing_someone_who_turns_with_us` |
| Never press R2 except as a jumping attack: stick centred >= 2 game frames (70 ms), then pushed at them with R2 on the same tick, and never while still in an animation (that buffers a plain heavy). *(2026-11-24)* | `test_r2_only_ever_as_a_flicked_jump` |
| Campers and casters (standing still without a shield up, or casting / buffing, 2-15 m away): charge them: unlock, block + sprint straight at them, then race to the back. Neutral spacing never assumes more than 3.5 m of melee reach (a long-range R2 / projectile once kept the bot at 6-7 m from a pyromancer). *(2026-11-24)* | `test_engages_a_camping_caster` |
| Wake-up machine: when they're down (after your backstab, kickstunned, staggered), get to the spot just behind them and wait there facing them, unlocked, stick centred; kick their back the very frame they wake up (with a learned wake-up time, pressed just before so it lands on that frame). No pressing while they're on the floor, no walking into them, no backing off. *(2026-11-24)* | `test_wake_up_machine_waits_at_their_back`, `test_kicks_their_back_the_frame_they_wake_up` |
| Instead of a roll-R1, do a ghost strike. | `test_havel_build_backstab_only` |
| Two-handed basically always, except one-handed after a toggle escape while racing for a backstab (toggling reaches max speed sooner, like blocking). | `test_two_handed_unless_toggle_escaped_and_racing` |
| Don't roll when poise can take the hit: trade for a backstab. No damage cap, just keep at least 20% HP after the hit. | `test_havel_trades_with_poise_and_commits_to_the_race` |

Other builds (`--allow-r1`): plain R1 pokes from the front, no kicks in neutral, stance cancel
(Y then R1) on oki. `test_pokes_are_plain_r1s_and_no_kicks_in_neutral`,
`test_kick_in_neutral_stance_cancel_on_oki`. Stance cancel is off on Havel (it swapped the
club's grip and got punished).

## Neutral and getting behind them

| Rule | Test |
|---|---|
| Don't make pivot backstabs the only strategy. From neutral: footsies, threatening with forward movement and baiting theirs. | `test_neutral_footsies_race_only_when_they_sit_still`, `test_footsies_threaten_and_bait` |
| Punish their forward movement with a run backstab: locked-on sprint at them, kick on contact (latency dependent). | `test_run_backstab_punishes_their_forward_movement` |
| Race for their back unlocked from the start. Block briefly to reach max speed sooner, then sprint. | `test_race_is_unlocked_blocks_then_sprints`, `test_race_gets_round_a_neutral_host_and_backstabs` |
| Locked-on strafing is for defensive movement. | `test_counter_strafe_keeps_lock_and_runs_through`, `test_counter_strafe_follows_their_side` |
| Commit; don't wiggle in their face. | `test_havel_trades_with_poise_and_commits_to_the_race` |
| Take the backstabs that are there (backs facing it). Backstab range 1.3 m (was 1.15 m: a bit more for players bent over), line-up range 1.45 m. *(range updated 2026-11-24)* | `test_takes_backstab_when_behind`, `test_lines_up_at_their_back_then_kicks` |
| If the opponent's back is facing you (within 60° of straight behind them, 5 m or closer), go for the backstab even if it means taking damage: no rolling, strafing or turning away, chase their back even mid-swing, kick once they're out of the swing. Only exception: a hit known to be lethal. *(2026-11-24)* | `test_back_turned_means_go_for_the_backstab` |

## Defence

| Rule | Test |
|---|---|
| Roll less. | `test_close_and_facing_us_no_rolling`, `test_sprint_guard_no_roll_from_a_cut_short_sprint`, `test_one_roll_per_swing` |
| On an incoming pivot (unlocked) backstab, unlock and turn to face them. | `test_unlocks_and_turns_on_an_incoming_pivot` |
| Hold block while down (face them faster on wake-up). When they're close enough to try a backstab, sometimes wait out the stun/backstab animation instead, to mix up their timing. | `test_holds_block_while_down_and_sometimes_waits_it_out` |
| Remember stamina. | `test_stamina_rules` |
| Heal between fights, including between 1 HP rounds: Start, then A until the Divine Blessing is used (only one), then back out of the menu. Only between fights (waiting in the dugout, the round is over at 1 HP, or no opponent within 15 m; never mid-fight, whatever the distance) and nobody's swinging at you; at 1 HP with 2+ others around, go to the dugout first and heal there. *(2026-11-24)* | `test_heals_between_fights_with_one_divine_blessing`, `test_heals_only_between_fights` |
| Know what can reach you: reach is learned per weapon and per attack from where the attack *started* (lunges close distance). Unknown weapons: lunges count as ~5 m, spears/halberds +1 m. *(2026-11-24: "stands there and gets hit by spears")* | `test_reach_per_attack_lunges_spears_and_spells` |
| Magic: a projectile (sorcery, fireball, lightning spear, arrow) aimed at you: sidestep across its path while they wind up, then roll so the i-frames cover the moment it lands (cast + release + distance / speed; sorcery timing learned from hits). Fire whip: get out of the cone. Don't stand in it. *(2026-11-24)* | `test_sidesteps_projectiles_and_flames_not_melee_at_range`, `test_rolls_when_the_sorcery_lands_not_when_it_is_cast` |
| Fire tempest / fire storm: roll spam directly away from the caster, toward the most open floor, until clear (8.5 m) or it's over; run if out of stamina. Never charge or chase a backstab into it. *(2026-11-24)* | `test_fire_tempest_roll_spam_away_toward_open_floor` |
| Fire surge: keep out of its range (hold at 6.5 m+, don't approach); inside its range (5.5 m, aimed at you), roll away toward open floor. *(2026-11-24)* | `test_fire_surge_keep_out_or_roll_away` |
| Corner campers (standing still within 2.2 m of an arena corner): back up and throw poison throwing knives until they leave their spot by 0.2 m (or come at you), then switch back to the soapstone. Every throw is locked on: hold R3, check the lock took, re-lock if it drops, never throw unlocked. Never sign or use the trinket with the knives selected. *(2026-11-24)* | `test_corner_camper_gets_poison_knives_until_they_leave` |

## Arena

| Rule | Test |
|---|---|
| Stay off the walls. A wall right behind you protects your back to a degree. | `test_spacing_drifts_off_the_walls`, `test_walls_change_tactics` |
| Fight anyone inside the arena walls (host, whites, other reds), nearest first. Ignore players outside the walls. | `test_targets_anyone_in_the_arena_nearest_first`, `test_picks_the_host_not_the_other_red_phantom` |
| If the target is outside the walls, wait in the middle. | `test_waits_in_the_middle_when_they_are_outside_the_walls` |
| At 1 HP (round lost), Traversal Trinket to the dugout, but only if 2 or more other players are in the world. One-on-one, stay. | `test_one_on_one_stays_at_1hp`, `test_arena_etiquette_dugout_queue` |
| In the dugout: wait while two are in the arena (inside the walls, at any HP: a player at 1 HP who hasn't left still counts); go back in once one of them has actually left and stayed gone for 3 s (or nobody's there). *(updated 2026-11-24)* | `test_arena_etiquette_dugout_queue` |
| Summoned and there are already two players in the arena: trinket straight to the dugout and wait your turn. Do not bow first, just get out. Only on arrival; never leave mid-round because a third player shows up. *(2026-11-24)* | `test_summoned_into_a_round_waits_its_turn` |
| When summoned into the host world's dugout, get out of it. | `test_unreadable_screen_never_presses_in_our_dugout` (summoned case), `test_dugout_escape` |

## Adapting to each opponent *(2026-11-24)*

| Rule | Test |
|---|---|
| Remember each opponent by Steam ID and play what has worked against them (`records/profiles.json`; seeded with `tools/seed_profiles.py`). Styles only move existing dials (how soon to race, how eager the run backstab is, footsie distances, how often to stay down on wake-up): every rule in this file still applies. | `test_profiles_learn_what_works_per_opponent`, `test_adapter_styles_reach_the_brain_and_score_wakeups` |
| Players switch setups: results are also kept per player per main weapon, and for everyone per weapon. A known player on a new weapon starts from their record and from what works against that weapon. Quick swaps don't count; a weapon held 5 s does. *(2026-11-24)* | `test_profiles_by_weapon_setup`, `test_setup_switch_needs_a_held_weapon` |
| Keep mixing styles (never fully predictable); new opponents start from the record against everyone. | `test_profiles_learn_what_works_per_opponent` |
| An exchange's score is the bot's share of the damage, counted until 1.5 s after the first real hit so poise trades resolve. | `test_exchange_scoring` |
| Backstabs get a bigger reward: an exchange where the bot lands one scores at least 0.9, one where it gets backstabbed at most 0.1. *(2026-11-24)* | `test_backstabs_get_a_bigger_reward` |
| What's learned on each player is absorbed by the general model: every result also counts for the record against everyone, which keeps ~1000 exchanges (a player's own record keeps ~70, since people change). *(2026-11-24)* | `test_player_results_feed_the_general_record` |
| The backstab net retrains itself on new fights: only while idle at home, at most every 6 h, at lowest CPU priority, frozen during fights or while you play, and a new net is installed only if it scores well on the newest fights it wasn't trained on. | `test_background_retrain_pauses_during_fights`, `test_retrain_tests_on_the_newest_fights` |

## Controls

| Rule | Test |
|---|---|
| Forgiving L3+R3 handoff to the bot (both clicked within 0.4 s); touching anything takes back over. | `test_sloppy_combo_still_counts`, `test_combo_hands_to_bot_and_any_input_takes_back` (handoff) |

## Deploying

- The user has said "just deploy it": restart/reload the bot after changes without asking, but never mid-escape or mid sign press. The bot only hot-reloads when settled at home, and only if the new code imports cleanly.

## Retired or rejected (don't bring these back)

- **Skid rule**: skids give backstab immunity, but the user said it's minor context for defence, not a style. Removed.
- **"Stand off" waiting in the arena**: rejected in favour of the dugout queue.
- **12 s dugout wait "giving a challenger a turn"**: made the bot look stuck. Removed.
- **Kick zone / kickstun confirm**: a misreading of "kick confirm". Kick confirm = use the kick input for backstab attempts.
- **R1 pokes and punishes on Havel**: no R1s on this build.
