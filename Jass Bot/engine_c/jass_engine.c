/*
 * Fast C engine for Jass Differenzler.
 * Compiles to shared library, loaded via ctypes from Python.
 *
 * Card ID = suit * 9 + value_index (0-35)
 *   Suits: herz=0, ecke=1, schaufel=2, kreuz=3
 *   Values: 6=0, 7=1, 8=2, 9=3, 10=4, U=5, O=6, K=7, A=8
 *
 * Hands represented as uint64_t bitmasks (bits 0-35).
 *
 * Build: gcc -O3 -march=native -shared -o jass_engine.dll jass_engine.c
 */

#include <stdint.h>
#include <string.h>

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

typedef uint64_t hand_t;

/* ===== Lookup tables ===== */

static int SUIT[36];           /* card -> suit (0-3) */
static int VALIDX[36];         /* card -> value index (0-8) */
static int PTS[36][2];         /* [card][0=normal,1=trump] -> points */
static int STR[36][2];         /* [card][0=normal,1=trump] -> strength */
static hand_t SUIT_MASK[4];   /* bitmask for each suit */
static int PUUR[4];            /* card ID of trump Under for each suit */
static int initialized = 0;

static const int PTS_NORMAL[9] = {0,0,0, 0,10,2,3,4,11};
static const int PTS_TRUMP[9]  = {0,0,0,14,10,20,3,4,11};
/* Non-trump strength: 6<7<8<9<10<U<O<K<A = 0..8 */
static const int STR_NORMAL[9] = {0,1,2,3,4,5,6,7,8};
/* Trump strength: 6<7<8<10<O<K<A<9<U */
/* By value_index:  6=0,7=1,8=2,9=7,10=3,U=8,O=4,K=5,A=6 */
static const int STR_TRUMP[9]  = {0,1,2,7,3,8,4,5,6};

EXPORT void init_tables(void) {
    int c, s;
    if (initialized) return;
    for (s = 0; s < 4; s++) {
        SUIT_MASK[s] = 0;
        PUUR[s] = s * 9 + 5; /* U is value_index 5 */
    }
    for (c = 0; c < 36; c++) {
        SUIT[c] = c / 9;
        VALIDX[c] = c % 9;
        PTS[c][0] = PTS_NORMAL[c % 9];
        PTS[c][1] = PTS_TRUMP[c % 9];
        STR[c][0] = STR_NORMAL[c % 9];
        STR[c][1] = STR_TRUMP[c % 9];
        SUIT_MASK[c / 9] |= (1ULL << c);
    }
    initialized = 1;
}

/* ===== Inline helpers ===== */

static inline int card_points(int card, int trump) {
    return PTS[card][SUIT[card] == trump ? 1 : 0];
}

static inline int card_strength(int card, int trump) {
    if (SUIT[card] == trump)
        return 100 + STR[card][1];
    return STR[card][0];
}

static inline int popcount64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) { count++; x &= x - 1; }
    return count;
#endif
}

static inline int ctz64(uint64_t x) {
#ifdef __GNUC__
    return __builtin_ctzll(x);
#else
    int n = 0;
    if (!(x & 0xFFFFFFFF)) { n += 32; x >>= 32; }
    if (!(x & 0xFFFF)) { n += 16; x >>= 16; }
    if (!(x & 0xFF)) { n += 8; x >>= 8; }
    if (!(x & 0xF)) { n += 4; x >>= 4; }
    if (!(x & 0x3)) { n += 2; x >>= 2; }
    if (!(x & 0x1)) { n += 1; }
    return n;
#endif
}

/* Iterate over set bits */
#define FOR_EACH_CARD(hand, card) \
    for (hand_t _tmp = (hand); _tmp; _tmp &= _tmp - 1) { \
        int card = ctz64(_tmp);

#define END_FOR }

/* ===== Core functions ===== */

EXPORT int trick_winner(int *players, int *cards, int n, int lead_suit, int trump) {
    int best_p = players[0];
    int best_c = cards[0];
    int best_str = card_strength(best_c, trump);
    int best_trump = (SUIT[best_c] == trump);
    int i;

    for (i = 1; i < n; i++) {
        int c = cards[i];
        int cs = SUIT[c];
        int is_t = (cs == trump);
        int str = card_strength(c, trump);

        if (is_t && !best_trump) {
            best_p = players[i]; best_c = c; best_str = str; best_trump = 1;
        } else if (is_t && best_trump && str > best_str) {
            best_p = players[i]; best_c = c; best_str = str;
        } else if (!is_t && !best_trump && cs == SUIT[best_c] && str > best_str) {
            best_p = players[i]; best_c = c; best_str = str;
        }
    }
    return best_p;
}

EXPORT int trick_points_sum(int *cards, int n, int trump) {
    int total = 0, i;
    for (i = 0; i < n; i++) total += card_points(cards[i], trump);
    return total;
}

/* Legal moves as bitmask (simplified: follow suit only, for simulation speed) */
static inline hand_t legal_moves_fast(hand_t hand, int lead_suit) {
    if (lead_suit < 0) return hand;
    hand_t follow = hand & SUIT_MASK[lead_suit];
    return follow ? follow : hand;
}

/* ===== Smart rollout policy ===== */

/* Find the highest strength of cards already played in the current trick */
static inline int table_best_str(int *tc_cards, int tc_n, int lead_suit, int trump) {
    int best = -1;
    int i;
    for (i = 0; i < tc_n; i++) {
        int c = tc_cards[i];
        int cs = SUIT[c];
        int s = card_strength(c, trump);
        /* Only lead-suit or trump cards can be winning */
        if (cs == trump || cs == lead_suit) {
            if (s > best) best = s;
        }
    }
    return best;
}

/* Would card c beat all cards currently on the table? */
static inline int beats_table(int c, int *tc_cards, int tc_n, int lead_suit, int trump) {
    int cs = SUIT[c];
    int s = card_strength(c, trump);
    int i;
    for (i = 0; i < tc_n; i++) {
        int oc = tc_cards[i];
        int os = SUIT[oc];
        int ostr = card_strength(oc, trump);
        /* Trump beats non-trump */
        if (os == trump && cs != trump) return 0;
        /* Same suit: must be stronger */
        if (os == cs && ostr >= s) return 0;
        /* They play trump, we play trump but weaker */
        if (os == trump && cs == trump && ostr >= s) return 0;
    }
    /* Off-suit non-trump can't win unless it's lead suit */
    if (cs != trump && cs != lead_suit && tc_n > 0) return 0;
    return 1;
}

/* Count cards of a suit in hand */
static inline int suit_count(hand_t hand, int suit) {
    return popcount64(hand & SUIT_MASK[suit]);
}

/* Highest card in a suit in hand, or -1 */
static inline int highest_in_suit(hand_t hand, int suit, int trump) {
    hand_t suited = hand & SUIT_MASK[suit];
    if (!suited) return -1;
    int best = -1, best_str = -1;
    FOR_EACH_CARD(suited, c)
        int s = card_strength(c, trump);
        if (s > best_str) { best_str = s; best = c; }
    END_FOR
    return best;
}

/* Lowest card in a suit in hand, or -1 */
static inline int lowest_in_suit(hand_t hand, int suit, int trump) {
    hand_t suited = hand & SUIT_MASK[suit];
    if (!suited) return -1;
    int best = -1, best_str = 999;
    FOR_EACH_CARD(suited, c)
        int s = card_strength(c, trump);
        if (s < best_str) { best_str = s; best = c; }
    END_FOR
    return best;
}

/*
 * Smart card picker for simulation rollouts.
 *
 * Context:
 *   hand: bitmask of player's cards
 *   lead_suit: suit led this trick (-1 if leading)
 *   trump: trump suit
 *   gap: target - current_points (positive = need more, negative = over target)
 *   tc_cards: cards already played in this trick
 *   tc_n: number of cards on table
 *   trick_num: 0-indexed trick number
 *   total_tricks: total tricks in game (usually 9)
 *   all_hands: all 4 players' hands (for estimating if card will hold)
 */
static int pick_card_smart(
    hand_t hand, int lead_suit, int trump, int gap,
    int *tc_cards, int tc_n,
    int trick_num, int total_tricks,
    hand_t all_hands[4], int player
) {
    hand_t legal = legal_moves_fast(hand, lead_suit);
    if (!legal) return -1;
    if (popcount64(legal) == 1) return ctz64(legal);

    int is_leading = (lead_suit < 0);
    int is_last_trick = (trick_num == total_tricks - 1);
    int tricks_left = total_tricks - trick_num;
    int abs_gap = gap < 0 ? -gap : gap;

    /* === LAST TRICK: +5 bonus decision === */
    if (is_last_trick && !is_leading) {
        if (gap >= 3 && gap <= 7) {
            /* Need ~5 points — try to win for the bonus */
            int best = -1, best_str = -1;
            FOR_EACH_CARD(legal, c)
                if (beats_table(c, tc_cards, tc_n, lead_suit, trump)) {
                    int s = card_strength(c, trump);
                    /* Pick cheapest winner */
                    if (best < 0 || s < best_str) { best_str = s; best = c; }
                }
            END_FOR
            if (best >= 0) return best;
        } else if (gap <= -3) {
            /* Over target — avoid winning */
            int best = -1, best_score = 999999;
            FOR_EACH_CARD(legal, c)
                int pts = card_points(c, trump);
                int s = card_strength(c, trump);
                int wins = beats_table(c, tc_cards, tc_n, lead_suit, trump);
                /* Prefer non-winning low cards */
                int score = pts * 1000 + s + (wins ? 50000 : 0);
                if (score < best_score) { best_score = score; best = c; }
            END_FOR
            return best;
        }
    }

    /* === SHEDDING: gap <= 0 (at or over target) === */
    if (gap <= 0) {
        if (is_leading) {
            /* Lead: dump lowest non-trump card from longest suit.
             * Keep trump and short suits for control. */
            int best = -1, best_score = 999999;
            FOR_EACH_CARD(legal, c)
                int cs = SUIT[c];
                int pts = card_points(c, trump);
                int s = card_strength(c, trump);
                int len = suit_count(hand, cs);
                /* Strongly avoid leading trump when shedding */
                int trump_pen = (cs == trump) ? 30000 : 0;
                /* Prefer dumping from long suits; keep singletons as voids */
                int len_adj = (len <= 1) ? 5000 : -len * 200;
                int score = pts * 1000 + s + trump_pen + len_adj;
                if (score < best_score) { best_score = score; best = c; }
            END_FOR
            return best;
        } else {
            /* Following: play lowest non-winning card if possible.
             * If we must win, play cheapest winner. */
            int best_nonwin = -1, best_nw_score = 999999;
            int best_any = -1, best_any_score = 999999;

            FOR_EACH_CARD(legal, c)
                int pts = card_points(c, trump);
                int s = card_strength(c, trump);
                int wins = beats_table(c, tc_cards, tc_n, lead_suit, trump);
                int score = pts * 1000 + s;

                if (!wins && score < best_nw_score) {
                    best_nw_score = score; best_nonwin = c;
                }
                if (score < best_any_score) {
                    best_any_score = score; best_any = c;
                }
            END_FOR

            return (best_nonwin >= 0) ? best_nonwin : best_any;
        }
    }

    /* === GAINING: gap > 0 (need more points) === */

    if (is_leading) {
        /* Leading when we need points.
         * KEY PRINCIPLE: don't overcommit. Lead from suits where we're
         * likely to win, but prefer moderate-value plays. Avoid playing
         * aces/high trump unless gap is large relative to tricks left. */
        int need_per_trick = (tricks_left > 0) ? (gap + tricks_left - 1) / tricks_left : gap;

        int best = -1, best_score = -999999;
        FOR_EACH_CARD(legal, c)
            int cs = SUIT[c];
            int pts = card_points(c, trump);
            int s = card_strength(c, trump);

            /* Is this the top card in its suit (likely to hold)? */
            int is_top = 1;
            int k;
            for (k = 0; k < 4; k++) {
                if (k == player) continue;
                int h = highest_in_suit(all_hands[k], cs, trump);
                if (h >= 0 && card_strength(h, trump) > s) { is_top = 0; break; }
            }

            int score = 0;

            /* Trump conservation: save Puur/Nell for late game */
            if (cs == trump) {
                if (VALIDX[c] == 5 && tricks_left > 2) { score -= 15000; goto next_lead; } /* Never lead Puur early */
                if (VALIDX[c] == 3 && tricks_left > 2) { score -= 10000; goto next_lead; } /* Never lead Nell early */
                score -= 3000; /* Mild penalty for leading other trump */
            }

            /* Reward: points from card, but penalize overshooting */
            if (is_top && pts <= gap) {
                /* Will likely win, won't overshoot: good lead */
                score += pts * 500 + 3000;
            } else if (is_top && pts > gap) {
                /* Will likely win but overshoots: risky */
                score += 1000 - (pts - gap) * 300;
            } else {
                /* Won't hold: play mid-range, opponents will likely take it */
                /* Prefer medium cards from long suits */
                int len = suit_count(hand, cs);
                score += pts * 100 + len * 50;
            }

            next_lead:
            if (score > best_score) { best_score = score; best = c; }
        END_FOR
        return best;

    } else {
        /* Following/off-suit, need points.
         * Categorize legal cards into winners and losers. */
        int table_pts = 0;
        {int j; for (j = 0; j < tc_n; j++) table_pts += card_points(tc_cards[j], trump);}

        int best_winner = -1, cheapest_winner = -1;
        int best_win_val = -1, cheapest_win_str = 999;
        int best_loser = -1, best_lose_score = 999999;

        FOR_EACH_CARD(legal, c)
            int pts = card_points(c, trump);
            int s = card_strength(c, trump);
            int wins = beats_table(c, tc_cards, tc_n, lead_suit, trump);

            if (wins) {
                int trick_val = table_pts + pts;
                if (is_last_trick) trick_val += 5;
                if (trick_val > best_win_val) { best_win_val = trick_val; best_winner = c; }
                if (s < cheapest_win_str) { cheapest_win_str = s; cheapest_winner = c; }
            } else {
                /* When losing, play lowest-value card */
                int score = pts * 1000 + s;
                if (score < best_lose_score) { best_lose_score = score; best_loser = c; }
            }
        END_FOR

        if (best_winner >= 0 && best_win_val > 0) {
            /* Can win. Only take it if it helps hit target. */
            if (best_win_val > gap + 5 && tricks_left > 1) {
                /* Would overshoot significantly: prefer cheapest winner or lose */
                if (best_loser >= 0 && gap > 8) {
                    /* Big gap but would way overshoot — still take it */
                    return cheapest_winner;
                }
                if (best_loser >= 0) return best_loser;
                return cheapest_winner;
            }
            /* Take the trick: use cheapest winner when close to target */
            if (gap <= best_win_val + 3) return cheapest_winner;
            return best_winner;
        }

        /* Can't win: dump lowest-value card */
        if (best_loser >= 0) return best_loser;

        /* Fallback */
        {
            int best = -1, best_s = 999999;
            FOR_EACH_CARD(legal, c)
                int score = card_points(c, trump) * 1000 + card_strength(c, trump);
                if (score < best_s) { best_s = score; best = c; }
            END_FOR
            return best;
        }
    }
}

/* Main heuristic card picker for simulation rollouts.
 * gap = target - current_points.
 * is_last_trick: nonzero if this is the final trick (+5 bonus at stake). */
EXPORT int pick_card(hand_t hand, int lead_suit, int trump, int gap, int is_last_trick) {
    hand_t legal = legal_moves_fast(hand, lead_suit);
    if (!legal) return -1;

    /* Last-trick +5 bonus awareness */
    if (is_last_trick) {
        if (gap >= 2 && gap <= 8) {
            /* Need ~5 pts from bonus — try to win this trick.
             * Pick highest strength legal card to maximize win chance. */
            int best = -1, best_str = -1;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s > best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
        if (gap >= -8 && gap <= -2) {
            /* Over target by 2-8 — avoid winning (bonus would make it worse).
             * Pick lowest strength legal card. */
            int best = -1, best_str = 999;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s < best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
    }

    int best = -1;
    int best_score = (gap <= 0) ? 999999 : -999999;
    FOR_EACH_CARD(legal, c)
        int pts = card_points(c, trump);
        int str = card_strength(c, trump);
        int score = pts * 1000 + str;
        if (gap <= 0) {
            if (score < best_score || best < 0) { best_score = score; best = c; }
        } else {
            if (score > best_score || best < 0) { best_score = score; best = c; }
        }
    END_FOR
    return best;
}

/* ===== Human-like opponent model for solver ===== */

/*
 * Dampened card picker for solver opponent model.
 * Instead of picking the absolute best/worst card, picks at a
 * dampened percentile — modeling real opponents who play at ~60-70%
 * efficiency rather than the theoretical optimum.
 *
 * Calibrated against 862 real opponent plays from Swisslos:
 *   gap > 30:  real avg 6.3 pts vs pick_card's ~11 pts (~57%)
 *   gap 10-30: real avg 4.4 pts vs pick_card's ~8 pts  (~55%)
 *   gap < 0:   real avg 3.2 pts vs pick_card's ~0 pts  (much higher)
 *
 * The effect: with n legal cards sorted by score, instead of picking
 * index 0 (min) or n-1 (max), pick offset positions toward center.
 * offset = n * DAMPING / 100, capped at half the range.
 *
 * ONLY used inside sa_solve / optimal_play for the solver.
 * Regular PIMC simulation still uses the original pick_card.
 */
#define OPPONENT_DAMPING 25  /* percent toward center (0=original, 50=median) */

static int pick_card_human(hand_t hand, int lead_suit, int trump, int gap, int is_last_trick) {
    hand_t legal = legal_moves_fast(hand, lead_suit);
    if (!legal) return -1;

    /* Last-trick +5 bonus: keep exact behavior (too important to dampen) */
    if (is_last_trick) {
        if (gap >= 2 && gap <= 8) {
            int best = -1, best_str = -1;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s > best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
        if (gap >= -8 && gap <= -2) {
            int best = -1, best_str = 999;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s < best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
    }

    /*
     * Asymmetric damping: only dampen GRABBING, not shedding.
     *
     * Shedding (gap <= 0): humans are good at this (play low = easy).
     *   → Use exact pick_card behavior (THE weakest card).
     *   → Keeps dodge calculations accurate (was dev=0.6).
     *
     * Grabbing (gap > 0): humans are bad at this (winning tricks = hard).
     *   → Play 2nd strongest instead of THE strongest (with 4+ cards).
     *   → Models ~60-70% grab efficiency seen in real data.
     */
    if (gap <= 0) {
        /* Shedding: exact same as original pick_card — play THE weakest */
        int best = -1, best_score = 999999;
        FOR_EACH_CARD(legal, c)
            int sc = card_points(c, trump) * 1000 + card_strength(c, trump);
            if (sc < best_score || best < 0) { best_score = sc; best = c; }
        END_FOR
        return best;
    }

    {
        /* Grabbing: track best + second-best, return 2nd with 4+ cards */
        int best = -1, best_sc = -999999;
        int second = -1, second_sc = -999999;
        int nc = 0;

        FOR_EACH_CARD(legal, c)
            int sc = card_points(c, trump) * 1000 + card_strength(c, trump);
            nc++;
            if (sc > best_sc || best < 0) {
                second = best; second_sc = best_sc;
                best = c; best_sc = sc;
            } else if (sc > second_sc || second < 0) {
                second = c; second_sc = sc;
            }
        END_FOR

        if (nc >= 4 && second >= 0) return second;
        return best;
    }
}

/* ===== Table-aware opponent model for solver ===== */

/*
 * Smarter card picker for the single-agent solver.
 * Unlike pick_card_human (which ignores table state), this function
 * checks beats_table when shedding (prefers non-winning cards),
 * avoids leading trump when shedding, and considers table points
 * when deciding whether to grab a trick.
 *
 * Parameters same as pick_card_human PLUS:
 *   tc_cards, tc_n: cards already on the table this trick
 *
 * This replaces pick_card_human in sa_solve / optimal_play.
 */
static int pick_card_solver(hand_t hand, int lead_suit, int trump, int gap,
                             int *tc_cards, int tc_n, int is_last_trick) {
    hand_t legal = legal_moves_fast(hand, lead_suit);
    if (!legal) return -1;
    if (popcount64(legal) == 1) return ctz64(legal);

    int is_leading = (lead_suit < 0);

    /* === LAST TRICK: same logic as pick_card_human === */
    if (is_last_trick) {
        if (gap >= 2 && gap <= 8) {
            int best = -1, best_str = -1;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s > best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
        if (gap >= -8 && gap <= -2) {
            int best = -1, best_str = 999;
            FOR_EACH_CARD(legal, c)
                int s = card_strength(c, trump);
                if (s < best_str) { best_str = s; best = c; }
            END_FOR
            return best;
        }
    }

    /* === SHEDDING: gap <= 0 (at or over target) === */
    if (gap <= 0) {
        if (!is_leading && tc_n > 0) {
            /* Following: prefer cards that DON'T beat the table.
             * This is the key improvement over pick_card_human which
             * just plays weakest without checking if it wins. */
            int best_nonwin = -1, best_nw_sc = 999999;
            int best_any = -1, best_any_sc = 999999;

            FOR_EACH_CARD(legal, c)
                int pts = card_points(c, trump);
                int s = card_strength(c, trump);
                int sc = pts * 1000 + s;
                int wins = beats_table(c, tc_cards, tc_n, lead_suit, trump);

                if (!wins && (sc < best_nw_sc || best_nonwin < 0)) {
                    best_nw_sc = sc; best_nonwin = c;
                }
                if (sc < best_any_sc || best_any < 0) {
                    best_any_sc = sc; best_any = c;
                }
            END_FOR

            return (best_nonwin >= 0) ? best_nonwin : best_any;
        } else {
            /* Leading: play weakest, but strongly avoid trump.
             * Leading trump when shedding is very bad — trump wins tricks. */
            int best = -1, best_sc = 999999;
            FOR_EACH_CARD(legal, c)
                int cs = SUIT[c];
                int pts = card_points(c, trump);
                int s = card_strength(c, trump);
                int trump_pen = (cs == trump) ? 50000 : 0;
                int sc = pts * 1000 + s + trump_pen;
                if (sc < best_sc || best < 0) { best_sc = sc; best = c; }
            END_FOR
            return best;
        }
    }

    /* === GRABBING: gap > 0 (need more points) === */
    if (!is_leading && tc_n > 0) {
        /* Following: table-aware grabbing.
         * Consider table points and avoid overshooting. */
        int table_pts = 0;
        {int j; for (j = 0; j < tc_n; j++) table_pts += card_points(tc_cards[j], trump);}

        int best_winner = -1, best_win_val = -1;
        int cheapest_winner = -1, cheapest_win_str = 999;
        int best_loser = -1, best_lose_sc = 999999;

        FOR_EACH_CARD(legal, c)
            int pts = card_points(c, trump);
            int s = card_strength(c, trump);
            int wins = beats_table(c, tc_cards, tc_n, lead_suit, trump);

            if (wins) {
                int trick_val = table_pts + pts;
                if (is_last_trick) trick_val += 5;
                if (trick_val > best_win_val) { best_win_val = trick_val; best_winner = c; }
                if (s < cheapest_win_str) { cheapest_win_str = s; cheapest_winner = c; }
            } else {
                int sc = pts * 1000 + s;
                if (sc < best_lose_sc) { best_lose_sc = sc; best_loser = c; }
            }
        END_FOR

        if (best_winner >= 0) {
            /* Can win. Take it if it fits the gap; avoid massive overshoot. */
            if (best_win_val > gap + 8 && best_loser >= 0) {
                /* Would overshoot too much — dump instead */
                return best_loser;
            }
            /* Use cheapest winner when close to target */
            if (gap <= best_win_val + 3) return cheapest_winner;
            return best_winner;
        }

        /* Can't win: dump lowest */
        if (best_loser >= 0) return best_loser;
    }

    /* Leading and need points: dampened grab (same as pick_card_human) */
    {
        int best = -1, best_sc = -999999;
        int second = -1, second_sc = -999999;
        int nc = 0;

        FOR_EACH_CARD(legal, c)
            int sc = card_points(c, trump) * 1000 + card_strength(c, trump);
            nc++;
            if (sc > best_sc || best < 0) {
                second = best; second_sc = best_sc;
                best = c; best_sc = sc;
            } else if (sc > second_sc || second < 0) {
                second = c; second_sc = sc;
            }
        END_FOR

        if (nc >= 4 && second >= 0) return second;
        return best;
    }
}

/* ===== Batch game simulation ===== */

/*
 * Simulate one game from start (all 9 cards).
 * Returns player 0's final points.
 */
static int sim_one_game(
    hand_t my_hand, hand_t h1, hand_t h2, hand_t h3,
    int trump, int my_target, int t1, int t2, int t3
) {
    hand_t hands[4] = {my_hand, h1, h2, h3};
    int pts[4] = {0, 0, 0, 0};
    int targets[4] = {my_target, t1, t2, t3};
    int leader = 0;
    int trick;

    for (trick = 0; trick < 9; trick++) {
        int trick_players[4], trick_cards[4];
        int tc = 0;
        int lead_suit = -1;
        int i;

        for (i = 0; i < 4; i++) {
            int p = (leader + i) % 4;
            if (!hands[p]) continue;

            int gap = targets[p] - pts[p];
            int card = pick_card(hands[p], lead_suit, trump, gap, trick == 8);
            if (card < 0) continue;

            hands[p] &= ~(1ULL << card);
            trick_players[tc] = p;
            trick_cards[tc] = card;
            tc++;

            if (lead_suit < 0) lead_suit = SUIT[card];
        }

        if (tc == 0) break;

        int winner = trick_winner(trick_players, trick_cards, tc, lead_suit, trump);
        int tpts = trick_points_sum(trick_cards, tc, trump);
        if (trick == 8) tpts += 5; /* last trick bonus */
        pts[winner] += tpts;
        leader = winner;
    }

    return pts[0];
}

/*
 * Simulate N games in batch.
 * my_hand: bitmask of player 0's cards
 * opp_hands: flat array [w0_p1, w0_p2, w0_p3, w1_p1, ...] (N*3 bitmasks)
 * opp_targets: flat array [w0_t1, w0_t2, w0_t3, ...] (N*3 ints)
 * results: output array of N ints (player 0's final points per world)
 */
EXPORT void simulate_games_batch(
    int n_worlds,
    uint64_t my_hand,
    uint64_t *opp_hands,
    int trump,
    int my_target,
    int *opp_targets,
    int *results
) {
    int w;
    for (w = 0; w < n_worlds; w++) {
        int base = w * 3;
        results[w] = sim_one_game(
            my_hand,
            opp_hands[base], opp_hands[base+1], opp_hands[base+2],
            trump, my_target,
            opp_targets[base], opp_targets[base+1], opp_targets[base+2]
        );
    }
}

/* ===== Endgame solver (max^n) ===== */

#define MAX_TT_SIZE (1 << 20)  /* 1M entries */

typedef struct {
    uint64_t key;
    int pts[4];
    int valid;
} tt_entry_t;

static tt_entry_t endgame_tt[MAX_TT_SIZE];

static void tt_clear(void) {
    memset(endgame_tt, 0, sizeof(endgame_tt));
}

static inline uint64_t tt_hash(hand_t h0, hand_t h1, hand_t h2, hand_t h3,
                                int p0, int p1, int p2, int p3,
                                int leader, int trick_num, int tc_n) {
    uint64_t h = h0 ^ (h1 * 2654435761ULL) ^ (h2 * 40503ULL) ^ (h3 * 12345678901ULL);
    h ^= ((uint64_t)p0 << 40) ^ ((uint64_t)p1 << 44) ^ ((uint64_t)p2 << 48) ^ ((uint64_t)p3 << 52);
    h ^= ((uint64_t)leader << 36) ^ ((uint64_t)trick_num << 32) ^ ((uint64_t)tc_n << 28);
    return h;
}

static void eg_solve(
    hand_t hands[4], int pts[4], int targets[4],
    int leader, int trump, int trick_num,
    int tc_players[4], int tc_cards[4], int tc_n,
    int result_pts[4]
);

static void eg_solve(
    hand_t hands[4], int pts[4], int targets[4],
    int leader, int trump, int trick_num,
    int tc_players[4], int tc_cards[4], int tc_n,
    int result_pts[4]
) {
    /* Terminal check */
    if (!hands[0] && !hands[1] && !hands[2] && !hands[3] && tc_n == 0) {
        memcpy(result_pts, pts, 4 * sizeof(int));
        return;
    }

    /* TT lookup */
    uint64_t h = tt_hash(hands[0], hands[1], hands[2], hands[3],
                          pts[0], pts[1], pts[2], pts[3],
                          leader, trick_num, tc_n);
    int idx = (int)(h % MAX_TT_SIZE);
    if (endgame_tt[idx].valid && endgame_tt[idx].key == h) {
        memcpy(result_pts, endgame_tt[idx].pts, 4 * sizeof(int));
        return;
    }

    int active = (leader + tc_n) % 4;
    hand_t hand = hands[active];
    if (!hand) {
        memcpy(result_pts, pts, 4 * sizeof(int));
        return;
    }

    int lead_suit = tc_n > 0 ? SUIT[tc_cards[0]] : -1;
    hand_t legal = legal_moves_fast(hand, lead_suit);
    if (!legal) {
        memcpy(result_pts, pts, 4 * sizeof(int));
        return;
    }

    int best_set = 0;
    int best_pts[4];
    int best_u = -999;

    FOR_EACH_CARD(legal, card)
        hand_t new_hands[4];
        memcpy(new_hands, hands, 4 * sizeof(hand_t));
        new_hands[active] &= ~(1ULL << card);

        int new_tc_players[4], new_tc_cards[4];
        memcpy(new_tc_players, tc_players, 4 * sizeof(int));
        memcpy(new_tc_cards, tc_cards, 4 * sizeof(int));
        new_tc_players[tc_n] = active;
        new_tc_cards[tc_n] = card;
        int new_tc_n = tc_n + 1;

        int sub_pts[4];

        if (new_tc_n == 4) {
            /* Resolve trick */
            int ls = SUIT[new_tc_cards[0]];
            int winner = trick_winner(new_tc_players, new_tc_cards, 4, ls, trump);
            int tpts = trick_points_sum(new_tc_cards, 4, trump);
            if (trick_num == 8) tpts += 5;

            int new_pts[4];
            memcpy(new_pts, pts, 4 * sizeof(int));
            new_pts[winner] += tpts;

            int empty_tc_p[4] = {0}, empty_tc_c[4] = {0};
            eg_solve(new_hands, new_pts, targets, winner, trump,
                     trick_num + 1, empty_tc_p, empty_tc_c, 0, sub_pts);
        } else {
            eg_solve(new_hands, pts, targets, leader, trump,
                     trick_num, new_tc_players, new_tc_cards, new_tc_n, sub_pts);
        }

        int u = -((sub_pts[active] - targets[active]) < 0
                   ? targets[active] - sub_pts[active]
                   : sub_pts[active] - targets[active]);

        if (!best_set || u > best_u) {
            best_u = u;
            memcpy(best_pts, sub_pts, 4 * sizeof(int));
            best_set = 1;
            if (u == 0) break; /* perfect hit, prune */
        }
    END_FOR

    if (best_set) {
        memcpy(result_pts, best_pts, 4 * sizeof(int));
    } else {
        memcpy(result_pts, pts, 4 * sizeof(int));
    }

    /* TT store */
    endgame_tt[idx].key = h;
    memcpy(endgame_tt[idx].pts, result_pts, 4 * sizeof(int));
    endgame_tt[idx].valid = 1;
}

typedef struct {
    int best_card;
    int final_points;
} endgame_result_t;

EXPORT endgame_result_t solve_endgame(
    uint64_t *hands_arr,   /* 4 bitmasks */
    int leader, int trump,
    int *targets,          /* 4 ints */
    int *points_arr,       /* 4 ints */
    int trick_number       /* 0-indexed */
) {
    endgame_result_t res = {-1, points_arr[0]};
    tt_clear();

    hand_t hands[4] = {hands_arr[0], hands_arr[1], hands_arr[2], hands_arr[3]};
    int pts[4] = {points_arr[0], points_arr[1], points_arr[2], points_arr[3]};
    int tgts[4] = {targets[0], targets[1], targets[2], targets[3]};

    /* Find active player */
    int active = leader % 4;
    if (active != 0) {
        /* Not our turn to lead — just solve and return final pts */
        int result[4];
        int tc_p[4] = {0}, tc_c[4] = {0};
        eg_solve(hands, pts, tgts, leader, trump, trick_number, tc_p, tc_c, 0, result);
        res.final_points = result[0];
        return res;
    }

    /* Our turn: find best card */
    hand_t my_hand = hands[0];
    if (!my_hand) return res;

    hand_t legal = my_hand; /* leading, all cards legal */
    int best_u = -999;

    FOR_EACH_CARD(legal, card)
        hand_t new_hands[4];
        memcpy(new_hands, hands, 4 * sizeof(hand_t));
        new_hands[0] &= ~(1ULL << card);

        int tc_p[4] = {0}, tc_c[4] = {0};
        tc_p[0] = 0;
        tc_c[0] = card;

        int result[4];
        eg_solve(new_hands, pts, tgts, leader, trump, trick_number, tc_p, tc_c, 1, result);

        int u = -((result[0] - tgts[0]) < 0
                   ? tgts[0] - result[0]
                   : result[0] - tgts[0]);

        if (u > best_u) {
            best_u = u;
            res.best_card = card;
            res.final_points = result[0];
            if (u == 0) break;
        }
    END_FOR

    return res;
}

/* ===== Mid-game simulation for move selection ===== */

/*
 * Simulate remaining game from a mid-game position.
 * Returns player 0's final points.
 */
static int sim_remaining(
    hand_t hands[4], int pts[4], int targets[4],
    int trump, int leader, int trick_start, int total_tricks
) {
    int trick;
    for (trick = trick_start; trick < total_tricks; trick++) {
        int trick_players[4], trick_cards[4];
        int tc = 0;
        int lead_suit = -1;
        int i;

        for (i = 0; i < 4; i++) {
            int p = (leader + i) % 4;
            if (!hands[p]) continue;

            int gap = targets[p] - pts[p];
            int card = pick_card(hands[p], lead_suit, trump, gap, trick == total_tricks - 1);
            if (card < 0) continue;

            hands[p] &= ~(1ULL << card);
            trick_players[tc] = p;
            trick_cards[tc] = card;
            tc++;

            if (lead_suit < 0) lead_suit = SUIT[card];
        }

        if (tc == 0) break;

        int winner = trick_winner(trick_players, trick_cards, tc, lead_suit, trump);
        int tpts = trick_points_sum(trick_cards, tc, trump);
        if (trick == total_tricks - 1) tpts += 5; /* last trick bonus */
        pts[winner] += tpts;
        leader = winner;
    }

    return pts[0];
}

/*
 * Evaluate candidate moves for player 0 across multiple worlds.
 *
 * For each candidate card, for each world:
 *   1. Build current trick from table cards + candidate
 *   2. Opponents who haven't played complete the trick (heuristic)
 *   3. Resolve trick, then simulate remaining tricks
 *   4. Compute utility = -|target - final_points|
 *
 * results: output array of n_candidates floats (sum of utilities; divide by n_worlds for avg)
 *
 * table_players/table_cards: cards already played in current trick BEFORE player 0
 *   n_table: count of cards on table (0-3)
 */
EXPORT void evaluate_moves_batch(
    int n_worlds,
    int n_candidates,
    int *candidates,
    uint64_t my_hand,
    uint64_t *opp_hands,
    int trump,
    int my_target,
    int *opp_targets,
    int my_points,
    int *opp_points,
    int leader,
    int trick_number,
    int total_tricks,
    int *table_players,
    int *table_cards,
    int n_table,
    int *results
) {
    int ci, w;

    for (ci = 0; ci < n_candidates; ci++) {
        int cand = candidates[ci];
        int total_util = 0;

        for (w = 0; w < n_worlds; w++) {
            int base = w * 3;
            hand_t hands[4];
            hands[0] = my_hand & ~(1ULL << cand);
            hands[1] = opp_hands[base];
            hands[2] = opp_hands[base + 1];
            hands[3] = opp_hands[base + 2];

            int pts[4];
            pts[0] = my_points;
            pts[1] = opp_points ? opp_points[base] : 0;
            pts[2] = opp_points ? opp_points[base + 1] : 0;
            pts[3] = opp_points ? opp_points[base + 2] : 0;

            int targets[4];
            targets[0] = my_target;
            targets[1] = opp_targets[base];
            targets[2] = opp_targets[base + 1];
            targets[3] = opp_targets[base + 2];

            /* Build current trick */
            int tc_players[4], tc_cards[4];
            int tc_n = 0;
            int lead_suit = -1;
            int j;

            /* Cards already on table before us */
            for (j = 0; j < n_table; j++) {
                tc_players[tc_n] = table_players[j];
                tc_cards[tc_n] = table_cards[j];
                if (tc_n == 0) lead_suit = SUIT[table_cards[j]];
                /* Remove from opponent hands */
                hands[table_players[j]] &= ~(1ULL << table_cards[j]);
                tc_n++;
            }

            /* Play our candidate card */
            tc_players[tc_n] = 0;
            tc_cards[tc_n] = cand;
            if (tc_n == 0) lead_suit = SUIT[cand];
            tc_n++;

            /* Opponents who come after us complete the trick */
            {
                /* Figure out which players still need to play */
                int played_mask = 0;
                for (j = 0; j < tc_n; j++) played_mask |= (1 << tc_players[j]);

                for (j = 0; j < 4; j++) {
                    int p = (leader + j) % 4;
                    if (played_mask & (1 << p)) continue;
                    if (!hands[p]) continue;

                    int gap = targets[p] - pts[p];
                    /* Opponents use simple policy */
                    int card = pick_card(hands[p], lead_suit, trump, gap, trick_number == total_tricks - 1);
                    if (card < 0) continue;

                    hands[p] &= ~(1ULL << card);
                    tc_players[tc_n] = p;
                    tc_cards[tc_n] = card;
                    tc_n++;
                }
            }

            /* Resolve trick */
            if (tc_n > 0) {
                int winner = trick_winner(tc_players, tc_cards, tc_n, lead_suit, trump);
                int tpts = trick_points_sum(tc_cards, tc_n, trump);
                if (trick_number == total_tricks - 1) tpts += 5;
                pts[winner] += tpts;

                /* Simulate remaining tricks */
                sim_remaining(hands, pts, targets, trump, winner,
                              trick_number + 1, total_tricks);
            }

            int diff = my_target - pts[0];
            if (diff < 0) diff = -diff;
            total_util -= diff;
        }

        results[ci] = total_util;
    }
}

/* ===== Single rollout from mid-game state (for ISMCTS) ===== */

/*
 * Rollout a single game from a mid-game position with partial trick.
 * Completes the partial trick using heuristic, then plays out remaining tricks.
 * Returns player 0's final points.
 *
 * hands: 4 bitmasks
 * pts: 4 ints (current points)
 * targets: 4 ints
 * trump: trump suit
 * leader: who leads current trick
 * trick_number: 0-indexed current trick
 * total_tricks: total tricks in game (usually 9)
 * tc_players/tc_cards: partial trick cards already played
 * n_tc: number of cards in partial trick (0-4)
 */
EXPORT int rollout_game(
    uint64_t *hands_arr,
    int *pts_arr,
    int *targets_arr,
    int trump,
    int leader,
    int trick_number,
    int total_tricks,
    int *tc_players,
    int *tc_cards,
    int n_tc
) {
    hand_t hands[4] = {hands_arr[0], hands_arr[1], hands_arr[2], hands_arr[3]};
    int pts[4] = {pts_arr[0], pts_arr[1], pts_arr[2], pts_arr[3]};
    int targets[4] = {targets_arr[0], targets_arr[1], targets_arr[2], targets_arr[3]};

    /* Complete the partial trick if any */
    if (n_tc > 0 && n_tc < 4) {
        int trick_p[4], trick_c[4];
        int tc_n = 0;
        int lead_suit = SUIT[tc_cards[0]];
        int played_mask = 0;
        int j;

        /* Copy existing partial trick */
        for (j = 0; j < n_tc; j++) {
            trick_p[tc_n] = tc_players[j];
            trick_c[tc_n] = tc_cards[j];
            hands[tc_players[j]] &= ~(1ULL << tc_cards[j]);
            played_mask |= (1 << tc_players[j]);
            tc_n++;
        }

        /* Remaining players complete the trick */
        for (j = 0; j < 4; j++) {
            int p = (leader + j) % 4;
            if (played_mask & (1 << p)) continue;
            if (!hands[p]) continue;

            int gap = targets[p] - pts[p];
            int card = pick_card(hands[p], lead_suit, trump, gap, trick_number == total_tricks - 1);
            if (card < 0) continue;

            hands[p] &= ~(1ULL << card);
            trick_p[tc_n] = p;
            trick_c[tc_n] = card;
            tc_n++;
        }

        /* Resolve trick */
        if (tc_n > 0) {
            int winner = trick_winner(trick_p, trick_c, tc_n, lead_suit, trump);
            int tpts = trick_points_sum(trick_c, tc_n, trump);
            if (trick_number == total_tricks - 1) tpts += 5;
            pts[winner] += tpts;
            leader = winner;
        }
        trick_number++;
    } else if (n_tc == 4) {
        /* Full trick on table — resolve it */
        int lead_suit = SUIT[tc_cards[0]];
        int j;
        for (j = 0; j < 4; j++) hands[tc_players[j]] &= ~(1ULL << tc_cards[j]);
        int winner = trick_winner(tc_players, tc_cards, 4, lead_suit, trump);
        int tpts = trick_points_sum(tc_cards, 4, trump);
        if (trick_number == total_tricks - 1) tpts += 5;
        pts[winner] += tpts;
        leader = winner;
        trick_number++;
    }

    /* Simulate remaining tricks */
    if (trick_number < total_tricks) {
        sim_remaining(hands, pts, targets, trump, leader, trick_number, total_tricks);
    }

    return pts[0];
}

/* ===== Declaration evaluation ===== */

/*
 * Evaluate a target by simulating across all worlds.
 * Returns sum of -|target - actual| (divide by n_worlds in Python for mean).
 */
EXPORT int eval_declaration(
    uint64_t my_hand,
    uint64_t *opp_hands,    /* n_worlds * 3 */
    int *opp_targets,       /* n_worlds * 3 */
    int n_worlds,
    int trump,
    int target
) {
    int total = 0;
    int w;
    for (w = 0; w < n_worlds; w++) {
        int base = w * 3;
        int score = sim_one_game(
            my_hand,
            opp_hands[base], opp_hands[base+1], opp_hands[base+2],
            trump, target,
            opp_targets[base], opp_targets[base+1], opp_targets[base+2]
        );
        int diff = target - score;
        if (diff < 0) diff = -diff;
        total -= diff;
    }
    return total;
}

/*
 * Find optimal declaration via coarse+fine sweep.
 * Returns best target.
 */
EXPORT int find_best_declaration(
    uint64_t my_hand,
    uint64_t *opp_hands,
    int *opp_targets,
    int n_worlds,
    int trump
) {
    int best_target = 0;
    int best_eu = -999999;
    int z, eu;

    /* Coarse sweep: 0-157 step 5 */
    for (z = 0; z <= 157; z += 5) {
        eu = eval_declaration(my_hand, opp_hands, opp_targets, n_worlds, trump, z);
        if (eu > best_eu) { best_eu = eu; best_target = z; }
    }

    /* Fine sweep: ±6 */
    int lo = best_target - 6;
    int hi = best_target + 7;
    if (lo < 0) lo = 0;
    if (hi > 158) hi = 158;
    for (z = lo; z < hi; z++) {
        eu = eval_declaration(my_hand, opp_hands, opp_targets, n_worlds, trump, z);
        if (eu > best_eu) { best_eu = eu; best_target = z; }
    }

    return best_target;
}

/*
 * Simulate all worlds once with a rough target estimate.
 * Returns scores in results array.
 */
EXPORT void simulate_for_distribution(
    uint64_t my_hand,
    uint64_t *opp_hands,
    int *opp_targets,
    int n_worlds,
    int trump,
    int est_target,
    int *results
) {
    int w;
    for (w = 0; w < n_worlds; w++) {
        int base = w * 3;
        results[w] = sim_one_game(
            my_hand,
            opp_hands[base], opp_hands[base+1], opp_hands[base+2],
            trump, est_target,
            opp_targets[base], opp_targets[base+1], opp_targets[base+2]
        );
    }
}

/* ======================================================================
 * SINGLE-AGENT OPTIMAL SOLVER
 *
 * Key insight: opponents play a fixed policy (pick_card heuristic).
 * This makes the game a single-agent optimization problem.
 * We branch ONLY on player 0's legal moves (~3 per trick).
 * Opponents respond deterministically → no opponent branching.
 * Total tree: ~3^9 = 20K nodes. Trivially solvable per world.
 *
 * Returns optimal play for player 0 that minimizes |target - score|.
 * ====================================================================== */

typedef struct {
    int best_card;   /* optimal first move (-1 at leaf) */
    int best_score;  /* player 0's final score on the optimal path */
} sa_result_t;

/*
 * Core recursive solver. Branches only on player 0's choices.
 * Opponents play deterministically via pick_card().
 *
 * hands[4]: current bitmask hands
 * pts[4]: current accumulated points
 * targets[4]: declared targets (used by pick_card for opponent gap)
 * trump: trump suit
 * leader: who leads this trick
 * trick_num: 0-indexed current trick
 * total_tricks: total tricks (usually 9)
 * my_target: the target we're optimizing against (minimize |my_target - pts[0]|)
 */
static sa_result_t sa_solve(
    hand_t hands[4], int pts[4], int targets[4],
    int trump, int leader, int trick_num, int total_tricks,
    int my_target
) {
    sa_result_t result;

    /* Base case: no more tricks */
    if (trick_num >= total_tricks || !hands[0]) {
        result.best_card = -1;
        result.best_score = pts[0];
        return result;
    }

    /* Find player 0's position in this trick */
    int my_pos_in_trick = (4 + 0 - leader) % 4; /* 0=we lead, 1=second, etc. */

    /* Phase 1: opponents before us play deterministically */
    int tc_players[4], tc_cards[4], tc_n = 0;
    int lead_suit = -1;
    hand_t h_after[4];
    int pts_copy[4];
    int i;
    for (i = 0; i < 4; i++) { h_after[i] = hands[i]; pts_copy[i] = pts[i]; }

    for (i = 0; i < my_pos_in_trick; i++) {
        int p = (leader + i) % 4;
        if (!h_after[p]) continue;
        int gap = targets[p] - pts_copy[p];
        int card = pick_card_solver(h_after[p], lead_suit, trump, gap, tc_cards, tc_n, trick_num == total_tricks - 1);
        if (card < 0) continue;
        h_after[p] &= ~(1ULL << card);
        tc_players[tc_n] = p;
        tc_cards[tc_n] = card;
        if (lead_suit < 0) lead_suit = SUIT[card];
        tc_n++;
    }

    /* Phase 2: branch on all of player 0's legal moves */
    hand_t legal = legal_moves_fast(h_after[0], lead_suit);
    if (!legal) {
        /* No legal moves — shouldn't happen, but handle gracefully */
        result.best_card = -1;
        result.best_score = pts_copy[0];
        return result;
    }

    int best_card = -1;
    int best_score = -1;
    int best_utility = -999999;

    FOR_EACH_CARD(legal, card)
        /* Save pre-branch state for opponents after us */
        hand_t h_branch[4];
        int pts_branch[4];
        for (i = 0; i < 4; i++) { h_branch[i] = h_after[i]; pts_branch[i] = pts_copy[i]; }

        /* Play our card */
        h_branch[0] &= ~(1ULL << card);
        int b_tc_players[4], b_tc_cards[4], b_tc_n = tc_n;
        for (i = 0; i < tc_n; i++) {
            b_tc_players[i] = tc_players[i];
            b_tc_cards[i] = tc_cards[i];
        }
        b_tc_players[b_tc_n] = 0;
        b_tc_cards[b_tc_n] = card;
        int b_lead = lead_suit;
        if (b_lead < 0) b_lead = SUIT[card];
        b_tc_n++;

        /* Phase 3: opponents after us play with table-aware model */
        for (i = my_pos_in_trick + 1; i < 4; i++) {
            int p = (leader + i) % 4;
            if (p == 0) continue; /* skip ourselves */
            if (!h_branch[p]) continue;
            int gap = targets[p] - pts_branch[p];
            int c2 = pick_card_solver(h_branch[p], b_lead, trump, gap, b_tc_cards, b_tc_n, trick_num == total_tricks - 1);
            if (c2 < 0) continue;
            h_branch[p] &= ~(1ULL << c2);
            b_tc_players[b_tc_n] = p;
            b_tc_cards[b_tc_n] = c2;
            b_tc_n++;
        }

        /* Resolve trick */
        int winner = trick_winner(b_tc_players, b_tc_cards, b_tc_n, b_lead, trump);
        int tpts = trick_points_sum(b_tc_cards, b_tc_n, trump);
        if (trick_num == total_tricks - 1) tpts += 5;
        pts_branch[winner] += tpts;

        /* Recurse for remaining tricks */
        sa_result_t sub = sa_solve(h_branch, pts_branch, targets, trump,
                                    winner, trick_num + 1, total_tricks, my_target);

        /* Evaluate: minimize |my_target - final_score|
         * Asymmetric penalty: overshoot costs 33% more than undershoot.
         * Data from 48 live rounds shows 52% overshoot vs 31% undershoot,
         * so the solver systematically underpredicts points collected.
         * This bias correction makes the solver prefer conservative plays. */
        {
        int over = sub.best_score - my_target;  /* positive = overshoot */
        int abs_diff = over < 0 ? -over : over;
        int utility = -abs_diff;
        if (over > 0) utility -= abs_diff / 3;  /* extra ~33% penalty for overshoot */

        /* Trump retention bonus: prefer keeping trumps for endgame.
         * In early tricks (0-4), penalize playing trumps slightly.
         * This makes play more ROBUST against unpredictable opponents:
         * - Early tricks: shed cheap cards, opponents are unpredictable
         * - Late tricks: deploy trumps with near-perfect information
         * The penalty is small (1 point) so it only breaks ties —
         * it won't override a clearly better play. */
        if (trick_num < 5 && SUIT[card] == trump) {
            utility -= 1;
        }

        if (utility > best_utility) {
            best_utility = utility;
            best_card = card;
            best_score = sub.best_score;
            if (utility == 0) break; /* perfect hit — prune */
        }
        } /* end asymmetric utility block */
    END_FOR

    result.best_card = best_card;
    result.best_score = best_score;
    return result;
}

/*
 * Exported single-agent solver for one world.
 * Returns best_card (bits 0-7) and best_score (bits 8-23) packed into one int.
 */
EXPORT int solve_single_agent(
    uint64_t my_hand,
    uint64_t *opp_hands_3,  /* 3 bitmasks: [p1, p2, p3] */
    int trump,
    int target,
    int leader,
    int *points_4,          /* 4 ints: current points [p0, p1, p2, p3] */
    int trick_number,
    int total_tricks,
    int *opp_targets_3      /* 3 ints: opponent targets [t1, t2, t3] */
) {
    init_tables();
    hand_t hands[4] = {my_hand, opp_hands_3[0], opp_hands_3[1], opp_hands_3[2]};
    int pts[4] = {points_4[0], points_4[1], points_4[2], points_4[3]};
    int targets[4] = {target, opp_targets_3[0], opp_targets_3[1], opp_targets_3[2]};

    sa_result_t r = sa_solve(hands, pts, targets, trump, leader,
                             trick_number, total_tricks, target);

    /* Pack: low byte = card (0-35 or 255 for -1), next bytes = score */
    int card_val = (r.best_card < 0) ? 255 : r.best_card;
    return (r.best_score << 8) | card_val;
}

/*
 * Optimal declaration: find the target T that minimizes average |T - achievable_score|
 * across sampled worlds.
 *
 * Uses coarse-then-fine sweep: step 5 on 500 worlds, then ±6 fine on all worlds.
 */
EXPORT int optimal_declaration(
    uint64_t my_hand,
    uint64_t *opp_hands,    /* n_worlds * 3 bitmasks */
    int *opp_targets,       /* n_worlds * 3 ints */
    int n_worlds,
    int trump
) {
    init_tables();
    int total_tricks = popcount64(my_hand);
    if (total_tricks <= 0) return 0;

    int coarse_n = n_worlds < 500 ? n_worlds : 500;
    int best_t = 0, best_eu = -999999999;
    int t, w;

    /* Phase 1: coarse sweep (step 5) over subset of worlds */
    for (t = 0; t <= 157; t += 5) {
        int total_eu = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_eu) schedule(dynamic, 4)
        #endif
        for (w = 0; w < coarse_n; w++) {
            int base = w * 3;
            hand_t hands[4] = {my_hand, opp_hands[base], opp_hands[base+1], opp_hands[base+2]};
            int pts[4] = {0, 0, 0, 0};
            int targets[4] = {t, opp_targets[base], opp_targets[base+1], opp_targets[base+2]};

            sa_result_t r = sa_solve(hands, pts, targets, trump, 0, 0, total_tricks, t);
            int diff = t - r.best_score;
            if (diff < 0) diff = -diff;
            total_eu -= diff;
        }
        if (total_eu > best_eu) { best_eu = total_eu; best_t = t; }
    }

    /* Phase 2: fine sweep (±7 around best) over ALL worlds */
    int lo = best_t - 7;
    int hi = best_t + 7;
    if (lo < 0) lo = 0;
    if (hi > 157) hi = 157;

    for (t = lo; t <= hi; t++) {
        int total_eu = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_eu) schedule(dynamic, 4)
        #endif
        for (w = 0; w < n_worlds; w++) {
            int base = w * 3;
            hand_t hands[4] = {my_hand, opp_hands[base], opp_hands[base+1], opp_hands[base+2]};
            int pts[4] = {0, 0, 0, 0};
            int targets[4] = {t, opp_targets[base], opp_targets[base+1], opp_targets[base+2]};

            sa_result_t r = sa_solve(hands, pts, targets, trump, 0, 0, total_tricks, t);
            int diff = t - r.best_score;
            if (diff < 0) diff = -diff;
            total_eu -= diff;
        }
        if (total_eu > best_eu) { best_eu = total_eu; best_t = t; }
    }

    return best_t;
}

/*
 * Optimal card play: for each candidate, solve all worlds with single-agent solver.
 * Replaces evaluate_moves_batch's heuristic rollout with exact optimal play.
 *
 * Same interface as evaluate_moves_batch for easy swap-in.
 */
EXPORT void optimal_play(
    int n_worlds,
    int n_candidates,
    int *candidates,
    uint64_t my_hand,
    uint64_t *opp_hands,
    int trump,
    int my_target,
    int *opp_targets,
    int my_points,
    int *opp_points,
    int leader,
    int trick_number,
    int total_tricks,
    int *table_players,
    int *table_cards,
    int n_table,
    int *results
) {
    init_tables();
    int ci, w;

    for (ci = 0; ci < n_candidates; ci++) {
        int cand = candidates[ci];
        int total_util = 0;

        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:total_util) schedule(dynamic, 4)
        #endif
        for (w = 0; w < n_worlds; w++) {
            int base = w * 3;
            hand_t hands[4];
            hands[0] = my_hand & ~(1ULL << cand);
            hands[1] = opp_hands[base];
            hands[2] = opp_hands[base + 1];
            hands[3] = opp_hands[base + 2];

            int pts[4];
            pts[0] = my_points;
            pts[1] = opp_points ? opp_points[base] : 0;
            pts[2] = opp_points ? opp_points[base + 1] : 0;
            pts[3] = opp_points ? opp_points[base + 2] : 0;

            int targets[4];
            targets[0] = my_target;
            targets[1] = opp_targets[base];
            targets[2] = opp_targets[base + 1];
            targets[3] = opp_targets[base + 2];

            /* Build current trick from table cards + our candidate */
            int tc_players[4], tc_cards[4], tc_n = 0;
            int lead_suit = -1;
            int j;

            for (j = 0; j < n_table; j++) {
                tc_players[tc_n] = table_players[j];
                tc_cards[tc_n] = table_cards[j];
                hands[table_players[j]] &= ~(1ULL << table_cards[j]);
                if (tc_n == 0) lead_suit = SUIT[table_cards[j]];
                tc_n++;
            }

            /* Our card */
            tc_players[tc_n] = 0;
            tc_cards[tc_n] = cand;
            if (lead_suit < 0) lead_suit = SUIT[cand];
            tc_n++;

            /* Opponents after us complete the trick */
            {
                int played_mask = 0;
                for (j = 0; j < tc_n; j++) played_mask |= (1 << tc_players[j]);

                for (j = 0; j < 4; j++) {
                    int p = (leader + j) % 4;
                    if (played_mask & (1 << p)) continue;
                    if (!hands[p]) continue;
                    int gap = targets[p] - pts[p];
                    int card = pick_card_solver(hands[p], lead_suit, trump, gap,
                                         tc_cards, tc_n, trick_number == total_tricks - 1);
                    if (card < 0) continue;
                    hands[p] &= ~(1ULL << card);
                    tc_players[tc_n] = p;
                    tc_cards[tc_n] = card;
                    tc_n++;
                }
            }

            /* Resolve trick */
            int winner = trick_winner(tc_players, tc_cards, tc_n, lead_suit, trump);
            int tpts = trick_points_sum(tc_cards, tc_n, trump);
            if (trick_number == total_tricks - 1) tpts += 5;
            pts[winner] += tpts;

            /* Solve remaining tricks optimally for player 0 */
            sa_result_t sub = sa_solve(hands, pts, targets, trump, winner,
                                        trick_number + 1, total_tricks, my_target);

            /* Asymmetric: penalize overshoot 33% more */
            int over = sub.best_score - my_target;
            int abs_diff = over < 0 ? -over : over;
            int util = -abs_diff;
            if (over > 0) util -= abs_diff / 3;
            total_util += util;
        }

        results[ci] = total_util;
    }
}

/* ======================================================================
 * RL GAME MANAGEMENT
 * Vectorized environment for reinforcement learning self-play.
 * ====================================================================== */

/* Fast xoshiro256** PRNG — one per game for reproducibility */
typedef struct {
    uint64_t s[4];
} rng_t;

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_t *rng) {
    const uint64_t result = rotl(rng->s[1] * 5, 7) * 9;
    const uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);
    return result;
}

static inline void rng_seed(rng_t *rng, uint64_t seed) {
    /* SplitMix64 to initialize xoshiro state from single seed */
    uint64_t z = seed;
    int i;
    for (i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15ULL;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

/* ── RL GameState ── */

#define PHASE_DECLARE 0
#define PHASE_PLAY    1
#define PHASE_DONE    2

typedef struct {
    hand_t  hands[4];          /* current hands (bitmask) */
    hand_t  initial_hands[4];  /* dealt hands (for observation) */
    int     trump;             /* 0-3 */
    int     declarations[4];   /* declared scores (-1 if not yet declared) */
    int     points[4];         /* accumulated points */
    int     trick_cards[4];    /* cards in current trick (-1 = empty) */
    int     trick_players[4];  /* player who played each card in trick */
    int     trick_count;       /* which trick 0-8 */
    int     cards_in_trick;    /* 0-3 cards played in current trick */
    int     leader;            /* who leads current trick */
    int     current_player;    /* whose turn (0-3) */
    int     phase;             /* PHASE_DECLARE, PHASE_PLAY, PHASE_DONE */
    int     decl_count;        /* how many players have declared (0-4) */
    hand_t  played_cards;      /* all cards played so far (for observation) */
    int     voids[4][4];       /* voids[player][suit] = 1 if known void */
    rng_t   rng;               /* per-game RNG */
} RLGameState;

/* Fisher-Yates shuffle of array[0..n-1] */
static void shuffle(int *arr, int n, rng_t *rng) {
    int i;
    for (i = n - 1; i > 0; i--) {
        int j = (int)(rng_next(rng) % (uint64_t)(i + 1));
        int tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
}

EXPORT void rl_game_init(RLGameState *g, uint64_t seed) {
    init_tables();
    memset(g, 0, sizeof(RLGameState));

    rng_seed(&g->rng, seed);

    /* Create deck and shuffle */
    int deck[36];
    int i;
    for (i = 0; i < 36; i++) deck[i] = i;
    shuffle(deck, 36, &g->rng);

    /* Deal 9 cards to each player */
    for (i = 0; i < 36; i++) {
        int p = i / 9;
        g->hands[p] |= (1ULL << deck[i]);
    }
    for (i = 0; i < 4; i++) g->initial_hands[i] = g->hands[i];

    /* Random trump */
    g->trump = (int)(rng_next(&g->rng) % 4);

    /* Init trick cards to -1 */
    for (i = 0; i < 4; i++) {
        g->trick_cards[i] = -1;
        g->declarations[i] = -1;
    }

    g->phase = PHASE_DECLARE;
    g->current_player = 0;
    g->leader = 0;
    g->played_cards = 0;
}

/* ── Correct legal moves (undertrump + Puur privilege) ── */

EXPORT hand_t rl_legal_moves(const RLGameState *g) {
    int p = g->current_player;
    hand_t hand = g->hands[p];
    if (!hand) return 0;
    if (g->phase != PHASE_PLAY) return 0;

    int trump = g->trump;

    /* Leading: anything is legal */
    if (g->cards_in_trick == 0) return hand;

    int lead_suit = SUIT[g->trick_cards[0]];
    hand_t follow = hand & SUIT_MASK[lead_suit];
    hand_t my_trump = hand & SUIT_MASK[trump];

    if (lead_suit == trump) {
        /* Trump was led — must play trump if you have it */
        if (!my_trump) return hand;  /* no trump: anything */

        /* Find highest trump on table */
        int highest_str = -1;
        int j;
        for (j = 0; j < g->cards_in_trick; j++) {
            int c = g->trick_cards[j];
            if (SUIT[c] == trump) {
                int s = STR[c][1];
                if (s > highest_str) highest_str = s;
            }
        }

        /* Cards that can overtrump */
        hand_t over = 0;
        FOR_EACH_CARD(my_trump, c)
            if (STR[c][1] > highest_str) over |= (1ULL << c);
        END_FOR

        if (over) {
            /* Must overtrump. Puur privilege: if only Puur can overtrump, may play any trump */
            int puur_id = PUUR[trump];
            if (over == (1ULL << puur_id)) {
                return my_trump;  /* Puur privilege */
            }
            return over;
        }
        /* Can't overtrump: play any trump */
        return my_trump;
    }

    /* Non-trump led */
    if (follow) return follow;  /* Can follow suit */

    /* Can't follow suit: check undertrump restriction */
    int highest_trump_str = -1;
    {
        int j;
        for (j = 0; j < g->cards_in_trick; j++) {
            int c = g->trick_cards[j];
            if (SUIT[c] == trump) {
                int s = STR[c][1];
                if (s > highest_trump_str) highest_trump_str = s;
            }
        }
    }

    if (highest_trump_str < 0) return hand;  /* No trump on table: anything */

    /* Trump on table: can't undertrump unless forced */
    hand_t non_trump = hand & ~SUIT_MASK[trump];
    hand_t over_trump = 0;
    FOR_EACH_CARD(my_trump, c)
        if (STR[c][1] > highest_trump_str) over_trump |= (1ULL << c);
    END_FOR

    hand_t allowed = non_trump | over_trump;
    if (allowed) return allowed;

    /* Only have undertrumps: forced */
    return hand;
}

/* ── Set declaration ── */

EXPORT int rl_set_declaration(RLGameState *g, int player, int value) {
    if (g->phase != PHASE_DECLARE) return -1;
    if (player < 0 || player > 3) return -1;
    if (g->declarations[player] >= 0) return -1;  /* already declared */
    if (value < 0) value = 0;
    if (value > 157) value = 157;

    g->declarations[player] = value;
    g->decl_count++;

    if (g->decl_count >= 4) {
        g->phase = PHASE_PLAY;
        g->current_player = g->leader;
    } else {
        /* Next player to declare */
        g->current_player = g->decl_count;
    }
    return 0;
}

/* ── Play a card ── */

EXPORT int rl_step(RLGameState *g, int card) {
    if (g->phase != PHASE_PLAY) return -1;

    int p = g->current_player;
    hand_t legal = rl_legal_moves(g);

    /* Validate */
    if (!(legal & (1ULL << card))) return -1;

    /* Remove card from hand */
    g->hands[p] &= ~(1ULL << card);
    g->played_cards |= (1ULL << card);

    /* Add to trick */
    g->trick_cards[g->cards_in_trick] = card;
    g->trick_players[g->cards_in_trick] = p;
    g->cards_in_trick++;

    /* Detect void */
    if (g->cards_in_trick > 1) {
        int lead_suit = SUIT[g->trick_cards[0]];
        if (SUIT[card] != lead_suit) {
            g->voids[p][lead_suit] = 1;
        }
    }

    /* Trick complete? */
    if (g->cards_in_trick == 4) {
        int lead_suit = SUIT[g->trick_cards[0]];
        int winner = trick_winner(g->trick_players, g->trick_cards, 4, lead_suit, g->trump);
        int tpts = trick_points_sum(g->trick_cards, 4, g->trump);
        if (g->trick_count == 8) tpts += 5;  /* last trick bonus */
        g->points[winner] += tpts;

        g->trick_count++;
        g->leader = winner;
        g->cards_in_trick = 0;
        {
            int k;
            for (k = 0; k < 4; k++) { g->trick_cards[k] = -1; g->trick_players[k] = -1; }
        }

        if (g->trick_count >= 9) {
            g->phase = PHASE_DONE;
            g->current_player = -1;
            return 0;
        }
        g->current_player = winner;
    } else {
        /* Next player in trick order */
        g->current_player = (g->leader + g->cards_in_trick) % 4;
    }

    return 0;
}

/* ── Get reward for player (only meaningful when done) ── */

EXPORT float rl_get_reward(const RLGameState *g, int player) {
    if (g->phase != PHASE_DONE) return 0.0f;
    int diff = g->declarations[player] - g->points[player];
    if (diff < 0) diff = -diff;
    return (float)(-diff);
}

/* ── Observation encoding ── */

#define OBS_PLAY_SIZE  204
#define OBS_DECL_SIZE   72

EXPORT int rl_obs_size_play(void)  { return OBS_PLAY_SIZE; }
EXPORT int rl_obs_size_decl(void)  { return OBS_DECL_SIZE; }

EXPORT void rl_get_obs_play(const RLGameState *g, int player, float *obs) {
    int i, j;
    memset(obs, 0, OBS_PLAY_SIZE * sizeof(float));

    hand_t my_hand = g->hands[player];
    int trump = g->trump;

    /* [0..35] my_hand */
    FOR_EACH_CARD(my_hand, c)
        obs[c] = 1.0f;
    END_FOR

    /* [36..71] played_all — cards in completed tricks (not current trick) */
    hand_t trick_on_table = 0;
    for (i = 0; i < g->cards_in_trick; i++) {
        if (g->trick_cards[i] >= 0) trick_on_table |= (1ULL << g->trick_cards[i]);
    }
    hand_t prev_played = g->played_cards & ~trick_on_table;
    FOR_EACH_CARD(prev_played, c)
        obs[36 + c] = 1.0f;
    END_FOR

    /* [72..107] is_trump */
    FOR_EACH_CARD(SUIT_MASK[trump], c)
        obs[72 + c] = 1.0f;
    END_FOR

    /* [108..143] legal_mask */
    if (player == g->current_player) {
        hand_t legal = rl_legal_moves(g);
        FOR_EACH_CARD(legal, c)
            obs[108 + c] = 1.0f;
        END_FOR
    }

    /* [144..179] current trick cards on table */
    FOR_EACH_CARD(trick_on_table, c)
        obs[144 + c] = 1.0f;
    END_FOR

    /* [180..191] void knowledge (3 opponents × 4 suits) */
    for (i = 0; i < 3; i++) {
        int opp = (player + 1 + i) % 4;
        for (j = 0; j < 4; j++) {
            obs[180 + i * 4 + j] = g->voids[opp][j] ? 1.0f : 0.0f;
        }
    }

    /* Scalars */
    obs[192] = (float)g->trick_count / 8.0f;
    obs[193] = (float)g->points[player] / 157.0f;
    float gap = (float)(g->declarations[player] - g->points[player]) / 157.0f;
    if (gap > 1.0f) gap = 1.0f;
    if (gap < -1.0f) gap = -1.0f;
    obs[194] = gap;
    obs[195] = (float)g->cards_in_trick / 3.0f;
    obs[196] = (player == g->leader) ? 1.0f : 0.0f;

    /* Opponent points and declarations (relative order) */
    for (i = 0; i < 3; i++) {
        int opp = (player + 1 + i) % 4;
        obs[197 + i] = (float)g->points[opp] / 157.0f;
        obs[200 + i] = (g->declarations[opp] >= 0) ? (float)g->declarations[opp] / 157.0f : 0.0f;
    }
    obs[203] = (g->declarations[player] >= 0) ? (float)g->declarations[player] / 157.0f : 0.0f;
}

EXPORT void rl_get_obs_decl(const RLGameState *g, int player, float *obs) {
    int i;
    memset(obs, 0, OBS_DECL_SIZE * sizeof(float));

    hand_t my_hand = g->hands[player];
    int trump = g->trump;

    /* [0..35] hand binary */
    FOR_EACH_CARD(my_hand, c)
        obs[c] = 1.0f;
    END_FOR

    /* [36..39] trump one-hot */
    obs[36 + trump] = 1.0f;

    /* [40..48] trump card positions (which of the 9 trump cards do I have) */
    hand_t my_trump = my_hand & SUIT_MASK[trump];
    FOR_EACH_CARD(my_trump, c)
        obs[40 + VALIDX[c]] = 1.0f;
    END_FOR

    /* [49..52] suit lengths / 9 */
    int suit_lens[4] = {0, 0, 0, 0};
    for (i = 0; i < 4; i++) suit_lens[i] = popcount64(my_hand & SUIT_MASK[i]);
    for (i = 0; i < 4; i++) obs[49 + i] = (float)suit_lens[i] / 9.0f;

    /* [53..56] suit points / 62 */
    int suit_pts[4] = {0, 0, 0, 0};
    FOR_EACH_CARD(my_hand, c)
        suit_pts[SUIT[c]] += card_points(c, trump);
    END_FOR
    for (i = 0; i < 4; i++) obs[53 + i] = (float)suit_pts[i] / 62.0f;

    /* [57..59] has_puur, has_nell, has trump ace */
    int trump_base = trump * 9;
    obs[57] = (my_hand & (1ULL << (trump_base + 5))) ? 1.0f : 0.0f;
    obs[58] = (my_hand & (1ULL << (trump_base + 3))) ? 1.0f : 0.0f;
    obs[59] = (my_hand & (1ULL << (trump_base + 8))) ? 1.0f : 0.0f;

    /* [60] trump_count / 9 */
    obs[60] = (float)suit_lens[trump] / 9.0f;

    /* [61] void_count / 3 */
    int void_count = 0;
    for (i = 0; i < 4; i++) {
        if (i != trump && suit_lens[i] == 0) void_count++;
    }
    obs[61] = (float)void_count / 3.0f;

    /* [62] total_points / 157 */
    int total_pts = 0;
    FOR_EACH_CARD(my_hand, c)
        total_pts += card_points(c, trump);
    END_FOR
    obs[62] = (float)total_pts / 157.0f;

    /* [63] trump_points / 62 */
    obs[63] = (float)suit_pts[trump] / 62.0f;

    /* [64] side_aces / 3 */
    int side_aces = 0;
    for (i = 0; i < 4; i++) {
        if (i != trump && (my_hand & (1ULL << (i * 9 + 8)))) side_aces++;
    }
    obs[64] = (float)side_aces / 3.0f;

    /* [65] high_trump_count / 3 */
    int high_trump = 0;
    if (my_hand & (1ULL << (trump_base + 5))) high_trump++;
    if (my_hand & (1ULL << (trump_base + 3))) high_trump++;
    if (my_hand & (1ULL << (trump_base + 8))) high_trump++;
    obs[65] = (float)high_trump / 3.0f;

    /* [66] protected_aces / 3 */
    int protected_aces = 0;
    for (i = 0; i < 4; i++) {
        if (i != trump && (my_hand & (1ULL << (i * 9 + 8))) && suit_lens[i] >= 2)
            protected_aces++;
    }
    obs[66] = (float)protected_aces / 3.0f;

    /* [67] long_suits / 3 */
    int long_suits = 0;
    for (i = 0; i < 4; i++) {
        if (i != trump && suit_lens[i] >= 4) long_suits++;
    }
    obs[67] = (float)long_suits / 3.0f;
}

/* ── Batch operations for vectorized env ── */

EXPORT void rl_batch_init(RLGameState *games, int n, uint64_t base_seed) {
    int i;
    for (i = 0; i < n; i++) {
        rl_game_init(&games[i], base_seed + (uint64_t)i * 6364136223846793005ULL);
    }
}

EXPORT void rl_batch_legal_moves(const RLGameState *games, int n, uint64_t *out) {
    int i;
    for (i = 0; i < n; i++) {
        out[i] = rl_legal_moves(&games[i]);
    }
}

EXPORT void rl_batch_step(RLGameState *games, int n, const int *actions, int *status) {
    int i;
    for (i = 0; i < n; i++) {
        status[i] = rl_step(&games[i], actions[i]);
    }
}

EXPORT void rl_batch_set_declarations(RLGameState *games, int n, const int *decls) {
    int i, p;
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            rl_set_declaration(&games[i], p, decls[i * 4 + p]);
        }
    }
}

EXPORT void rl_batch_get_rewards(const RLGameState *games, int n, float *out) {
    int i, p;
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            out[i * 4 + p] = rl_get_reward(&games[i], p);
        }
    }
}

EXPORT void rl_batch_get_obs_play(const RLGameState *games, int n, float *out) {
    int i;
    for (i = 0; i < n; i++) {
        rl_get_obs_play(&games[i], games[i].current_player, &out[i * OBS_PLAY_SIZE]);
    }
}

EXPORT void rl_batch_get_obs_decl(const RLGameState *games, int n, int player, float *out) {
    int i;
    for (i = 0; i < n; i++) {
        rl_get_obs_decl(&games[i], player, &out[i * OBS_DECL_SIZE]);
    }
}

EXPORT void rl_batch_get_current_player(const RLGameState *games, int n, int *out) {
    int i;
    for (i = 0; i < n; i++) {
        out[i] = games[i].current_player;
    }
}

EXPORT void rl_batch_get_phase(const RLGameState *games, int n, int *out) {
    int i;
    for (i = 0; i < n; i++) {
        out[i] = games[i].phase;
    }
}

/* Get points for all players in all games. out is [n * 4] */
EXPORT void rl_batch_get_points(const RLGameState *games, int n, int *out) {
    int i, p;
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            out[i * 4 + p] = games[i].points[p];
        }
    }
}

/* Get declarations for all players in all games. out is [n * 4] */
EXPORT void rl_batch_get_declarations(const RLGameState *games, int n, int *out) {
    int i, p;
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            out[i * 4 + p] = games[i].declarations[p];
        }
    }
}

/* Rank-based rewards: 1st=+3, 2nd=+1, 3rd=-1, 4th=-3. Ties share average.
 * deviations: [n*4] input, rewards: [n*4] output floats */
EXPORT void rl_batch_rank_rewards(const int *deviations, int n, float *rewards) {
    static const float RANK_REWARD[4] = {3.0f, 1.0f, -1.0f, -3.0f};
    int i, j, k;
    for (i = 0; i < n; i++) {
        int dev[4];
        int idx[4] = {0, 1, 2, 3};
        for (j = 0; j < 4; j++) dev[j] = deviations[i * 4 + j];
        /* Sort indices by deviation ascending (lowest = best) */
        for (j = 0; j < 3; j++) {
            for (k = j + 1; k < 4; k++) {
                if (dev[idx[k]] < dev[idx[j]]) {
                    int tmp = idx[j]; idx[j] = idx[k]; idx[k] = tmp;
                }
            }
        }
        /* Assign rewards, handling ties */
        j = 0;
        while (j < 4) {
            int tie_start = j;
            float sum_r = RANK_REWARD[j];
            while (j + 1 < 4 && dev[idx[j + 1]] == dev[idx[tie_start]]) {
                j++;
                sum_r += RANK_REWARD[j];
            }
            float avg_r = sum_r / (float)(j - tie_start + 1);
            for (k = tie_start; k <= j; k++) {
                rewards[i * 4 + idx[k]] = avg_r;
            }
            j++;
        }
    }
}

/* ── Heuristic/random play for opponent seats in league training ── */

/* Pick a random legal card using the game's RNG */
EXPORT int rl_pick_random(RLGameState *g) {
    hand_t legal = rl_legal_moves(g);
    if (!legal) return -1;
    int n = popcount64(legal);
    int choice = (int)(rng_next(&g->rng) % (uint64_t)n);
    int i = 0;
    FOR_EACH_CARD(legal, c)
        if (i == choice) return c;
        i++;
    END_FOR
    return ctz64(legal);
}

/* Pick card using gap-based heuristic (same as sim rollouts) */
EXPORT int rl_pick_heuristic(const RLGameState *g) {
    int p = g->current_player;
    hand_t hand = g->hands[p];
    if (!hand) return -1;

    int lead_suit = (g->cards_in_trick > 0) ? SUIT[g->trick_cards[0]] : -1;
    int gap = g->declarations[p] - g->points[p];
    int is_last = (g->trick_count == 8);

    /* Use legal_moves_fast for the heuristic (matches sim behavior) */
    return pick_card(hand, lead_suit, g->trump, gap, is_last);
}

/* Pick card with overridden gap (for aggressive/passive opponents) */
EXPORT int rl_pick_heuristic_gap(const RLGameState *g, int gap_override) {
    int p = g->current_player;
    hand_t hand = g->hands[p];
    if (!hand) return -1;
    int lead_suit = (g->cards_in_trick > 0) ? SUIT[g->trick_cards[0]] : -1;
    int is_last = (g->trick_count == 8);
    return pick_card(hand, lead_suit, g->trump, gap_override, is_last);
}

/* Step with heuristic play — auto-play for non-learning seats.
 * opp_mode: 0=heuristic, 1=random, 2=aggressive(gap=+100), 3=passive(gap=-100),
 *           4=human-sim (85% heuristic, 15% random)
 * Returns the card played. */
EXPORT int rl_step_opponent(RLGameState *g, int opp_mode) {
    int card;
    switch (opp_mode) {
        case 1: card = rl_pick_random(g); break;
        case 2: card = rl_pick_heuristic_gap(g, 100); break;
        case 3: card = rl_pick_heuristic_gap(g, -100); break;
        case 4: /* human-sim: 85% heuristic, 15% random */
            if ((int)(rng_next(&g->rng) % 100) < 15)
                card = rl_pick_random(g);
            else
                card = rl_pick_heuristic(g);
            break;
        default: card = rl_pick_heuristic(g); break;
    }
    if (card >= 0) {
        rl_step(g, card);
    }
    return card;
}

/* Batch step opponents — steps all games where current_player != seat.
 * opp_modes: [n] per-game opponent mode (0-4).
 * Advances each non-seat game by ONE step.
 * Returns number of games stepped. */
EXPORT int rl_batch_step_opponents(RLGameState *games, int n,
                                   const int *opp_modes, int seat) {
    int i, count = 0;
    for (i = 0; i < n; i++) {
        if (games[i].phase != PHASE_PLAY) continue;
        if (games[i].current_player == seat) continue;
        rl_step_opponent(&games[i], opp_modes[i]);
        count++;
    }
    return count;
}

/* ── Expert hand score estimation (matches Python estimate_hand_score) ── */

static int expert_hand_score(hand_t hand, int trump) {
    hand_t trump_hand = hand & SUIT_MASK[trump];
    int trump_count = popcount64(trump_hand);

    /* Trump face value × 1.7 */
    int trump_face = 0;
    FOR_EACH_CARD(trump_hand, c)
        trump_face += PTS[c][1];  /* trump points */
    END_FOR
    int base = (int)(trump_face * 1.7f);

    /* Trump count bonus */
    if (trump_count >= 5) {
        int alt = 25 + trump_count * 5;
        if (alt > base) base = alt;
    } else if (trump_count >= 3) {
        base += (trump_count - 2) * 3;
    }

    /* Puur (Under of trump) and Nell (9 of trump) */
    int has_puur = 0, has_nell = 0;
    int puur_id = trump * 9 + 5;  /* Under = value_index 5 */
    int nell_id = trump * 9 + 3;  /* 9 = value_index 3 */
    if (hand & (1ULL << puur_id)) { has_puur = 1; base += 4; }
    if (hand & (1ULL << nell_id)) { has_nell = 1; base += 2; }

    /* Side suits analysis */
    int side_bonus = 0;
    int void_count = 0;
    int s;
    for (s = 0; s < 4; s++) {
        if (s == trump) continue;
        hand_t suit_hand = hand & SUIT_MASK[s];
        int suit_len = popcount64(suit_hand);

        if (suit_len == 0) { void_count++; continue; }

        int has_ace = (hand & (1ULL << (s * 9 + 8))) ? 1 : 0;  /* A = value_index 8 */
        int has_king = (hand & (1ULL << (s * 9 + 7))) ? 1 : 0; /* K = value_index 7 */
        int has_ten = (hand & (1ULL << (s * 9 + 4))) ? 1 : 0;  /* 10 = value_index 4 */

        if (has_ace) {
            side_bonus += (suit_len >= 2) ? 9 : 5;
        }
        if (has_king && suit_len >= 3) {
            side_bonus += 2;
        }
        if (has_ten && has_ace) {
            side_bonus += 4;
        }
    }

    /* Void suit bonus */
    if (trump_count >= 1 && void_count >= 1) {
        int usable = trump_count;
        if (has_puur) usable--;
        if (has_nell) usable--;
        if (usable < 0) usable = 0;
        int eff = (void_count < usable) ? void_count : usable;
        side_bonus += eff * 8;
    }

    int result = base + side_bonus;
    if (result < 0) result = 0;
    if (result > 157) result = 157;
    return result;
}

/* Batch expert declaration for all players in all games.
 * out: [n*4] array, out[i*4+p] = expert score for player p in game i */
EXPORT void rl_batch_expert_declare(const RLGameState *games, int n, int *out) {
    int i, p;
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            out[i * 4 + p] = expert_hand_score(games[i].initial_hands[p], games[i].trump);
        }
    }
}

/* Batch pick actions for C opponents (without stepping).
 * For each game where current_player != seat and phase == PLAY,
 * picks an action based on opp_mode. Writes to actions[i].
 * Games where current_player == seat get actions[i] = -1. */
EXPORT void rl_batch_pick_opponents(RLGameState *games, int n,
                                     const int *opp_modes, int seat,
                                     int *actions) {
    int i;
    for (i = 0; i < n; i++) {
        if (games[i].phase != PHASE_PLAY || games[i].current_player == seat) {
            actions[i] = -1;
            continue;
        }
        switch (opp_modes[i]) {
            case 1: actions[i] = rl_pick_random(&games[i]); break;
            case 2: actions[i] = rl_pick_heuristic_gap(&games[i], 100); break;
            case 3: actions[i] = rl_pick_heuristic_gap(&games[i], -100); break;
            case 4: /* human-sim: 85% heuristic, 15% random */
                if ((int)(rng_next(&games[i].rng) % 100) < 15)
                    actions[i] = rl_pick_random(&games[i]);
                else
                    actions[i] = rl_pick_heuristic(&games[i]);
                break;
            default: actions[i] = rl_pick_heuristic(&games[i]); break;
        }
    }
}

/* Batch step with action array. Steps all games where actions[i] >= 0.
 * Games with actions[i] < 0 are skipped. */
EXPORT void rl_batch_step_actions(RLGameState *games, int n, const int *actions) {
    int i;
    for (i = 0; i < n; i++) {
        if (actions[i] >= 0 && games[i].phase == PHASE_PLAY) {
            rl_step(&games[i], actions[i]);
        }
    }
}

/* Batch expert declare with noise and scale.
 * scale: multiplier (1.2=aggressive, 0.7=passive, 1.0=normal)
 * noise_range: uniform noise in [-noise_range, +noise_range] (0=none, 5=human-sim)
 * seed: for reproducible noise */
EXPORT void rl_batch_expert_declare_noisy(const RLGameState *games, int n,
                                          int *out, int noise_range,
                                          float scale, uint64_t seed) {
    rng_t rng;
    int i, p;
    rng_seed(&rng, seed);
    for (i = 0; i < n; i++) {
        for (p = 0; p < 4; p++) {
            int base = expert_hand_score(games[i].initial_hands[p], games[i].trump);
            int val = (int)(base * scale);
            if (noise_range > 0) {
                int noise = (int)(rng_next(&rng) % (uint64_t)(2 * noise_range + 1)) - noise_range;
                val += noise;
            }
            if (val < 0) val = 0;
            if (val > 157) val = 157;
            out[i * 4 + p] = val;
        }
    }
}

EXPORT int rl_gamestate_size(void) {
    return (int)sizeof(RLGameState);
}
