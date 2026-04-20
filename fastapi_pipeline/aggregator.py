from datetime import datetime

LABEL_MAP = {'R': 'resonate', 'C': 'contradict', 'D': 'diverge'}

# ── Helpers ───────────────────────────────────────────────────────────────────

def round_to_100(d: dict) -> dict:
    """Round float percentages to ints that sum exactly to 100."""
    rounded = {k: round(v) for k, v in d.items()}
    diff = 100 - sum(rounded.values())
    if diff:
        largest = max(rounded, key=rounded.get)
        rounded[largest] += diff
    return rounded


# ── Passage-level (Steps 4 + 5) ──────────────────────────────────────────────

THINK_ADDS_R = [True, True, True, False, True]  # T4 reversed
FEEL_ADDS_R  = [True, True, True, True, True]

def score_from_bools(qs: list[bool], adds_r: list[bool]) -> tuple[float, float]:
    R = sum(1 for q, r in zip(qs, adds_r) if q == r) / 5
    return R, 1 - R

def unmatched_d(u: dict) -> float:
    return 1.0 if u.get('divergence', False) else 0.3

def compute_passage_scores(scorer_output: dict,
                           decomp_a: dict,
                           decomp_b: dict,
                           wc_a: int,
                           wc_b: int) -> dict | None:
    """
    Aggregate scorer output into final R/C/D + confidence for one passage pair.
    scorer_output contains think_q/feel_q boolean arrays per matched pair.
    """
    matched     = scorer_output.get('matched_pairs', [])
    unmatched_a = scorer_output.get('unmatched_a', [])
    unmatched_b = scorer_output.get('unmatched_b', [])

    weights_a = {sc['id']: sc['weight'] for sc in decomp_a.get('subclaims', [])}
    weights_b = {sc['id']: sc['weight'] for sc in decomp_b.get('subclaims', [])}

    tR=tC=tD=fR=fC=fD=total_w=0.0
    for mp in matched:
        tR_raw, tC_raw = score_from_bools(mp['think_q'], THINK_ADDS_R)
        fR_raw, fC_raw = score_from_bools(mp['feel_q'],  FEEL_ADDS_R)

        w   = min(mp['weight_a'], mp['weight_b'])
        tR += w * tR_raw
        tC += w * tC_raw
        fR += w * fR_raw
        fC += w * fC_raw
        total_w += w
    for u in unmatched_a:
        uid = u['id'] if isinstance(u, dict) else u
        d   = unmatched_d(u) if isinstance(u, dict) else 0.3
        w   = weights_a.get(uid, 0.10)
        tD += d * w; fD += d * w; total_w += d * w

    for u in unmatched_b:
        uid = u['id'] if isinstance(u, dict) else u
        d   = unmatched_d(u) if isinstance(u, dict) else 0.3
        w   = weights_b.get(uid, 0.10)
        tD += d * w; fD += d * w; total_w += d * w

    if total_w == 0:
        return None

    think = round_to_100({'R': tR/total_w*100, 'C': tC/total_w*100, 'D': tD/total_w*100})
    feel  = round_to_100({'R': fR/total_w*100, 'C': fC/total_w*100, 'D': fD/total_w*100})

    n_sc_a   = len(decomp_a.get('subclaims', []))
    n_sc_b   = len(decomp_b.get('subclaims', []))
    depth    = min(wc_a, wc_b) / max(wc_a, wc_b) if max(wc_a, wc_b) > 0 else 0
    coverage = len(matched) / max(n_sc_a, n_sc_b) if max(n_sc_a, n_sc_b) > 0 else 0
    dominance = (max(think.values()) + max(feel.values())) / 200
    conf     = min(0.95, max(0.20, round(0.30*depth + 0.40*coverage + 0.30*dominance, 2)))

    return {
        'think':          think,
        'feel':           feel,
        'dominant_think': LABEL_MAP[max(think, key=think.get)],
        'dominant_feel':  LABEL_MAP[max(feel,  key=feel.get)],
        'match_count':    len(matched),
        'confidence':     conf,
        'computed_at':    datetime.utcnow().isoformat(),
    }


# ── Top-level aggregate (called from compatibility_agent.py) ──────────────────

def aggregate(decomp_a: dict,
              decomp_b: dict,
              scoring: dict,
              book_id: str = '',
              passage_id: str = '') -> dict | None:
    """
    Entry point called by run_compatibility_agent.
    Wraps compute_passage_scores + aggregate_book + aggregate_profile
    for a single passage pair.
    """
    wc_a = sum(len(s['claim'].split()) for s in decomp_a.get('subclaims', []))
    wc_b = sum(len(s['claim'].split()) for s in decomp_b.get('subclaims', []))

    ps = compute_passage_scores(scoring, decomp_a, decomp_b, wc_a, wc_b)
    if ps is None:
        return None
    print(ps)
    ps['passage_id'] = passage_id
    ps['book_id']    = book_id
    return ps

# ── Book-level aggregation ────────────────────────────────────────────────────

from collections import defaultdict

def aggregate_book_level(passage_results: list[dict]) -> list[dict]:
    """
    Given passage-level compatibility_results rows, group by
    (user_a, user_b, book_id) and average R/C/D scores.
    Input rows must have: user_a, user_b, book_id,
    think_R, think_C, think_D, feel_R, feel_C, feel_D, confidence.
    Returns list of book-level dicts ready for BQ insert.
    """
    groups = defaultdict(list)
    for r in passage_results:
        ua  = str(r.get('user_a', ''))
        ub  = str(r.get('user_b', ''))
        bid = str(r.get('book_id', ''))
        key = (min(ua, ub), max(ua, ub), bid)
        groups[key].append(r)

    book_rows = []
    for (ua, ub, bid), rows in groups.items():
        n = len(rows)
        if n == 0:
            continue

        def avg(field):
            vals = [float(r.get(field, 0)) for r in rows if isinstance(r.get(field, 0), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        think_R    = avg('think_R')
        think_C    = avg('think_C')
        think_D    = avg('think_D')
        feel_R     = avg('feel_R')
        feel_C     = avg('feel_C')
        feel_D     = avg('feel_D')
        confidence = avg('confidence')

        think          = {'R': round(think_R), 'C': round(think_C), 'D': round(think_D)}
        feel           = {'R': round(feel_R),  'C': round(feel_C),  'D': round(feel_D)}
        dominant_think = LABEL_MAP[max(think, key=think.get)]
        dominant_feel  = LABEL_MAP[max(feel,  key=feel.get)]

        overall_r = (think_R + feel_R) / 2
        overall_c = (think_C + feel_C) / 2
        overall_d = (think_D + feel_D) / 2
        overall   = {'resonate': overall_r, 'contradict': overall_c, 'diverge': overall_d}
        verdict   = max(overall, key=overall.get)

        book_rows.append({
            'user_a':         ua,
            'user_b':         ub,
            'book_id':        bid,
            'think_R':        think_R,
            'think_C':        think_C,
            'think_D':        think_D,
            'feel_R':         feel_R,
            'feel_C':         feel_C,
            'feel_D':         feel_D,
            'dominant_think': dominant_think,
            'dominant_feel':  dominant_feel,
            'verdict':        verdict,
            'confidence':     confidence,
            'passage_count':  n,
            'timestamp':      datetime.utcnow().isoformat(),
        })
    return book_rows


# ── Profile-level aggregation ─────────────────────────────────────────────────

def aggregate_profile_level(book_results: list[dict]) -> list[dict]:
    """
    Given book-level compatibility rows (output of aggregate_book_level),
    group by (user_a, user_b) across all books and average R/C/D scores.
    Returns list of profile-level dicts ready for BQ insert.
    """
    groups = defaultdict(list)
    for r in book_results:
        ua  = str(r.get('user_a', ''))
        ub  = str(r.get('user_b', ''))
        key = (min(ua, ub), max(ua, ub))
        groups[key].append(r)

    profile_rows = []
    for (ua, ub), rows in groups.items():
        n = len(rows)
        if n == 0:
            continue

        def avg(field):
            vals = [float(r.get(field, 0)) for r in rows if isinstance(r.get(field, 0), (int, float))]
            return round(sum(vals) / len(vals), 2) if vals else 0.0

        think_R    = avg('think_R')
        think_C    = avg('think_C')
        think_D    = avg('think_D')
        feel_R     = avg('feel_R')
        feel_C     = avg('feel_C')
        feel_D     = avg('feel_D')
        confidence = avg('confidence')

        think          = {'R': round(think_R), 'C': round(think_C), 'D': round(think_D)}
        feel           = {'R': round(feel_R),  'C': round(feel_C),  'D': round(feel_D)}
        dominant_think = LABEL_MAP[max(think, key=think.get)]
        dominant_feel  = LABEL_MAP[max(feel,  key=feel.get)]

        overall_r = (think_R + feel_R) / 2
        overall_c = (think_C + feel_C) / 2
        overall_d = (think_D + feel_D) / 2
        overall   = {'resonate': overall_r, 'contradict': overall_c, 'diverge': overall_d}
        verdict   = max(overall, key=overall.get)

        books = list({str(r.get('book_id', '')) for r in rows if r.get('book_id')})

        profile_rows.append({
            'user_a':         ua,
            'user_b':         ub,
            'think_R':        think_R,
            'think_C':        think_C,
            'think_D':        think_D,
            'feel_R':         feel_R,
            'feel_C':         feel_C,
            'feel_D':         feel_D,
            'dominant_think': dominant_think,
            'dominant_feel':  dominant_feel,
            'verdict':        verdict,
            'confidence':     confidence,
            'passage_count':  n,
            'book_count':     len(books),
            'timestamp':      datetime.utcnow().isoformat(),
        })
    return profile_rows