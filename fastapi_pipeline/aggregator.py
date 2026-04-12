from datetime import datetime

LABEL_MAP = {'R': 'resonate', 'C': 'contradict', 'D': 'diverge'}

THINK_ADDS_R = [True, True, True, False, True]  # T4 reversed
FEEL_ADDS_R  = [True, True, True, True, True]

def round_to_100(d: dict) -> dict:
    rounded = {k: round(v) for k, v in d.items()}
    diff = 100 - sum(rounded.values())
    if diff:
        largest = max(rounded, key=rounded.get)
        rounded[largest] += diff
    return rounded

def score_from_bools(qs: list[bool], adds_r: list[bool]) -> tuple[float, float]:
    R = sum(1 for q, r in zip(qs, adds_r) if q == r) / 5
    return R, 1 - R

def unmatched_d(u: dict) -> float:
    return 1.0 if u.get('divergence', False) else 0.3


# ── Passage-level ─────────────────────────────────────────────────────────────

def compute_passage_scores(scorer_output: dict,
                           decomp_a: dict,
                           decomp_b: dict,
                           wc_a: int,
                           wc_b: int) -> dict | None:

    matched     = scorer_output.get('matched_pairs', [])
    unmatched_a = scorer_output.get('unmatched_a', [])
    unmatched_b = scorer_output.get('unmatched_b', [])

    weights_a = {sc['id']: sc['weight'] for sc in decomp_a.get('subclaims', [])}
    weights_b = {sc['id']: sc['weight'] for sc in decomp_b.get('subclaims', [])}

    tR=tC=tD=fR=fC=fD=total_w=0.0

    for mp in matched:
        tR_raw, tC_raw = score_from_bools(mp['think_q'], THINK_ADDS_R)
        fR_raw, fC_raw = score_from_bools(mp['feel_q'],  FEEL_ADDS_R)

        w = min(mp['weight_a'], mp['weight_b'])
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

    # confidence — signal dominance + coverage + depth (no gate_confidence)
    n_sc_a   = len(decomp_a.get('subclaims', []))
    n_sc_b   = len(decomp_b.get('subclaims', []))
    depth    = min(wc_a, wc_b) / max(wc_a, wc_b) if max(wc_a, wc_b) > 0 else 0
    coverage = len(matched) / max(n_sc_a, n_sc_b) if max(n_sc_a, n_sc_b) > 0 else 0
    dominance = (max(think.values()) + max(feel.values())) / 200  # 0-1
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


# ── Top-level ─────────────────────────────────────────────────────────────────

def aggregate(decomp_a: dict,
              decomp_b: dict,
              scoring: dict,
              book_id: str = '',
              passage_id: str = '') -> dict | None:

    wc_a = sum(len(s['claim'].split()) for s in decomp_a.get('subclaims', []))
    wc_b = sum(len(s['claim'].split()) for s in decomp_b.get('subclaims', []))

    ps = compute_passage_scores(scoring, decomp_a, decomp_b, wc_a, wc_b)
    if ps is None:
        return None

    ps['passage_id'] = passage_id
    ps['book_id']    = book_id
    ps['think_rationale']=scoring['think_rationale']
    ps['feel_rationale'] = scoring['feel_rationale']
    return ps