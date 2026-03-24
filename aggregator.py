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

def compute_passage_scores(scorer_output: dict,
                           decomp_a: dict,
                           decomp_b: dict,
                           wc_a: int,
                           wc_b: int) -> dict | None:
    """
    Aggregate scorer output into final R/C/D + confidence for one passage pair.
    Weight per matched pair = min(weight_a, weight_b)
    Weight per unmatched   = the sub-claim's own weight from decomp
    """
    matched     = scorer_output.get('matched_pairs', [])
    unmatched_a = scorer_output.get('unmatched_a', [])
    unmatched_b = scorer_output.get('unmatched_b', [])

    # normalise unmatched entries — scorer sometimes returns dicts instead of id strings
    unmatched_a = [u['id'] if isinstance(u, dict) else u for u in unmatched_a]
    unmatched_b = [u['id'] if isinstance(u, dict) else u for u in unmatched_b]

    weights_a = {sc['id']: sc['weight'] for sc in decomp_a.get('subclaims', [])}
    weights_b = {sc['id']: sc['weight'] for sc in decomp_b.get('subclaims', [])}

    tR=tC=tD=fR=fC=fD=total_w=0.0
    gate_confs = []

    for mp in matched:
        t_sum = mp['think']['R'] + mp['think']['C']
        f_sum = mp['feel']['R']  + mp['feel']['C']
        tR_n  = mp['think']['R'] / t_sum if t_sum > 0 else 0.5
        tC_n  = mp['think']['C'] / t_sum if t_sum > 0 else 0.5
        fR_n  = mp['feel']['R']  / f_sum if f_sum > 0 else 0.5
        fC_n  = mp['feel']['C']  / f_sum if f_sum > 0 else 0.5

        w   = min(mp['weight_a'], mp['weight_b'])
        tR += w * tR_n
        tC += w * tC_n
        fR += w * fR_n
        fC += w * fC_n
        total_w += w
        gate_confs.append(mp['gate_confidence'])

    for uid in unmatched_a:
        w = weights_a.get(uid, 0.10)
        tD += w; fD += w; total_w += w

    for uid in unmatched_b:
        w = weights_b.get(uid, 0.10)
        tD += w; fD += w; total_w += w

    if total_w == 0:
        return None

    think = round_to_100({'R': tR/total_w*100, 'C': tC/total_w*100, 'D': tD/total_w*100})
    feel  = round_to_100({'R': fR/total_w*100, 'C': fC/total_w*100, 'D': fD/total_w*100})

    n_sc_a   = len(decomp_a.get('subclaims', []))
    n_sc_b   = len(decomp_b.get('subclaims', []))
    depth    = min(wc_a, wc_b) / max(wc_a, wc_b) if max(wc_a, wc_b) > 0 else 0
    coverage = len(matched) / max(n_sc_a, n_sc_b) if max(n_sc_a, n_sc_b) > 0 else 0
    mq       = sum(gate_confs) / len(gate_confs) if gate_confs else 0.50
    conf     = min(0.95, max(0.20, round(0.30*depth + 0.40*coverage + 0.30*mq, 2)))

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