"""
pipeline_bias_check.py
======================
CI/CD Pipeline — Automated Bias Detection Gate

What this does:
  1. Imports model_bias_detection.py (the actual bias tool)
  2. Loads all_profile_scores_fixed.json (the model's output)
  3. Runs bias detection on Think AND Feel dimensions separately
  4. Runs per-book slicing (always enforced)
  5. Runs per-user demographic slicing (gender, age, personality, reader_type)
  6. Computes max disparity (gap between best and worst group)
  7. Logs everything + saves a JSON run record
  8. Sends email alert if bias is detected
  9. Exits 1 = deployment BLOCKED, or exits 0 = deployment proceeds

Can also be imported by another pipeline file:
    from pipeline_bias_check import run_bias_check
    run_bias_check(pairs_path="data/reports/all_profile_scores_fixed.json")
"""

import argparse
import json
import logging
import os
import smtplib
import sys
import traceback
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import pandas as pd

# ============================================================
# CONFIG
# ============================================================

# Deployment gate behaviour
BLOCK_ON_BIAS = True    # True  = block pipeline when bias found (strict)
                        # False = log warning and continue (lenient)

BIAS_THRESHOLD = 0.15   # flag if any group's F1 drops more than this below overall

# Max disparity: flag if best group F1 minus worst group F1 exceeds this
MAX_DISPARITY_THRESHOLD = 0.20
REQUIRED_SLICES = ["gender", "age_group", "personality", "reader_type"]

# File paths
PAIRS_FILE = "data/reports/all_profile_scores_fixed.json"
USERS_FILE = "data/users_processed.json"
OUTPUT_DIR = "data/bias_results"
LOG_DIR    = "data/logs"

# Email alert — set these as GitHub Actions secrets:
#   ALERT_EMAIL_SENDER    e.g. yourteam@gmail.com
#   ALERT_EMAIL_PASSWORD  Gmail app password (not your login password)
#   ALERT_EMAIL_TO        who receives the alert
#   SMTP_HOST / SMTP_PORT optional overrides (defaults to Gmail)
EMAIL_SENDER   = os.environ.get("ALERT_EMAIL_SENDER",   "")
EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")
EMAIL_TO       = os.environ.get("ALERT_EMAIL_TO",       "")
SMTP_HOST      = os.environ.get("SMTP_HOST",            "smtp.gmail.com")
SMTP_PORT      = int(os.environ.get("SMTP_PORT",        "587"))

# ============================================================
# LOGGING — writes to console AND logs/bias_check.log
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "bias_check.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_path, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("pipeline_bias_check")


# ============================================================
# 1. IMPORT MODEL BIAS DETECTION
# ============================================================
def _import_mbd():
    """
    Load model_bias_detection.py directly by file path.
    Override its threshold to match our pipeline config.
    Exits with code 1 if the import fails.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "model_bias_detection",
            os.path.join(os.path.dirname(__file__), "model_bias_detection.py")
        )
        mbd = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mbd)
        mbd.BIAS_FLAG_F1 = BIAS_THRESHOLD   # sync threshold
        return mbd
    except Exception:
        log.error("Could not load model_bias_detection.py.")
        log.error(f"Expected at: {os.path.join(os.path.dirname(__file__), 'model_bias_detection.py')}")
        log.error(traceback.format_exc())
        sys.exit(1)


# ============================================================
# 2. MAX DISPARITY CHECK
# ============================================================
def compute_max_disparity(findings: list) -> dict:
    """
    Scans all slice results (Think + Feel, all demographic columns)
    and finds the single best and worst group by F1.
    """
    all_groups = []
    for f in findings:
        if not f:
            continue
        for grp_row in f.get("by_group", []):
            grp_name = grp_row.get(f["slice"], grp_row.get("index", "unknown"))
            f1_val   = grp_row.get("f1_macro", 0.0)
            all_groups.append({
                "dimension": f["dimension"],
                "slice":     f["slice"],
                "group":     str(grp_name),
                "f1":        float(f1_val),
            })

    if len(all_groups) < 2:
        log.warning("  Not enough groups for disparity check.")
        return {"disparity": None, "best": None, "worst": None, "flagged": False}

    best    = max(all_groups, key=lambda x: x["f1"])
    worst   = min(all_groups, key=lambda x: x["f1"])
    gap     = round(best["f1"] - worst["f1"], 4)
    flagged = gap > MAX_DISPARITY_THRESHOLD

    log.info(f"  Max disparity : {gap:.3f}")
    log.info(f"  Best  group   : [{best['dimension']}] {best['group']} "
             f"in '{best['slice']}' @ F1 {best['f1']:.3f}")
    log.info(f"  Worst group   : [{worst['dimension']}] {worst['group']} "
             f"in '{worst['slice']}' @ F1 {worst['f1']:.3f}")

    if flagged:
        log.error(f"  ⚠  Disparity {gap:.3f} exceeds threshold {MAX_DISPARITY_THRESHOLD}")
    else:
        log.info(f"  ✓  Disparity within threshold ({MAX_DISPARITY_THRESHOLD})")

    return {"disparity": gap, "best": best, "worst": worst, "flagged": flagged}


# ============================================================
# 3. PER-BOOK SLICING (always enforced)
# ============================================================
def check_per_book(df: pd.DataFrame, mbd) -> list:
    """
    Loops over every unique book in the data.
    Within each book, runs analyse_slice() for both Think and Feel
    across every demographic column in REQUIRED_SLICES.

    Returns a flat list of finding dicts (same format as run_analysis).
    """
    if "book" not in df.columns:
        log.warning("  'book' column not found — skipping per-book check.")
        return []

    results = []
    books   = df["book"].dropna().unique()
    log.info(f"  Books found: {list(books)}")

    for book in books:
        subset = df[df["book"] == book].copy()
        log.info(f"\n  ── Book: {book}  ({len(subset)} records) ──")

        for dim in mbd.DIMENSIONS:
            for col in REQUIRED_SLICES:
                if col not in subset.columns:
                    continue
                report_sink = []
                result = mbd.analyse_slice(subset, col, dim, OUTPUT_DIR, report_sink)
                if result:
                    result["book_context"] = book
                    results.append(result)

    return results


# ============================================================
# 4. EMAIL ALERT
# ============================================================
def send_email_alert(run_id: str, biased_slices: list,
                     disparity: dict, failures: list):
    """
    Sends a plain-text email listing every biased slice,
    the max disparity, and all flagged groups with their F1 drops.
    If credentials are not set, logs a warning and skips silently.
    """
    if not EMAIL_SENDER or not EMAIL_TO or not EMAIL_PASSWORD:
        log.warning("Email credentials not set — skipping email alert.")
        log.warning("Set ALERT_EMAIL_SENDER / ALERT_EMAIL_PASSWORD / ALERT_EMAIL_TO as secrets.")
        return

    subject = f"[MOMENTO] ⚠ Bias Detected — Deployment Blocked (run {run_id})"

    lines = [
        "Momento ML Pipeline — Bias Detection Alert",
        f"Run ID   : {run_id}",
        f"Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Status   : DEPLOYMENT BLOCKED",
        "",
        "── Biased slices ──────────────────────────",
    ]
    for s in biased_slices:
        lines.append(f"  • {s}")

    if disparity.get("flagged"):
        lines += [
            "",
            "── Max disparity ──────────────────────────",
            f"  Gap   : {disparity['disparity']:.3f}",
            f"  Best  : [{disparity['best']['dimension']}] "
            f"{disparity['best']['group']} @ F1 {disparity['best']['f1']:.3f}",
            f"  Worst : [{disparity['worst']['dimension']}] "
            f"{disparity['worst']['group']} @ F1 {disparity['worst']['f1']:.3f}",
        ]

    lines += ["", "── Flagged groups ─────────────────────────"]
    for f in failures:
        lines.append(f"  • [{f['dimension']}][{f['slice']}] "
                     f"{f['group']} — F1 drop {f['f1_drop']:.3f}")

    lines += [
        "",
        f"Full report : {OUTPUT_DIR}/pipeline_run_{run_id}.json",
        f"Log file    : {log_path}",
        "",
        "Fix bias before re-running the pipeline.",
    ]

    try:
        msg            = MIMEMultipart()
        msg["From"]    = EMAIL_SENDER
        msg["To"]      = EMAIL_TO
        msg["Subject"] = subject
        msg.attach(MIMEText("\n".join(lines), "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_TO, msg.as_string())

        log.info(f"  Email alert sent → {EMAIL_TO}")
    except Exception as e:
        log.warning(f"  Failed to send email: {e}")
        log.warning("  Pipeline will still block — email is non-critical.")


# ============================================================
# MAIN GATE — run_bias_check()
# ============================================================
def run_bias_check(pairs_path=PAIRS_FILE, users_path=USERS_FILE):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    SEP    = "=" * 60

    log.info(SEP)
    log.info("  PIPELINE — Bias Detection Gate")
    log.info(f"  Run ID            : {run_id}")
    log.info(f"  Bias threshold    : F1 drop > {BIAS_THRESHOLD}")
    log.info(f"  Max disparity cap : {MAX_DISPARITY_THRESHOLD}")
    log.info(f"  Block on bias     : {BLOCK_ON_BIAS}")
    log.info(SEP)

    # ── Import the bias tool ──────────────────────────────────
    mbd = _import_mbd()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Guard: skip cleanly if model output doesn't exist yet ─
    if not pairs_path or not os.path.exists(pairs_path):
        log.warning(f"Pairs file not found at '{pairs_path}'.")
        log.warning("Skipping bias check — exiting 0 (no block).")
        log.warning("Re-run once the model has produced the output file.")
        sys.exit(0)

    # ── Load model output + user demographics ─────────────────
    log.info(f"\nLoading data...")
    try:
        demo = mbd.load_user_demographics(users_path)
        gt   = mbd.load_ground_truth(mbd.GROUND_TRUTH_FILE)
        df   = mbd.load_pairs(pairs_path, demo, gt)
        log.info(f"Total pairs loaded: {len(df)}")
    except Exception:
        log.error(f"Failed to load data:\n{traceback.format_exc()}")
        sys.exit(1)

    # ── Run bias detection: Think + Feel, all slices ──────────
    log.info(f"\n{SEP}")
    log.info("  Running bias detection (Think + Feel dimensions)...")
    log.info(SEP)

    report   = []
    findings = []

    # Overall demographic slices (Think + Feel)
    findings += mbd.run_analysis(df, OUTPUT_DIR, report)

    # Per-book slices (always enforced)
    log.info(f"\n{SEP}")
    log.info("  Per-book slicing (always enforced)...")
    log.info(SEP)
    findings += check_per_book(df, mbd)

    # ── Max disparity check ───────────────────────────────────
    log.info(f"\n{SEP}")
    log.info("  Max Disparity Check...")
    log.info(SEP)
    disparity = compute_max_disparity(findings)

    # ── Collect results ───────────────────────────────────────
    biased_slices = list({
        f"{f['dimension']}:{f['slice']}"
        for f in findings if f and f.get("bias_status") == "BIAS DETECTED"
    })
    clean_slices = list({
        f"{f['dimension']}:{f['slice']}"
        for f in findings if f and f.get("bias_status") == "NO BIAS"
    })

    failures = [
        {
            "dimension": f["dimension"],
            "slice":     f["slice"],
            "group":     g.get(f["slice"], g.get("index", "?")),
            "f1":        g.get("f1_macro", 0),
            "f1_drop":   round(f["overall_f1"] - g.get("f1_macro", 0), 4),
        }
        for f in findings if f
        for g in f.get("flagged", [])
    ]

    bias_detected = bool(biased_slices) or disparity.get("flagged", False)

    # ── Summary ───────────────────────────────────────────────
    log.info(f"\n{SEP}")
    log.info("  SUMMARY")
    log.info(SEP)
    log.info(f"  Clean  : {clean_slices  or 'none'}")
    log.info(f"  Biased : {biased_slices or 'none'}")
    log.info(f"  Max disparity : {disparity.get('disparity')} "
             f"({'FLAGGED' if disparity.get('flagged') else 'OK'})")
    log.info(f"  Verdict : {'BLOCKED' if bias_detected else 'PASSED'}")

    # ── Save run record ───────────────────────────────────────
    record = {
        "run_id":        run_id,
        "timestamp":     datetime.now().isoformat(),
        "pairs_path":    pairs_path,
        "threshold":     BIAS_THRESHOLD,
        "max_disparity": MAX_DISPARITY_THRESHOLD,
        "block_on_bias": BLOCK_ON_BIAS,
        "clean_slices":  clean_slices,
        "biased_slices": biased_slices,
        "disparity":     disparity,
        "failures":      failures,
        "verdict":       "BLOCKED" if bias_detected else "PASSED",
    }
    record_path = os.path.join(OUTPUT_DIR, f"pipeline_run_{run_id}.json")
    with open(record_path, "w") as fh:
        json.dump(record, fh, indent=2)
    log.info(f"\n  Run record → {record_path}")

    # ── Email + block/pass ────────────────────────────────────
    if bias_detected:
        send_email_alert(run_id, biased_slices, disparity, failures)

        if BLOCK_ON_BIAS:
            log.error(SEP)
            log.error("  BIAS CHECK FAILED — Deployment BLOCKED")
            log.error(f"  Report : {record_path}")
            log.error(SEP)
            sys.exit(1)
        else:
            log.warning("  BLOCK_ON_BIAS = False — warning logged, pipeline continues.")
            return record
    else:
        log.info(SEP)
        log.info("  BIAS CHECK PASSED!")
        log.info(SEP)
        sys.exit(0)


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Momento — Pipeline: Bias Detection Gate"
    )
    parser.add_argument(
        "--pairs",
        default=PAIRS_FILE,
        help="Path to model output JSON"
    )
    parser.add_argument(
        "--users",
        default=USERS_FILE,
        help="Path to users_processed.json (optional — skipped if not found)"
    )
    args = parser.parse_args()

    run_bias_check(pairs_path=args.pairs, users_path=args.users)