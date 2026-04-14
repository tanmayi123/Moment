"""
preprocessor_fastapi.py — MOMENT Preprocessing (FastAPI version)
=================================================================
Config hardcoded. Input: DataFrames from CloudSQLLoader.
Output: (moments, passages, books, users) as lists of dicts.
"""

import re
import hashlib
import logging
import unicodedata
from datetime import datetime
from typing import Tuple, List

import pandas as pd
import textstat
from langdetect import detect, LangDetectException

logger = logging.getLogger(__name__)

CFG = {
    "validation": {
        "min_words":         10,
        "max_words":         600,
        "min_chars":         50,
        "quality_threshold": 0.5,
    },
    "issues": {
        "profanity_ratio_threshold": 0.90,
        "caps_threshold":            0.90,
        "punct_threshold":           0.90,
        "repetitive_chars":          25,
        "repetitive_words_threshold":0.90,
    },
    "timestamp_format": "%Y-%m-%dT%H:%M:%S",
}

PROFANITY = {
    "damn", "hell", "crap", "ass", "bastard", "bitch",
    "shit", "fuck", "piss", "dick", "cock", "cunt",
    "whore", "slut", "fag", "retard"
}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text):
    if not text:
        return ""
    text = str(text)
    try:
        text = text.encode("latin-1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        pass
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2014", "--").replace("\u2013", "-")
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u2026", "...").replace("\u00a0", " ").replace("\u200b", "")
    text = re.sub(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
                  "[EMAIL REMOVED]", text)
    text = text.replace("\t", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_text(text):
    v = CFG["validation"]
    if not text or not text.strip():
        return {"is_valid": False, "quality_score": 0.0,
                "quality_issues": ["empty_text"], "word_count": 0, "language": "unknown"}

    issues = []
    words = text.split()
    word_count = len(words)

    if word_count < v["min_words"]:
        issues.append(f"too_short: {word_count} words")
    elif word_count > v["max_words"]:
        issues.append(f"too_long: {word_count} words")

    if len(text.replace(" ", "")) < v["min_chars"]:
        issues.append("too_few_chars")

    language = "en"
    if len(text) >= 20:
        try:
            language = detect(text)
            if language != "en":
                issues.append(f"wrong_language: {language}")
        except LangDetectException:
            language = "unknown"

    letters = [c.lower() for c in text if c.isalpha()]
    if len(letters) >= 10:
        vowel_ratio = sum(1 for c in letters if c in "aeiou") / len(letters)
        if vowel_ratio < 0.15 or vowel_ratio > 0.60:
            issues.append("gibberish_detected")

    score = 1.0
    for issue in issues:
        if "too_short" in issue or "too_long" in issue:
            score -= 0.3
        elif "wrong_language" in issue:
            score -= 0.4
        elif "gibberish" in issue:
            score -= 0.5
        elif "too_few_chars" in issue:
            score -= 0.1
    score = max(0.0, min(1.0, score))

    return {
        "is_valid": len(issues) == 0 and score >= v["quality_threshold"],
        "quality_score": round(score, 4),
        "quality_issues": "; ".join(issues) if issues else "",
        "word_count": word_count,
        "language": language,
    }


# ── Issue detection ───────────────────────────────────────────────────────────

def detect_issues(text):
    ic = CFG["issues"]
    if not text:
        return {"has_pii": False, "pii_types": "", "has_profanity": False,
        "profanity_ratio": 0.0, "is_spam": False, "spam_reasons": ""}

    pii_types, spam_reasons = [], []

    if re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text):
        pii_types.append("email")
    if re.search(r"\b\d{3}[-\s]\d{2}[-\s]\d{4}\b", text):
        pii_types.append("ssn")

    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    profane = [w for w in words if w in PROFANITY]
    profanity_ratio = round(len(profane) / max(len(words), 1), 4)
    has_profanity = len(profane) > 0 and profanity_ratio >= ic["profanity_ratio_threshold"]

    letters = [c for c in text if c.isalpha()]
    if letters:
        caps_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
        if caps_ratio > ic["caps_threshold"]:
            spam_reasons.append(f"excessive_caps: {caps_ratio:.1%}")

    return {
    "has_pii":        len(pii_types) > 0,
    "pii_types":      "; ".join(pii_types) if pii_types else "",
    "has_profanity":  has_profanity,
    "profanity_ratio":profanity_ratio,
    "is_spam":        len(spam_reasons) > 0,
    "spam_reasons":   "; ".join(spam_reasons) if spam_reasons else "",
}


# ── Prompt injection detection ───────────────────────────────────────────────

_INJECTION_PATTERNS = [
    # instruction override
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"disregard\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"forget\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)",
    r"do\s+not\s+follow\s+(the\s+)?(previous|prior|above|earlier)\s+(instructions?|rules?)",
    r"new\s+instructions?\s*:",
    r"your\s+(new\s+)?(instructions?|objective|goal|task|purpose|role)\s+(is|are)\s*:",
    # role / persona hijack
    r"you\s+are\s+now\s+[a-z]",
    r"act\s+as\s+(a\s+|an\s+)?[a-z]",
    r"pretend\s+(you\s+are|to\s+be)\s+[a-z]",
    r"(developer|jailbreak|unrestricted|dan)\s+mode",
    r"\bDAN\b",
    # system prompt impersonation
    r"(^|\n)\s*system\s*:",
    r"(^|\n)\s*###\s*(instruction|system|prompt)",
    r"<\|im_start\|>",
    r"\[INST\]",
    r"(^|\n)\s*(USER|ASSISTANT|HUMAN|AI)\s*:",
]

_INJECTION_RE = re.compile(
    "|".join(_INJECTION_PATTERNS),
    re.IGNORECASE | re.MULTILINE,
)

def has_prompt_injection(text: str) -> bool:
    if not text:
        return False
    return bool(_INJECTION_RE.search(text))


# ── Metrics ───────────────────────────────────────────────────────────────────

def calculate_metrics(text):
    if not text or not text.strip():
        return {"word_count": 0, "char_count": 0, "sentence_count": 0,
                "avg_word_length": 0.0, "avg_sentence_length": 0.0,
                "readability_score": 0.0}

    words = text.split()
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) >= 3]
    if not sentences:
        sentences = [text]

    word_count      = len(words)
    char_count      = len(text.replace(" ", "").replace("\n", ""))
    sentence_count  = len(sentences)
    alpha_chars     = sum(len(re.sub(r"[^a-zA-Z]", "", w)) for w in words)
    avg_word_length = round(alpha_chars / max(word_count, 1), 2)
    avg_sent_length = round(word_count / max(sentence_count, 1), 2)

    try:
        readability = max(0.0, min(100.0, textstat.flesch_reading_ease(text)))
    except Exception:
        readability = 0.0

    return {
        "word_count":         word_count,
        "char_count":         char_count,
        "sentence_count":     sentence_count,
        "avg_word_length":    avg_word_length,
        "avg_sentence_length":avg_sent_length,
        "readability_score":  round(readability, 2),
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def preprocess_all(
    moments_df: pd.DataFrame,
    books_df:   pd.DataFrame,
    users_df:   pd.DataFrame,
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    """Returns (moments, passages, books, users)"""
    ts = datetime.now().strftime(CFG["timestamp_format"])
    moments, passages = _process_moments(moments_df, ts)
    books             = _process_books(books_df, ts)
    users             = _process_users(users_df, ts)
    return moments, passages, books, users


def _process_moments(df: pd.DataFrame, ts: str) -> Tuple[List[dict], List[dict]]:
    """
    Split moments into:
    - moments_processed: all fields except passage text
    - passages_processed: book_id, passage_id, passage text
    """
    moments  = []
    passages = []
    seen_passages = set()  # deduplicate passages

    for _, row in df.iterrows():
        try:
            moment_id  = str(row.get("moment_id", ""))
            user_id    = str(row.get("user_id", ""))    # UUID, no hashing
            book_id    = str(row.get("book_id", ""))    # books.id UUID
            passage_id = str(row.get("passage_id", "")) # passage_key
            passage    = str(row.get("passage", ""))
            raw_text   = str(row.get("interpretation", ""))

            cleaned = clean_text(raw_text)

            if has_prompt_injection(cleaned):
                logger.warning(
                    f"Prompt injection detected — skipping moment_id={moment_id} user_id={user_id}"
                )
                continue

            validation = validate_text(cleaned)
            issues     = detect_issues(cleaned)
            metrics    = calculate_metrics(cleaned)

            if not validation["is_valid"]:
                logger.warning(f"Invalid text — skipping moment_id={moment_id}: {validation['quality_issues']}")
                continue
            if issues["has_pii"]:
                logger.warning(f"PII detected — skipping moment_id={moment_id}: {issues['pii_types']}")
                continue
            if issues["has_profanity"]:
                logger.warning(f"Profanity detected — skipping moment_id={moment_id}")
                continue
            if issues["is_spam"]:
                logger.warning(f"Spam detected — skipping moment_id={moment_id}: {issues['spam_reasons']}")
                continue

            # ── moments_processed ────────────────────────────────
            moments.append({
                "moment_id":              moment_id,
                "user_id":                user_id,
                "book_id":                book_id,
                "passage_id":             passage_id,
                "chapter":                str(row.get("chapter", "")),
                "page_num": int(row["page_num"]) if row.get("page_num") not in (None, "", "t", "f") else None,
                "cleaned_interpretation": cleaned,
                "original_word_count":    row.get("word_count", 0),
                "is_valid":               validation["is_valid"],
                "quality_score":          validation["quality_score"],
                "quality_issues":         validation["quality_issues"],
                "language":               validation["language"],
                "has_pii":                issues["has_pii"],
                "pii_types":              issues["pii_types"],
                "has_profanity":          issues["has_profanity"],
                "profanity_ratio":        issues["profanity_ratio"],
                "is_spam":                issues["is_spam"],
                "spam_reasons":           issues["spam_reasons"],
                "word_count":             metrics["word_count"],
                "char_count":             metrics["char_count"],
                "sentence_count":         metrics["sentence_count"],
                "avg_word_length":        metrics["avg_word_length"],
                "avg_sentence_length":    metrics["avg_sentence_length"],
                "readability_score":      metrics["readability_score"],
                "created_at":             str(row.get("created_at", "")),
                "timestamp":              ts,
            })

            # ── passages_processed (deduplicated by passage_id) ──
            if passage_id and passage_id not in seen_passages:
                seen_passages.add(passage_id)
                passages.append({
                    "passage_id": passage_id,
                    "book_id":    book_id,
                    "passage":    passage,
                })

        except Exception as e:
            logger.error(f"Error processing moment row: {e}")
            continue

    logger.info(f"Processed {len(moments)} moments, {len(passages)} passages")
    return moments, passages


def _process_books(df: pd.DataFrame, ts: str) -> List[dict]:
    """
    books_processed:
    - book_id = books.id (UUID)
    - everything else as-is
    """
    records = []
    seen_books = set()
    for _, row in df.iterrows():
        try:
            book_id = str(row.get("book_id", ""))
            if book_id and book_id in seen_books:
                logger.warning(f"Duplicate book_id — skipping book_id={book_id}")
                continue
            if book_id:
                seen_books.add(book_id)
            records.append({
                "book_id":        book_id,
                "book_title":     str(row.get("book_title", "")),
                "book_author":    str(row.get("book_author", "")),
                "year": int(row["year"]) if row.get("year") not in (None, "", "t", "f") else None,
                "gutenberg_id":   str(row.get("gutenberg_id", "")),
                "cover_url":      str(row.get("cover_url", "")),
                "opening_passage":str(row.get("opening_passage", "")),
                "epub_url":       str(row.get("epub_url", "")),
                "text_url":       str(row.get("text_url", "")),
                "timestamp":      ts,
            })
        except Exception as e:
            logger.error(f"Error processing book row: {e}")
            continue

    logger.info(f"Processed {len(records)} books")
    return records


def _process_users(df: pd.DataFrame, ts: str) -> List[dict]:
    """
    users_processed:
    - user_id = users.id (UUID, same as moments.user_id — NO hashing)
    - all other columns as-is
    """
    records = []
    seen_users = set()
    for _, row in df.iterrows():
        try:
            user_id = str(row.get("user_id", ""))
            if user_id and user_id in seen_users:
                logger.warning(f"Duplicate user_id — skipping user_id={user_id}")
                continue
            if user_id:
                seen_users.add(user_id)
            records.append({
                "user_id":                user_id,
                "firebase_uid":           str(row.get("firebase_uid", "")),
                "first_name":             str(row.get("first_name", "")),
                "last_name":              str(row.get("last_name", "")),
                "readername":             str(row.get("readername", "")),
                "bio":                    str(row.get("bio", "")),
                "gender":                 str(row.get("gender", "")),
                "photo_url":              str(row.get("photo_url", "")),
                "dark_mode":              bool(row.get("dark_mode", False)),
                "moments_layout_mode":    str(row.get("moments_layout_mode", "")),
                "passage_first":          bool(row.get("passage_first", False)),
                "last_read_book_id":      str(row.get("last_read_book_id", "")),
                "onboarding_complete":    bool(row.get("onboarding_complete", False)),
                "consent_given":          bool(row.get("consent_given", False)),
                "consent_at":             str(row.get("consent_at", "")),
                "created_at":             str(row.get("created_at", "")),
                "last_login_at":          str(row.get("last_login_at", "")),
                "last_hero_gut_id":       str(row.get("last_hero_gut_id", "")),
                "guide_book_gut_id":      str(row.get("guide_book_gut_id", "")),
                "reading_state":          str(row.get("reading_state", "")),
                "last_captured_type":     str(row.get("last_captured_type", "")),
                "last_captured_shelf_id": str(row.get("last_captured_shelf_id", "")),
                "timestamp":              ts,
            })
        except Exception as e:
            logger.error(f"Error processing user row: {e}")
            continue

    logger.info(f"Processed {len(records)} users")
    return records