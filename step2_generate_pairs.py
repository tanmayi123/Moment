import json
from itertools import combinations

# ── Load portraits ────────────────────────────────────────────
with open("output/portraits.json", "r") as f:
    portraits = json.load(f)

# ── Group users by book ───────────────────────────────────────
users_by_book = {}
for portrait in portraits:
    book = portrait["book_title"]
    if book not in users_by_book:
        users_by_book[book] = []
    users_by_book[book].append(portrait)

# ── Generate pairs per book ───────────────────────────────────
pairs = []

for book, book_portraits in users_by_book.items():
    print(f"\nBook: {book}")
    print(f"  Users with portraits: {len(book_portraits)}")

    book_pairs = list(combinations(book_portraits, 2))
    print(f"  Candidate pairs: {len(book_pairs)}")

    for portrait_a, portrait_b in book_pairs:
        pairs.append({
            "book_title": book,
            "user_a_id": portrait_a["user_id"],
            "user_a_name": portrait_a["character_name"],
            "user_b_id": portrait_b["user_id"],
            "user_b_name": portrait_b["character_name"],
            "portrait_a": portrait_a["portrait"],
            "portrait_b": portrait_b["portrait"]
        })

print(f"\n✅ Total candidate pairs: {len(pairs)}")

# ── Save output ───────────────────────────────────────────────
with open("output/candidate_pairs.json", "w") as f:
    json.dump(pairs, f, indent=2)

print(f"✅ Saved to output/candidate_pairs.json")