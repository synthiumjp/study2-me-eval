"""
Run this locally on your machine where ao-childes.txt is available.
Usage: python corpus_analysis_script.py /path/to/ao-childes.txt

Outputs a JSON file (corpus_noun_stats.json) that you upload to Claude
for integration into the paper.
"""
import sys
import json
import re
from collections import defaultdict, Counter

if len(sys.argv) < 2:
    print("Usage: python corpus_analysis_script.py /path/to/ao-childes.txt")
    sys.exit(1)

corpus_path = sys.argv[1]

# The nouns used in the ME evaluation battery
# (from me_item_battery.json / analysis_report.txt)
TARGET_NOUNS = [
    'ball', 'book', 'car', 'hat', 'cup',
    'dog', 'cat', 'fish', 'bird',
    # synonyms relevant to per-item analysis
    'puppy', 'kitty', 'doggy',
]

print(f"Loading corpus from {corpus_path}...")
with open(corpus_path, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"Loaded {len(lines)} lines")

# Tokenize simply (lowercase, split on whitespace/punctuation)
def tokenize(line):
    return re.findall(r"[a-z']+", line.lower())

# ── 1. Raw frequency of each target noun ──
freq = Counter()
for line in lines:
    tokens = tokenize(line)
    for t in tokens:
        if t in TARGET_NOUNS:
            freq[t] += 1

print("\n=== NOUN FREQUENCIES ===")
for noun in sorted(TARGET_NOUNS, key=lambda n: -freq[n]):
    print(f"  {noun}: {freq[noun]}")

# ── 2. Repetition within 3/5/10 sentence windows ──
# Group lines by transcript (assume blank-line or filename separators;
# if ao-childes is one line per sentence continuously, treat as one block)
# For safety, use a sliding window over all lines

def repetition_rate(window_size):
    """For each target noun occurrence, what fraction have the same noun
    appearing again within the next `window_size` sentences?"""
    hits = defaultdict(int)
    total = defaultdict(int)
    
    for i, line in enumerate(lines):
        tokens = set(tokenize(line))
        for noun in TARGET_NOUNS:
            if noun in tokens:
                total[noun] += 1
                # Check next window_size lines
                for j in range(i+1, min(i+1+window_size, len(lines))):
                    future_tokens = set(tokenize(lines[j]))
                    if noun in future_tokens:
                        hits[noun] += 1
                        break
    return hits, total

print("\n=== REPETITION RATES (within 3 sentences) ===")
hits3, total3 = repetition_rate(3)
rates_3 = {}
for noun in sorted(TARGET_NOUNS, key=lambda n: -freq[n]):
    if total3[noun] > 0:
        rate = hits3[noun] / total3[noun]
        rates_3[noun] = rate
        print(f"  {noun}: {rate:.1%} ({hits3[noun]}/{total3[noun]})")

print("\n=== REPETITION RATES (within 5 sentences) ===")
hits5, total5 = repetition_rate(5)
rates_5 = {}
for noun in sorted(TARGET_NOUNS, key=lambda n: -freq[n]):
    if total5[noun] > 0:
        rate = hits5[noun] / total5[noun]
        rates_5[noun] = rate
        print(f"  {noun}: {rate:.1%} ({hits5[noun]}/{total5[noun]})")

# ── 3. Count of 3+ consecutive naming windows ──
print("\n=== WINDOWS WITH 3+ MENTIONS IN 10 SENTENCES ===")
burst_counts = defaultdict(int)
for i in range(len(lines) - 9):
    window_tokens = []
    for j in range(10):
        window_tokens.extend(tokenize(lines[i + j]))
    c = Counter(window_tokens)
    for noun in TARGET_NOUNS:
        if c[noun] >= 3:
            burst_counts[noun] += 1

for noun in sorted(TARGET_NOUNS, key=lambda n: -burst_counts[n]):
    if burst_counts[noun] > 0:
        print(f"  {noun}: {burst_counts[noun]} windows")

# ── 4. Two-noun contrastive contexts ──
# How often do two different target nouns appear in the same sentence?
print("\n=== TWO-NOUN CO-OCCURRENCE (same sentence) ===")
cooccur = 0
cooccur_examples = []
for line in lines:
    tokens = set(tokenize(line))
    target_in_line = [n for n in TARGET_NOUNS if n in tokens]
    if len(target_in_line) >= 2:
        cooccur += 1
        if len(cooccur_examples) < 5:
            cooccur_examples.append((target_in_line, line[:80]))

print(f"  Total sentences with 2+ target nouns: {cooccur} / {len(lines)} "
      f"({100*cooccur/len(lines):.2f}%)")
for nouns, ex in cooccur_examples:
    print(f"    {nouns}: {ex}")

# ── 5. Save JSON for upload ──
output = {
    'n_lines': len(lines),
    'frequencies': dict(freq),
    'repetition_rate_3sent': {k: round(v, 4) for k, v in rates_3.items()},
    'repetition_rate_5sent': {k: round(v, 4) for k, v in rates_5.items()},
    'burst_windows_3in10': dict(burst_counts),
    'two_noun_cooccurrence': cooccur,
    'two_noun_cooccurrence_pct': round(100 * cooccur / len(lines), 3),
}

out_path = 'corpus_noun_stats.json'
with open(out_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved {out_path} — upload this file to Claude.")
