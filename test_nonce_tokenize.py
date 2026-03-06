"""Test nonce token addition strategies for the AR tokenizer."""
from transformers import AutoTokenizer
from tokenizers import AddedToken

# Strategy 1: Add with space prefix as plain strings
t1 = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
nonces_space = [' wug', ' dax', ' blicket', ' toma', ' zib', ' mep', ' fep', ' gorp']
num = t1.add_tokens(nonces_space)
print(f"Strategy 1 — add with space prefix (added {num}):")
for n in nonces_space:
    ids = t1.encode("the" + n, add_special_tokens=False)
    print(f"  'the{n}' -> {ids} -> {[t1.decode(i) for i in ids]}")

print()

# Strategy 2: Add plain, encode with space
t2 = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
nonces_plain = ['wug', 'dax', 'blicket', 'toma', 'zib', 'mep', 'fep', 'gorp']
num = t2.add_tokens(nonces_plain)
print(f"Strategy 2 — add plain strings (added {num}):")
for n in nonces_plain:
    ids = t2.encode("the " + n, add_special_tokens=False)
    print(f"  'the {n}' -> {ids} -> {[t2.decode(i) for i in ids]}")

print()

# Strategy 3: Check how existing words tokenize for reference
t3 = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
print("Reference — how familiar words tokenize:")
for w in ['ball', 'dog', 'cup', 'cat']:
    ids = t3.encode("the " + w, add_special_tokens=False)
    print(f"  'the {w}' -> {ids} -> {[t3.decode(i) for i in ids]}")

print()

# Strategy 4: Use Ġ prefix (ByteLevel BPE convention)
t4 = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
nonces_g = ['\u0120wug', '\u0120dax', '\u0120blicket', '\u0120toma', '\u0120zib', '\u0120mep', '\u0120fep', '\u0120gorp']
num = t4.add_tokens(nonces_g)
print(f"Strategy 4 — add with Ġ prefix (added {num}):")
for n_raw, n_g in zip(nonces_plain, nonces_g):
    ids = t4.encode("the " + n_raw, add_special_tokens=False)
    print(f"  'the {n_raw}' -> {ids} -> {[t4.decode(i) for i in ids]}")
