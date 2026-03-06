"""ME nonce test with correctly added space-prefixed nonce tokens."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math

t = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
m = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')

# Add nonce tokens WITH space prefix
nonces = [' wug', ' dax', ' blicket', ' toma', ' zib', ' mep', ' fep', ' gorp']
num_added = t.add_tokens(nonces)
old_size = m.get_input_embeddings().weight.shape[0]
m.resize_token_embeddings(len(t))

# Init new embeddings from existing distribution
with torch.no_grad():
    emb = m.get_input_embeddings().weight
    existing_mean = emb[:old_size].mean(dim=0)
    existing_std = emb[:old_size].std(dim=0)
    for i in range(old_size, len(t)):
        emb[i] = torch.normal(existing_mean, existing_std)

# Verify single-token
print("Nonce token check:")
for n in nonces:
    ids = t.encode("the" + n, add_special_tokens=False)
    print(f"  'the{n}' -> {ids} -> {[t.decode(i) for i in ids]} (nonce is single token: {len(ids)==2})")

print("\nReference:")
for w in ['ball', 'dog', 'cup', 'cat']:
    ids = t.encode("the " + w, add_special_tokens=False)
    print(f"  'the {w}' -> {ids} -> {[t.decode(i) for i in ids]}")

m.eval()

def logp(prompt, word):
    """Log prob of word as next token(s) after prompt."""
    ids = t.encode(prompt, add_special_tokens=False)
    # word should encode with space prefix to match BPE convention
    wids = t.encode(" " + word.strip(), add_special_tokens=False)
    full = torch.tensor([ids + wids])
    with torch.no_grad():
        out = m(full)
        lp = torch.log_softmax(out.logits, dim=-1)
    total = 0.0
    for i, wid in enumerate(wids):
        pos = len(ids) - 1 + i
        total += lp[0, pos, wid].item()
    return total

# Strip the space prefix for display but keep it for encoding
nonce_names = ['wug', 'dax', 'blicket', 'toma', 'zib', 'mep', 'fep', 'gorp']

# ME test: 'there is a [FAM] and a [NONCE1] . the [NONCE2] is the ___'
tests = [
    ('ball', 'wug', 'dax'),
    ('dog', 'blicket', 'toma'),
    ('cup', 'zib', 'mep'),
    ('cat', 'fep', 'gorp'),
    ('book', 'toma', 'wug'),
    ('car', 'dax', 'blicket'),
    ('hat', 'gorp', 'fep'),
    ('bird', 'mep', 'zib'),
]

print("\n--- ME TEST (familiar first) ---")
print('{:>14s}  {:>10s}  {:>10s}  {:>10s}  {:>4s}'.format('pair', 'logP_fam', 'logP_nonce', 'diff', 'ME?'))
print('-' * 55)

me_count = 0
for fam, nonce_ref, nonce_q in tests:
    prompt = 'there is a ' + fam + ' and a ' + nonce_ref + ' . the ' + nonce_q + ' is the'
    lp_fam = logp(prompt, fam)
    lp_nonce = logp(prompt, nonce_ref)
    diff = lp_nonce - lp_fam
    me = diff > 0
    if me: me_count += 1
    label = fam + '/' + nonce_ref
    print('{:>14s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>4s}'.format(
        label, lp_fam, lp_nonce, diff, 'YES' if me else 'no'))

print(f"\nME-consistent: {me_count}/8")

print("\n--- RECENCY SWAP (nonce first) ---")
print('{:>14s}  {:>10s}  {:>10s}  {:>10s}  {:>4s}'.format('pair', 'logP_fam', 'logP_nonce', 'diff', 'ME?'))
print('-' * 55)

me_count_swap = 0
for fam, nonce_ref, nonce_q in tests:
    prompt = 'there is a ' + nonce_ref + ' and a ' + fam + ' . the ' + nonce_q + ' is the'
    lp_fam = logp(prompt, fam)
    lp_nonce = logp(prompt, nonce_ref)
    diff = lp_nonce - lp_fam
    me = diff > 0
    if me: me_count_swap += 1
    label = fam + '/' + nonce_ref
    print('{:>14s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>4s}'.format(
        label, lp_fam, lp_nonce, diff, 'YES' if me else 'no'))

print(f"\nME-consistent (swap): {me_count_swap}/8")

# Nonce-nonce baseline
print("\n--- NONCE-NONCE BASELINE ---")
print('{:>14s}  {:>10s}  {:>10s}  {:>10s}'.format('pair', 'logP_n1', 'logP_n2', 'abs_diff'))
print('-' * 50)

nn_tests = [
    ('wug', 'blicket', 'toma'),
    ('dax', 'zib', 'mep'),
    ('fep', 'gorp', 'dax'),
]
for n1, n2, nq in nn_tests:
    prompt = 'there is a ' + n1 + ' and a ' + n2 + ' . the ' + nq + ' is the'
    lp1 = logp(prompt, n1)
    lp2 = logp(prompt, n2)
    label = n1 + '/' + n2
    print('{:>14s}  {:>10.3f}  {:>10.3f}  {:>10.3f}'.format(label, lp1, lp2, abs(lp1 - lp2)))
