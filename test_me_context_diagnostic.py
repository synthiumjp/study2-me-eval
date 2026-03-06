"""Context-dependence diagnostic for AR nonce ME.

If ME is genuine, the two-referent preamble is necessary.
If it's just token-frequency matching, the preamble is irrelevant.

Tests:
1. full_context:  "there is a [fam] and a [nonce1] . the [nonce2] is the ___"
2. no_preamble:   "the [nonce2] is the ___"  (no disambiguation context)
3. single_ref:    "there is a [fam] . the [nonce2] is the ___"  (only familiar introduced)
4. nonce_only:    "there is a [nonce1] . the [nonce2] is the ___"  (only nonce introduced)

If ME: full_context should show nonce > familiar. no_preamble/single_ref should NOT.
If frequency matching: all conditions should look similar.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math

t = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
m = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')

nonces = [' wug', ' dax', ' blicket', ' toma', ' zib', ' mep', ' fep', ' gorp']
t.add_tokens(nonces)
old_size = m.get_input_embeddings().weight.shape[0]
m.resize_token_embeddings(len(t), mean_resizing=False)

# Strategy B init: each nonce from a random anchor
anchor_words = [' asp', ' helm', ' rib', ' cot', ' gob', ' cork', ' hen', ' pan', ' pod', ' rod', ' pip']
anchor_ids = []
for w in anchor_words:
    ids = t.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        anchor_ids.append(ids[0])

torch.manual_seed(0)
with torch.no_grad():
    emb = m.get_input_embeddings().weight
    base_emb = emb[:old_size].clone()
    perm = torch.randperm(len(anchor_ids))
    for i in range(old_size, len(t)):
        a_idx = anchor_ids[perm[i - old_size] % len(anchor_ids)]
        emb[i] = base_emb[a_idx] + 0.1 * torch.randn_like(base_emb[a_idx])
    if hasattr(m, 'lm_head') and m.lm_head.weight.shape[0] == len(t):
        m.lm_head.weight[old_size:] = emb[old_size:].clone()

m.eval()

def logp(prompt, word):
    ids = t.encode(prompt, add_special_tokens=False)
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

conditions = {
    'full_context': lambda f, nr, nq: 'there is a ' + f + ' and a ' + nr + ' . the ' + nq + ' is the',
    'no_preamble':  lambda f, nr, nq: 'the ' + nq + ' is the',
    'fam_only':     lambda f, nr, nq: 'there is a ' + f + ' . the ' + nq + ' is the',
    'nonce_only':   lambda f, nr, nq: 'there is a ' + nr + ' . the ' + nq + ' is the',
    'swap_context': lambda f, nr, nq: 'there is a ' + nr + ' and a ' + f + ' . the ' + nq + ' is the',
}

print("CONTEXT-DEPENDENCE DIAGNOSTIC")
print("If ME is real: full_context and swap_context should show nonce > familiar.")
print("Other conditions should NOT show this pattern.\n")

for cond_name, make_prompt in conditions.items():
    me_count = 0
    diffs = []
    print(f"--- {cond_name} ---")
    print('{:>14s}  {:>10s}  {:>10s}  {:>10s}  {:>4s}'.format('pair', 'logP_fam', 'logP_nonce', 'diff', 'ME?'))
    print('-' * 55)

    for fam, nr, nq in tests:
        prompt = make_prompt(fam, nr, nq)
        lp_fam = logp(prompt, fam)
        lp_nonce = logp(prompt, nr)
        diff = lp_nonce - lp_fam
        me = diff > 0
        if me: me_count += 1
        diffs.append(diff)
        label = fam + '/' + nr
        print('{:>14s}  {:>10.3f}  {:>10.3f}  {:>10.3f}  {:>4s}'.format(
            label, lp_fam, lp_nonce, diff, 'YES' if me else 'no'))

    mean_d = sum(diffs) / len(diffs)
    print(f"  ME-consistent: {me_count}/8, mean diff: {mean_d:.3f}\n")
