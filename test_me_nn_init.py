"""Test ME with nearest-neighbor nonce embedding init.

Instead of random init from mean/std, initialize each nonce token's embedding
as the average of the K nearest low-frequency noun embeddings. This places
nonces in a "noun-like" region of embedding space with less variance across seeds.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math

t_base = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
m_base = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')

# Find low-frequency nouns to use as anchors
# These are real words that are single-token and infrequent — similar to what
# a nonce word "should" look like in embedding space
anchor_words = [
    ' asp', ' helm', ' rib', ' cot', ' gob', ' cork',  # our pseudo-nonces
    ' hen', ' rib', ' pan', ' pod', ' rod', ' pip',     # other rare nouns
]

# Get their token IDs and embeddings
anchor_ids = []
for w in anchor_words:
    ids = t_base.encode(w, add_special_tokens=False)
    if len(ids) == 1:
        anchor_ids.append(ids[0])

with torch.no_grad():
    emb = m_base.get_input_embeddings().weight
    anchor_embs = emb[anchor_ids]
    anchor_mean = anchor_embs.mean(dim=0)
    anchor_std = anchor_embs.std(dim=0)

print(f"Found {len(anchor_ids)} single-token anchor nouns")
print(f"Anchor embedding mean norm: {anchor_mean.norm():.3f}")
print(f"Random embedding mean norm: {emb.mean(dim=0).norm():.3f}")
print(f"Anchor std mean: {anchor_std.mean():.4f}")
print(f"Global std mean: {emb.std(dim=0).mean():.4f}")

nonces = [' wug', ' dax', ' blicket', ' toma', ' zib', ' mep', ' fep', ' gorp']
nonce_names = ['wug', 'dax', 'blicket', 'toma', 'zib', 'mep', 'fep', 'gorp']

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

# Strategy A: Mean of anchors + small noise (10 seeds)
print("\n=== STRATEGY A: Anchor mean + small noise ===")
seeds = list(range(10))
all_results_a = []

for seed in seeds:
    torch.manual_seed(seed)
    t = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
    t.add_tokens(nonces)
    m = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')
    old_size = m.get_input_embeddings().weight.shape[0]
    m.resize_token_embeddings(len(t), mean_resizing=False)

    with torch.no_grad():
        e = m.get_input_embeddings().weight
        for i in range(old_size, len(t)):
            # Anchor mean + small Gaussian noise (10% of anchor std)
            e[i] = anchor_mean + 0.1 * torch.normal(torch.zeros_like(anchor_std), anchor_std)
        if hasattr(m, 'lm_head') and m.lm_head.weight.shape[0] == len(t):
            m.lm_head.weight[old_size:] = e[old_size:].clone()

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

    me_count = 0
    me_swap = 0
    diffs = []
    diffs_s = []
    for fam, nr, nq in tests:
        p = 'there is a ' + fam + ' and a ' + nr + ' . the ' + nq + ' is the'
        d = logp(p, nr) - logp(p, fam)
        diffs.append(d)
        if d > 0: me_count += 1

        ps = 'there is a ' + nr + ' and a ' + fam + ' . the ' + nq + ' is the'
        ds = logp(ps, nr) - logp(ps, fam)
        diffs_s.append(ds)
        if ds > 0: me_swap += 1

    all_results_a.append((seed, me_count, me_swap, sum(diffs)/len(diffs), diffs, diffs_s))

print('{:>6s}  {:>6s}  {:>6s}  {:>10s}'.format('seed', 'ME/8', 'swap/8', 'mean_diff'))
print('-' * 35)
for seed, mc, ms, md, _, _ in all_results_a:
    print('{:>6d}  {:>6d}  {:>6d}  {:>10.3f}'.format(seed, mc, ms, md))
t_me = sum(r[1] for r in all_results_a)
t_sw = sum(r[2] for r in all_results_a)
print('{:>6s}  {:>6.1f}  {:>6.1f}'.format('mean', t_me/10, t_sw/10))

print('\n--- Per-item stability (Strategy A) ---')
print('{:>14s}  {:>8s}  {:>8s}'.format('pair', 'ME/10', 'swap/10'))
print('-' * 34)
for idx, (fam, nr, nq) in enumerate(tests):
    im = sum(1 for r in all_results_a if r[4][idx] > 0)
    ims = sum(1 for r in all_results_a if r[5][idx] > 0)
    print('{:>14s}  {:>8d}  {:>8d}'.format(fam+'/'+nr, im, ims))


# Strategy B: Each nonce gets a DIFFERENT random anchor noun's embedding + noise
print("\n=== STRATEGY B: Each nonce = random anchor + noise ===")
all_results_b = []

for seed in seeds:
    torch.manual_seed(seed)
    t = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
    t.add_tokens(nonces)
    m = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')
    old_size = m.get_input_embeddings().weight.shape[0]
    m.resize_token_embeddings(len(t), mean_resizing=False)

    with torch.no_grad():
        e = m.get_input_embeddings().weight
        perm = torch.randperm(len(anchor_ids))
        for i in range(old_size, len(t)):
            # Pick a random anchor and add small noise
            a_idx = anchor_ids[perm[i - old_size] % len(anchor_ids)]
            e[i] = emb[a_idx] + 0.1 * torch.randn_like(emb[a_idx])
        if hasattr(m, 'lm_head') and m.lm_head.weight.shape[0] == len(t):
            m.lm_head.weight[old_size:] = e[old_size:].clone()

    m.eval()

    me_count = 0
    me_swap = 0
    diffs = []
    diffs_s = []
    for fam, nr, nq in tests:
        p = 'there is a ' + fam + ' and a ' + nr + ' . the ' + nq + ' is the'
        d = logp(p, nr) - logp(p, fam)
        diffs.append(d)
        if d > 0: me_count += 1

        ps = 'there is a ' + nr + ' and a ' + fam + ' . the ' + nq + ' is the'
        ds = logp(ps, nr) - logp(ps, fam)
        diffs_s.append(ds)
        if ds > 0: me_swap += 1

    all_results_b.append((seed, me_count, me_swap, sum(diffs)/len(diffs), diffs, diffs_s))

print('{:>6s}  {:>6s}  {:>6s}  {:>10s}'.format('seed', 'ME/8', 'swap/8', 'mean_diff'))
print('-' * 35)
for seed, mc, ms, md, _, _ in all_results_b:
    print('{:>6d}  {:>6d}  {:>6d}  {:>10.3f}'.format(seed, mc, ms, md))
t_me = sum(r[1] for r in all_results_b)
t_sw = sum(r[2] for r in all_results_b)
print('{:>6s}  {:>6.1f}  {:>6.1f}'.format('mean', t_me/10, t_sw/10))

print('\n--- Per-item stability (Strategy B) ---')
print('{:>14s}  {:>8s}  {:>8s}'.format('pair', 'ME/10', 'swap/10'))
print('-' * 34)
for idx, (fam, nr, nq) in enumerate(tests):
    im = sum(1 for r in all_results_b if r[4][idx] > 0)
    ims = sum(1 for r in all_results_b if r[5][idx] > 0)
    print('{:>14s}  {:>8d}  {:>8d}'.format(fam+'/'+nr, im, ims))
