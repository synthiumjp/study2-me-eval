"""Test ME across 10 random seeds for nonce embedding initialization."""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, math, copy

base_tokenizer = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
base_model = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')

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

seeds = list(range(10))
all_results = []  # list of (seed, me_count, me_count_swap, item_diffs)

for seed in seeds:
    torch.manual_seed(seed)

    t = AutoTokenizer.from_pretrained('checkpoints/ar_pilot')
    t.add_tokens(nonces)

    m = AutoModelForCausalLM.from_pretrained('checkpoints/ar_pilot')
    old_size = m.get_input_embeddings().weight.shape[0]
    m.resize_token_embeddings(len(t), mean_resizing=False)

    # Random init from existing distribution
    with torch.no_grad():
        emb = m.get_input_embeddings().weight
        existing_mean = emb[:old_size].mean(dim=0)
        existing_std = emb[:old_size].std(dim=0)
        for i in range(old_size, len(t)):
            emb[i] = torch.normal(existing_mean, existing_std)
        # Also set lm_head for new tokens
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

    me_count = 0
    me_count_swap = 0
    diffs = []
    diffs_swap = []

    for fam, nonce_ref, nonce_q in tests:
        # Standard order
        prompt = 'there is a ' + fam + ' and a ' + nonce_ref + ' . the ' + nonce_q + ' is the'
        lp_fam = logp(prompt, fam)
        lp_nonce = logp(prompt, nonce_ref)
        diff = lp_nonce - lp_fam
        if diff > 0: me_count += 1
        diffs.append(diff)

        # Swapped order
        prompt_s = 'there is a ' + nonce_ref + ' and a ' + fam + ' . the ' + nonce_q + ' is the'
        lp_fam_s = logp(prompt_s, fam)
        lp_nonce_s = logp(prompt_s, nonce_ref)
        diff_s = lp_nonce_s - lp_fam_s
        if diff_s > 0: me_count_swap += 1
        diffs_swap.append(diff_s)

    mean_diff = sum(diffs) / len(diffs)
    mean_diff_swap = sum(diffs_swap) / len(diffs_swap)
    all_results.append((seed, me_count, me_count_swap, mean_diff, mean_diff_swap, diffs, diffs_swap))

# Print results
print('{:>6s}  {:>6s}  {:>6s}  {:>10s}  {:>10s}'.format(
    'seed', 'ME/8', 'swap/8', 'mean_diff', 'swap_diff'))
print('-' * 45)

total_me = 0
total_swap = 0
for seed, mc, mcs, md, mds, _, _ in all_results:
    total_me += mc
    total_swap += mcs
    print('{:>6d}  {:>6d}  {:>6d}  {:>10.3f}  {:>10.3f}'.format(seed, mc, mcs, md, mds))

print('-' * 45)
print('{:>6s}  {:>6.1f}  {:>6.1f}  {:>10s}  {:>10s}'.format(
    'mean', total_me/10, total_swap/10, '', ''))

# Per-item analysis: which items are ME-consistent across seeds?
print('\n--- Per-item stability ---')
print('{:>14s}  {:>12s}  {:>12s}'.format('pair', 'ME_count/10', 'swap_count/10'))
print('-' * 42)

for idx, (fam, nonce_ref, nonce_q) in enumerate(tests):
    item_me = sum(1 for _, _, _, _, _, d, _ in all_results if d[idx] > 0)
    item_swap = sum(1 for _, _, _, _, _, _, ds in all_results if ds[idx] > 0)
    label = fam + '/' + nonce_ref
    print('{:>14s}  {:>12d}  {:>12d}'.format(label, item_me, item_swap))
