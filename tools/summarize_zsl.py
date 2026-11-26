"""Summarize zero-shot runs into a Markdown table (mean +- std over seeds).

  python tools/summarize_zsl.py                     # all runs under zero-shot/checkpoints
  python tools/summarize_zsl.py --out results/zsl.md

For each run it reads the per-epoch `REPL` lines written by zero-shot/train_zsl.py and reports:
  best-epoch  : the epoch with the highest upstream metric on the test set. This is how the
                upstream code selects its reported number (there is no validation split).
  final-epoch : the last epoch (no test-set selection).
Each is given under two metrics:
  upstream : per-class accuracy computed inside each test batch of 32, averaged over batches
             (utils.per_class_avg_accuracy as used by the upstream code)
  full     : per-class accuracy over the whole test set, averaged over classes (the standard
             metric of Xian et al., which the paper says it follows)
ZSL reports top-1; GZSL reports u / s / H.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
from collections import defaultdict

PAPER = {  # Table 2 of the paper. ZSL: T1; GZSL: (u, s, H)
    'zsl': {'sun': 60.9, 'cub': 56.9, 'awa1': 70.8, 'awa2': 69.3, 'apy': 42.2},
    'gzsl': {'sun': (27.9, 40.2, 33.0), 'cub': (41.0, 61.4, 49.2), 'awa1': (50.5, 84.4, 63.2),
             'awa2': (36.7, 90.6, 52.2), 'apy': (24.3, 75.4, 36.8)},
}
LINE = re.compile(r'REPL epoch=(\d+) (.*)')


def parse(log):
    rows = []
    for line in open(log, encoding='utf8', errors='replace'):
        m = LINE.search(line)
        if m:
            rows.append({k: float(v) for k, v in (kv.split('=') for kv in m.group(2).split())})
    return rows


def fmt(xs):
    if not xs:
        return '-'
    if len(xs) == 1:
        return f'{xs[0]:.1f}'
    return f'{st.mean(xs):.1f} ± {st.stdev(xs):.1f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default='zero-shot/checkpoints')
    ap.add_argument('--out', default=None, help='write the Markdown table here')
    ap.add_argument('--json', default=None, help='also write the rows as JSON here')
    ap.add_argument('--min-epochs', type=int, default=0, help='skip runs with fewer logged epochs')
    ap.add_argument('--configs-file', default='configs_zsl.json',
                    help='only report configs named in this file (ad-hoc runs are skipped)')
    ap.add_argument('--all', action='store_true', help='report every run, named or not')
    args = ap.parse_args()
    with open(args.configs_file) as f:
        known = {k for k in json.load(f) if not k.startswith('_')}

    groups = defaultdict(list)  # (cfg, dataset, gzsl) -> list of (seed, rows)
    for log in sorted(glob.glob(os.path.join(args.root, '*', '*', '*', 'run_*', 'log_train.txt'))):
        run_dir = os.path.dirname(log)
        a = dict(json.load(open(os.path.join(run_dir, 'arguments.json'))))
        if not args.all and a['config_name'] not in known:
            continue
        rows = parse(log)
        if not rows or len(rows) < max(args.min_epochs, 1) or len(rows) < a['epochs']:
            continue  # unfinished or smoke runs
        groups[(a['config_name'], a['dataset'], a['gzsl'])].append((a.get('seed'), rows))

    records = []  # one per table row
    for (cfg, ds, gzsl), runs in sorted(groups.items()):
        seeds = ','.join(str(s) for s, _ in runs)
        fin = [r[-1] for _, r in runs]
        if gzsl:
            best = [max(r, key=lambda x: x['upstream_h']) for _, r in runs]
            paper = PAPER['gzsl'].get(ds)
            for i, k in enumerate(['u', 's', 'h']):
                records.append(dict(config=cfg, dataset=ds, setting='gzsl', seeds=seeds, metric=k.upper(),
                                    best_upstream=fmt([b['upstream_' + k] for b in best]),
                                    best_full=fmt([b['full_' + k] for b in best]),
                                    final_upstream=fmt([f['upstream_' + k] for f in fin]),
                                    final_full=fmt([f['full_' + k] for f in fin]),
                                    paper=str(paper[i]) if paper else '-'))
        else:
            best = [max(r, key=lambda x: x['upstream_top1']) for _, r in runs]
            records.append(dict(config=cfg, dataset=ds, setting='zsl', seeds=seeds, metric='T1',
                                best_upstream=fmt([b['upstream_top1'] for b in best]),
                                best_full=fmt([b['full_top1'] for b in best]),
                                final_upstream=fmt([f['upstream_top1'] for f in fin]),
                                final_full=fmt([f['full_top1'] for f in fin]),
                                paper=str(PAPER['zsl'].get(ds, '-'))))

    out = ['| config | seeds | metric | best-epoch (upstream) | best-epoch (full) | '
           'final-epoch (upstream) | final-epoch (full) | paper |',
           '|---|---|---|---|---|---|---|---|']
    out += [f'| {r["config"]} | {r["seeds"]} | {r["metric"]} | {r["best_upstream"]} | {r["best_full"]} | '
            f'{r["final_upstream"]} | {r["final_full"]} | {r["paper"]} |' for r in records]
    text = '\n'.join(out) + '\n'
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        open(args.out, 'w', encoding='utf8').write(text)
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        json.dump(records, open(args.json, 'w', encoding='utf8'), ensure_ascii=False, indent=1)


if __name__ == '__main__':
    main()
