"""Run TAFE-Net zero-shot configs from configs_zsl.json, for one or more seeds.

  python tools/run_zsl.py awa2-zsl awa2-gzsl --seeds 0 1 2
  python tools/run_zsl.py all --seeds 0 1 2          # every config in the file
  python tools/run_zsl.py awa2-zsl --dry-run         # print the commands only

Shared hyper-parameters are the paper's (Sec. 4.1): Adam, batch 32, lr 1e-4 for the
prediction network and weight generators, 1e-5 for the task embedding network, beta 0.1.
Uses the current interpreter (sys.executable), so it works the same on Windows and Linux.
Stdout of each run goes to logs/<config>-seed<k>.out; the training script also writes
zero-shot/checkpoints/<dataset>/<arch>/<config>/run_<n>/log_train.txt and arguments.json.
"""
import argparse
import json
import os
import subprocess
import sys

COMMON = {'--data-root': 'data/zero-shot', '--input-dim': '2048', '--feat-dim': '2048',
          '--lr': '1e-4', '--elr': '1e-5', '--blr': '1e-4', '--beta': '0.1', '--batch-size': '32'}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('configs', nargs='+', help="config names in configs_zsl.json, or 'all'")
    ap.add_argument('--seeds', nargs='+', type=int, default=[0])
    ap.add_argument('--file', default='configs_zsl.json')
    ap.add_argument('--num-workers', default='0', help='DataLoader workers (0 is fastest on Windows)')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    with open(args.file) as f:
        cfgs = {k: v for k, v in json.load(f).items() if not k.startswith('_')}
    names = list(cfgs) if args.configs == ['all'] else args.configs
    unknown = [n for n in names if n not in cfgs]
    if unknown:
        sys.exit(f'unknown config(s): {unknown}; available: {list(cfgs)}')

    os.makedirs('logs', exist_ok=True)
    failed = []
    for name in names:
        for seed in args.seeds:
            opts = {**COMMON, **cfgs[name], '-cfg': name, '--cmd': 'train',
                    '--seed': str(seed), '--num-workers': args.num_workers}
            cmd = [sys.executable, '-m', 'zero-shot.train_zsl']
            for k, v in opts.items():
                cmd += [k] if v == '' else [k, v]
            log = os.path.join('logs', f'{name}-seed{seed}.out')
            print(' '.join(cmd), '>', log, flush=True)
            if args.dry_run:
                continue
            with open(log, 'w') as out:
                rc = subprocess.call(cmd, stdout=out, stderr=subprocess.STDOUT)
            if rc != 0:
                failed.append((name, seed))
                print(f'  FAILED (exit {rc}); see {log}', flush=True)
    if failed:
        sys.exit(f'failed runs: {failed}')


if __name__ == '__main__':
    main()
