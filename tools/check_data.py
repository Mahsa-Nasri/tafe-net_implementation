"""Data preflight for TAFE-Net jobs. Exits non-zero on any problem.

  python -I tools/check_data.py compose data/compositional-zs/mit-states [--feat resnet101]
  python -I tools/check_data.py zsl data/zero-shot/awa2

compose: split sizes, train/test disjointness, metadata coverage, GloVe coverage of every word the
         loader will look up, and (with --feat) that the feature file covers every metadata key with
         the right dimension and finite values.
zsl:     xlsa17 .mat keys and split sizes. The expected sizes are the "proposed split" numbers from
         Xian et al. (TPAMI 2018, Table 2).
"""
import argparse
import sys
from os.path import join, basename, normpath

COMPOSE = {
    'mit-states': dict(train_pairs=1262, test_pairs=700, train=34562, test=19191),
    'stanfordvrd': dict(train_pairs=6672, test_pairs=1029, train=30355, test=1169),
}
DIMS = {'resnet18': 512, 'resnet50': 2048, 'resnet101': 2048, 'dla34': 512, 'dla102': 1024}
# dataset: (n_images, trainval, test_seen, test_unseen, n_attributes, n_classes)
ZSL = {
    'awa1': (30475, 19832, 4958, 5685, 85, 50),
    'awa2': (37322, 23527, 5882, 7913, 85, 50),
    'cub': (11788, 7057, 1764, 2967, 312, 200),
    'sun': (14340, 10320, 2580, 1440, 102, 717),
    'apy': (15339, 5932, 1483, 7924, 64, 32),
}
problems = []


def expect(name, got, want):
    ok = got == want
    print(f'  [{"OK " if ok else "BAD"}] {name}: {got}' + ('' if ok else f' (expected {want})'))
    if not ok:
        problems.append(name)


def compose(root, feat):
    import torch
    ds = basename(normpath(root))
    triple = ds == 'stanfordvrd'
    rd = lambda f: [tuple(l.split()) for l in open(join(root, 'compositional-split', f)).read().strip().split('\n')]
    tr, te = rd('train_pairs.txt'), rd('test_pairs.txt')
    exp = COMPOSE[ds]
    expect('train pairs', len(tr), exp['train_pairs'])
    expect('test pairs', len(te), exp['test_pairs'])
    expect('train/test overlap', len(set(tr) & set(te)), 0)
    expect('malformed pair lines', sum(len(p) != (3 if triple else 2) for p in tr + te), 0)

    meta = torch.load(join(root, 'metadata.t7'), weights_only=True)
    key = (lambda m: (m['sub'], m['pred'], m['obj'])) if triple else (lambda m: (m['attr'], m['obj']))
    trs, tes = set(tr), set(te)
    expect('train instances', sum(key(m) in trs for m in meta), exp['train'])
    expect('test instances', sum(key(m) in tes for m in meta), exp['test'])

    words = {w for p in tr + te for name in p for w in name.lower().split('-')}
    found = set()
    with open(join(root, 'glove', 'glove.6B.300d.txt'), encoding='utf8') as f:
        for line in f:
            w = line.split(' ', 1)[0]
            if w in words:
                found.add(w)
    miss = sorted(words - found)
    expect('words missing from GloVe', len(miss), 0)
    if miss:
        print('    missing:', miss[:30])

    if feat:
        d = torch.load(join(root, f'{feat}_features.t7'), weights_only=True)
        f, files = d['features'], set(d['files'])
        expect(f'{feat} feature dim', f.shape[1], DIMS[feat])
        expect(f'{feat} rows == files', f.shape[0], len(d['files']))
        expect(f'{feat} metadata keys without a feature', len({m['image'] for m in meta} - files), 0)
        expect(f'{feat} non-finite values', int((~torch.isfinite(f)).sum()), 0)
        expect(f'{feat} all-zero rows', int((f.abs().sum(1) == 0).sum()), 0)


def zsl(root):
    from scipy.io import loadmat
    ds = basename(normpath(root)).lower()
    n, trv, ts, tu, na, nc = ZSL[ds]
    fl = loadmat(join(root, 'res101.mat'))
    sp = loadmat(join(root, 'att_splits.mat'))
    expect('res101 features shape', tuple(fl['features'].shape), (2048, n))
    expect('label count', fl['labels'].size, n)
    expect('n classes', len(set(fl['labels'].ravel().tolist())), nc)
    for k, want in [('trainval_loc', trv), ('test_seen_loc', ts), ('test_unseen_loc', tu)]:
        expect(k, sp[k].size, want)
    att = sp['original_att'] if 'original_att' in sp else sp['att']
    print(f'  attribute key used by the loader: {"original_att" if "original_att" in sp else "att"}')
    expect('attribute matrix shape', tuple(att.shape), (na, nc))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('kind', choices=['compose', 'zsl'])
    ap.add_argument('root')
    ap.add_argument('--feat', default=None, choices=list(DIMS))
    args = ap.parse_args()
    print(f'check_data {args.kind} {args.root}' + (f' --feat {args.feat}' if args.feat else ''))
    compose(args.root, args.feat) if args.kind == 'compose' else zsl(args.root)
    if problems:
        print(f'FAILED: {problems}')
        sys.exit(1)
    print('all checks passed')


if __name__ == '__main__':
    main()
