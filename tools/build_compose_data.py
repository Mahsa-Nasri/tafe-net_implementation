"""Build TAFE-Net compositional-zs data folders from public sources.

The upstream feature bundle is access-locked, so this rebuilds what
`compositional-zs/compose_data.py` expects under <out>/<dataset>/:

    compositional-split/train_pairs.txt   one composition per line
    compositional-split/test_pairs.txt
    metadata.t7                           list of dicts (see below)
    glove/glove.6B.300d.txt               symlink to the GloVe file
    images                                symlink to the image root (used by tools/extract_features.py)

MIT-States (original attr-ops split, rebuilt from the natural split of Purushwalkam et al. 2019):
    train pairs = natural train_pairs (1262 seen pairs)
    test pairs  = unseen pairs of natural val + test (300 + 400 = 700)
    metadata    = {'image', 'attr', 'obj'} for every image of those pairs, as in attr-ops
                  (the loader assigns images to train/test purely by pair).

StanfordVRD (SPO triplets, as in Red Wine / TAFE-Net):
    train pairs = unique (sub, pred, obj) in the VRD train annotations (6672)
    test pairs  = unique test triplets never seen in training (1029)
    metadata    = one entry per relationship instance:
                  {'image': '<file>#<idx>', 'file', 'sub', 'pred', 'obj', 'box': [x0, y0, x1, y1]}
                  'box' is the union of the subject and object boxes (our choice; upstream does not say).
                  Kept: every train-image instance, and test-image instances whose triplet is unseen.
                  Test-image instances of seen triplets are dropped, so no test image enters training.
    Multi-word names ('next to', 'traffic light') are hyphenated, because the pair files are whitespace
    separated and the GloVe loader averages hyphenated words.

Run with `python -I` from the repo root. Inputs are untrusted downloads; .t7 inputs are loaded with
weights_only=True. `--json` writes metadata.json instead of metadata.t7 (no torch needed; for checks).
"""
import argparse
import json
import os
import sys
from collections import Counter
from os.path import join, exists, isfile, abspath

EXPECTED = {
    'mit-states': dict(train_pairs=1262, test_pairs=700, train_images=34562, test_images=19191),
    'stanfordvrd': dict(train_pairs=6672, test_pairs=1029, train_instances=30355, test_instances=1169),
}


def read_pairs(path):
    with open(path) as f:
        return [tuple(line.split()) for line in f.read().strip().split('\n')]


def write_pairs(path, pairs):
    with open(path, 'w') as f:
        f.write('\n'.join(' '.join(p) for p in pairs) + '\n')


def load_t7(path):
    import torch
    return torch.load(path, weights_only=True)


def save_meta(out_dir, meta, as_json):
    if as_json:
        with open(join(out_dir, 'metadata.json'), 'w') as f:
            json.dump(meta, f)
    else:
        import torch
        tmp = join(out_dir, 'metadata.t7.tmp')
        torch.save(meta, tmp)
        os.replace(tmp, join(out_dir, 'metadata.t7'))


def link(src, dst):
    if os.path.islink(dst) or exists(dst):
        if os.path.islink(dst) and os.readlink(dst) == src:
            return
        sys.exit(f'ERROR: {dst} exists and is not a link to {src}')
    os.symlink(src, dst)


def check(name, got, want):
    flag = 'OK ' if got == want else 'MISMATCH'
    print(f'  [{flag}] {name}: {got} (expected {want})')
    return got == want


def build_mit(args):
    nat = args.natural
    meta_src = join(nat, 'metadata_compositional-split-natural.t7')
    meta_in = json.load(open(args.natural_meta_json)) if args.natural_meta_json else load_t7(meta_src)
    split = join(nat, 'compositional-split-natural')
    train = read_pairs(join(split, 'train_pairs.txt'))
    seen = set(train)
    test, have = [], set()
    for p in read_pairs(join(split, 'val_pairs.txt')) + read_pairs(join(split, 'test_pairs.txt')):
        if p not in seen and p not in have:
            test.append(p)
            have.add(p)

    meta, missing = [], 0
    for m in meta_in:
        key = (m['attr'], m['obj'])
        if key in seen or key in have:
            if args.images and not isfile(join(args.images, m['image'])):
                missing += 1
                continue
            meta.append({'image': m['image'], 'attr': m['attr'], 'obj': m['obj']})
    n_tr = sum((m['attr'], m['obj']) in seen for m in meta)
    n_te = len(meta) - n_tr
    return train, test, meta, dict(train_pairs=len(train), test_pairs=len(test),
                                   train_images=n_tr, test_images=n_te), missing


def build_vrd(args):
    root = args.vrd_json
    objs = [o.replace(' ', '-') for o in json.load(open(join(root, 'objects.json')))]
    preds = [p.replace(' ', '-') for p in json.load(open(join(root, 'predicates.json')))]
    ann = {s: json.load(open(join(root, f'annotations_{s}.json'))) for s in ('train', 'test')}

    def spo(r):
        return objs[r['subject']['category']], preds[r['predicate']], objs[r['object']['category']]

    train, seen = [], set()
    for rels in ann['train'].values():
        for r in rels:
            t = spo(r)
            if t not in seen:
                seen.add(t)
                train.append(t)
    test, have = [], set()
    for rels in ann['test'].values():
        for r in rels:
            t = spo(r)
            if t not in seen and t not in have:
                have.add(t)
                test.append(t)

    meta, missing = [], 0
    counts = Counter()
    for split in ('train', 'test'):
        img_dir = join(args.images, f'{split}_images') if args.images else None
        for fname, rels in ann[split].items():
            if img_dir and not isfile(join(img_dir, fname)):
                missing += 1
                continue
            for i, r in enumerate(rels):
                t = spo(r)
                if split == 'test' and t not in have:
                    continue
                ys, xs = [], []
                for k in ('subject', 'object'):
                    y0, y1, x0, x1 = r[k]['bbox']          # VRD order: [ymin, ymax, xmin, xmax]
                    ys += [y0, y1]
                    xs += [x0, x1]
                meta.append({'image': f'{fname}#{i}', 'file': f'{split}_images/{fname}',
                             'sub': t[0], 'pred': t[1], 'obj': t[2],
                             'box': [min(xs), min(ys), max(xs), max(ys)]})
                counts[split] += 1
    return train, test, meta, dict(train_pairs=len(train), test_pairs=len(test),
                                   train_instances=counts['train'], test_instances=counts['test']), missing


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('dataset', choices=['mit-states', 'stanfordvrd'])
    ap.add_argument('--out', default='data/compositional-zs', help='output root')
    ap.add_argument('--glove', required=True, help='path to glove.6B.300d.txt')
    ap.add_argument('--images', default=None, help='image root (MIT: <root>/<attr_obj>/x.jpg; VRD: <root>/{train,test}_images/)')
    ap.add_argument('--natural', help='MIT: extracted natural-split folder (contains metadata_compositional-split-natural.t7)')
    ap.add_argument('--natural-meta-json', help='MIT: JSON copy of the natural metadata (local checks without torch)')
    ap.add_argument('--vrd-json', help='VRD: folder with annotations_{train,test}.json, objects.json, predicates.json')
    ap.add_argument('--json', action='store_true', help='write metadata.json instead of metadata.t7')
    ap.add_argument('--force', action='store_true', help='allow writing into an existing output folder')
    args = ap.parse_args()

    out_dir = join(args.out, args.dataset)
    if exists(join(out_dir, 'metadata.t7')) and not args.force:
        sys.exit(f'ERROR: {out_dir}/metadata.t7 exists; refusing to overwrite (use --force)')
    os.makedirs(join(out_dir, 'compositional-split'), exist_ok=True)
    os.makedirs(join(out_dir, 'glove'), exist_ok=True)

    train, test, meta, stats, missing = (build_mit if args.dataset == 'mit-states' else build_vrd)(args)
    assert not set(train) & set(test), 'train and test compositions overlap'

    write_pairs(join(out_dir, 'compositional-split', 'train_pairs.txt'), train)
    write_pairs(join(out_dir, 'compositional-split', 'test_pairs.txt'), test)
    save_meta(out_dir, meta, args.json)
    link(abspath(args.glove), join(out_dir, 'glove', 'glove.6B.300d.txt'))
    if args.images:
        link(abspath(args.images), join(out_dir, 'images'))

    print(f'{args.dataset}: wrote {out_dir} ({len(meta)} metadata entries)')
    if args.images:
        print(f'  image files missing on disk (skipped): {missing}')
    ok = all([check(k, stats[k], v) for k, v in EXPECTED[args.dataset].items()])
    sys.exit(0 if ok and missing == 0 else 1)


if __name__ == '__main__':
    main()
