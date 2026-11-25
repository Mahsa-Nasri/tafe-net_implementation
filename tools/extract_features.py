"""Extract pooled CNN features for TAFE-Net compositional-zs (replaces the access-locked upstream features).

Reads <root>/metadata.t7 (from tools/build_compose_data.py) and the images (--images, default <root>/images), and writes
<root>/<arch>_features.t7 = {'files': [image keys], 'features': FloatTensor[N, D]}, the format
compositional-zs/compose_data.py loads. Every requested backbone is run on the same decoded batch,
so each image is decoded once.

Choices (ours, not upstream's; record them with results):
  - Backbones: ImageNet-pretrained torchvision ResNet-18/50/101 (IMAGENET1K_V1 weights, the 2019-era
    weights) and timm DLA-34/102. Output = global-average-pooled penultimate features
    (512 / 2048 / 2048 / 512 / 1024), matching the --input-dim values in configs.json.
  - MIT-States: Resize(256) + CenterCrop(224), as in attributes-as-operators' feature script.
  - StanfordVRD: crop the subject-object union box, then Resize((224, 224)).
  - fp32, eval mode, ImageNet mean/std.
Images whose decode fails are listed and the run exits non-zero without writing features.
"""
import argparse
import os
import sys
import time
from os.path import join

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

DIMS = {'resnet18': 512, 'resnet50': 2048, 'resnet101': 2048, 'dla34': 512, 'dla102': 1024}
NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class Items(Dataset):
    def __init__(self, img_root, meta, vrd):
        self.root, self.vrd = img_root, vrd
        seen, self.items = set(), []
        for m in meta:
            if m['image'] not in seen:
                seen.add(m['image'])
                self.items.append(m)
        if vrd:
            self.tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), NORM])
        else:
            self.tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                                          transforms.ToTensor(), NORM])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        m = self.items[i]
        try:
            img = Image.open(join(self.root, m['file'] if self.vrd else m['image'])).convert('RGB')
            if self.vrd:
                x0, y0, x1, y1 = m['box']
                img = img.crop((x0, y0, x1, y1))
            return self.tf(img), m['image'], ''
        except Exception as e:  # report, do not silently skip
            return torch.zeros(3, 224, 224), m['image'], f'{type(e).__name__}: {e}'


# Pretrained weights are read from local files (downloaded and hash-checked off-cluster, uploaded via the transfer node;
# see MANIFEST.md), so jobs never download. These are exactly torchvision's IMAGENET1K_V1 files and timm's
# hf-hub `timm/<arch>.in1k` safetensors.
WEIGHTS = {'resnet18': 'resnet18-f37072fd.pth', 'resnet50': 'resnet50-0676ba61.pth',
           'resnet101': 'resnet101-63fe2227.pth', 'dla34': 'dla34.in1k.safetensors',
           'dla102': 'dla102.in1k.safetensors'}


def backbone(arch, weights_dir):
    path = join(weights_dir, WEIGHTS[arch])
    if arch.startswith('resnet'):
        net = getattr(models, arch)(weights=None)
        net.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
        net.fc = nn.Identity()
    else:
        import timm
        net = timm.create_model(arch, pretrained=True, num_classes=0, pretrained_cfg_overlay=dict(file=path))
    print(f'  {arch}: loaded {path}', flush=True)
    return net.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('root', help='data/compositional-zs/<dataset>')
    ap.add_argument('--archs', nargs='+', default=list(DIMS), choices=list(DIMS))
    ap.add_argument('--images', default=None,
                    help='image root (default <root>/images); MIT: <attr_obj>/x.jpg, VRD: {train,test}_images/x.jpg')
    ap.add_argument('--weights-dir', default='data/weights', help='folder with the pretrained weight files')
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0, help='only the first N items (smoke test; output gets .partial)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    vrd = 'stanfordvrd' in os.path.basename(os.path.normpath(args.root))
    suffix = '.partial' if args.limit else ''
    outs = {a: join(args.root, f'{a}_features.t7{suffix}') for a in args.archs}
    for a, p in outs.items():
        if os.path.exists(p) and not args.force:
            sys.exit(f'ERROR: {p} exists; refusing to overwrite (use --force)')

    meta = torch.load(join(args.root, 'metadata.t7'), weights_only=True)
    ds = Items(args.images or join(args.root, 'images'), meta, vrd)
    if args.limit:
        ds.items = ds.items[:args.limit]
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'{len(ds)} items, vrd={vrd}, device={dev}, archs={args.archs}', flush=True)

    nets = {a: backbone(a, args.weights_dir).to(dev) for a in args.archs}
    feats = {a: [] for a in args.archs}
    keys, errors, t0 = [], [], time.time()
    with torch.no_grad():
        for i, (x, k, err) in enumerate(dl):
            x = x.to(dev, non_blocking=True)
            for a, net in nets.items():
                feats[a].append(net(x).float().cpu())
            keys += list(k)
            errors += [(kk, e) for kk, e in zip(k, err) if e]
            if i % 50 == 0:
                print(f'  batch {i}/{len(dl)}  {time.time() - t0:.0f}s', flush=True)

    if errors:
        print(f'ERROR: {len(errors)} images failed to decode; no features written. First 20:')
        for k, e in errors[:20]:
            print(f'  {k}: {e}')
        sys.exit(1)
    for a in args.archs:
        f = torch.cat(feats[a])
        assert f.shape == (len(keys), DIMS[a]), (a, f.shape)
        assert torch.isfinite(f).all(), f'{a}: non-finite features'
        tmp = outs[a] + '.tmp'
        torch.save({'files': keys, 'features': f}, tmp)
        os.replace(tmp, outs[a])
        print(f'wrote {outs[a]}  {tuple(f.shape)}  mean={f.mean():.4f}')
    print(f'done in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
