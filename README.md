# TAFE-Net replication (and extension)

A replication of **TAFE-Net: Task-Aware Feature Embeddings for Low Shot Learning**
(Wang, Yu, Wang, Darrell, Gonzalez, CVPR 2019, [arXiv:1904.05967](https://arxiv.org/abs/1904.05967)),
built on the authors' code ([ucbdrive/tafe-net](https://github.com/ucbdrive/tafe-net), Apache-2.0).
The authors' original README is kept as [README_upstream.md](README_upstream.md).

What this repository adds to the upstream code:

- **Reproducible pipeline**: scripted download and checksum checks, data preflight checks, configs for
  all five zero-shot datasets (upstream only ships AWA2), multi-seed runs, and a results summarizer.
- **Two evaluation metrics, side by side**: the upstream batched metric and the standard per-class
  accuracy (see [Deviations](#differences-between-the-paper-and-the-code)).
- **Rebuilt compositional data** (MIT-States, StanfordVRD), because the upstream feature bundle is no
  longer publicly accessible.
- **Extension**: *to be added* (see [Extension](#extension)).

> Status: AWA2 ZSL run with 3 seeds; it does **not** reach the paper's number (see [2.3](#23-results)).
> Other ZSL datasets and GZSL not yet run. Compositional (Table 4) pipeline written, not yet run.

---

## 1. Setup

Tested on Windows 11 + RTX 3080 Laptop (8 GB), Python 3.12, PyTorch 2.6 (CUDA 12.4). Linux works the same way.

```bash
git clone https://github.com/Mahsa-Nasri/tafe-net_implementation.git
cd tafe-net_implementation
git checkout replication

python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install scipy tqdm
```

All commands below are run from the repository root.

## 2. Zero-shot learning (paper Table 2)

### 2.1 Data

The paper uses the "proposed splits" and ResNet-101 features of Xian et al.
([xlsa17](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly), ~922 MB).

```bash
bash tools/fetch_local.sh xlsa17          # downloads to data/_incoming/xlsa17/, checks the size
mkdir -p data/_extract && cd data/_extract && unzip -q ../_incoming/xlsa17/download && cd ../..
mkdir -p data/zero-shot
for d in AWA1 AWA2 CUB SUN APY; do mv data/_extract/xlsa17/data/$d data/zero-shot/$(echo $d | tr A-Z a-z); done
```

On Windows run these in Git Bash. Then check the split sizes against Xian et al. (Table 2):

```bash
for d in awa1 awa2 cub sun apy; do python tools/check_data.py zsl data/zero-shot/$d; done
```

Every dataset should end with `all checks passed`.

### 2.2 Train and evaluate

Each run trains on `trainval` and evaluates on the test split after every epoch.

```bash
python tools/run_zsl.py all --seeds 0 1 2        # all 12 configs x 3 seeds
python tools/run_zsl.py awa2-zsl --seeds 0       # or a single config
python tools/summarize_zsl.py --out results/zsl.md
```

The configs are in [configs_zsl.json](configs_zsl.json). Shared settings follow the paper (Sec. 4.1):
Adam, batch size 32, learning rate 1e-4 for the prediction network and weight generators and 1e-5 for the
task embedding network, divided by 10 at epochs 30 and 45, 60 epochs in total (AWA1: 10 epochs with the
drop at epoch 5), embedding-loss weight β = 0.1, and a 2-layer task network for aPY.
`awa2-*-upstream` reproduce the upstream `configs.json` exactly (it uses hidden size 512, the paper says 2048).

A 60-epoch AWA2 run takes about 30 minutes on an RTX 3080 Laptop GPU.

### 2.3 Results

AWA2, conventional ZSL, top-1 (%), mean ± std over seeds 0, 1, 2 ([results/zsl.md](results/zsl.md)):

| config | task-net hidden | best epoch, upstream metric | best epoch, standard metric | final epoch, upstream metric | final epoch, standard metric | paper |
|---|---|---|---|---|---|---|
| `awa2-zsl-upstream` (upstream `configs.json`) | 512 | 48.7 ± 4.4 | 45.3 ± 3.9 | 33.6 ± 0.7 | 33.0 ± 0.9 | 69.3 |
| `awa2-zsl` (paper text) | 2048 | 36.7 ± 3.2 | 32.5 ± 4.9 | 31.9 ± 4.3 | 25.8 ± 3.7 | 69.3 |

"Best epoch" is the epoch with the highest upstream metric on the test set (how the upstream code picks
its number). "Final epoch" is epoch 60, with no selection on the test set.

What we observe:

- **We do not reproduce the paper's AWA2 result.** Even the most generous number (best epoch on the test
  set, upstream metric) is about 20 points below 69.3.
- **The paper's hidden size (2048) is worse than the upstream config's (512)**, by 12 points at the best epoch.
- **Unseen-class accuracy peaks early (epochs 4 to 27) and then falls**, while training accuracy on the
  seen classes reaches about 97%. After the learning rate drops at epoch 30, the model fits the seen classes and
  stops transferring to unseen ones.
- **The classification loss hardly moves during training** (about 0.74 to 0.68). This fits the double
  sigmoid in the upstream loss (difference 3 below): `BCEWithLogitsLoss` applied to an output that has
  already gone through a sigmoid gives a weak gradient. This is a hypothesis we have not tested yet.
- Runs are deterministic: the same seed gives the same numbers.

Remaining datasets (AWA1, CUB, SUN, aPY) and GZSL: *not yet run*
(`python tools/run_zsl.py all --seeds 0 1 2`).

Paper numbers for reference (Table 2):

| | SUN | CUB | AWA1 | AWA2 | aPY |
|---|---|---|---|---|---|
| ZSL T1 | 60.9 | 56.9 | 70.8 | 69.3 | 42.2 |
| GZSL H | 33.0 | 49.2 | 63.2 | 52.2 | 36.8 |

## 3. Unseen attribute-object composition (paper Table 4)

The upstream features for MIT-States and StanfordVRD are on a Google Drive link that is no longer
accessible, so this repository rebuilds them from public sources:

```bash
bash tools/fetch_local.sh natural mitstates_hf vrd_json glove
python -I tools/build_compose_data.py mit-states --natural <natural dir> --images <mit-states images> --glove <glove.6B.300d.txt>
python -I tools/build_compose_data.py stanfordvrd --vrd-json <json dir> --images <vrd images> --glove <glove.6B.300d.txt>
python -I tools/extract_features.py data/compositional-zs/mit-states --archs resnet101 --weights-dir data/weights
python -I tools/check_data.py compose data/compositional-zs/mit-states --feat resnet101
python -m compositional-zs.train_compose --cmd train -a tafenet_default -cfg mit-states-resnet-101 ...   # see configs.json
```

The docstrings of [build_compose_data.py](tools/build_compose_data.py) and
[extract_features.py](tools/extract_features.py) explain every choice made.

*Not yet run.*

## 4. Extension

*To be added.*

## 5. Differences between the paper and the code

Things we found while replicating. Each one is either kept as is, for a faithful replication of the
published code, or reported as a separate result.

| # | Paper | Upstream code | What we do |
|---|---|---|---|
| 1 | Average per-class top-1 accuracy (Xian et al.) | Per-class accuracy computed **inside each test batch of 32**, then averaged over batches (`utils.per_class_avg_accuracy`) | Report both: `upstream` and `full` (whole test set) |
| 2 | Not stated | The reported number is the **best epoch on the test set** (no validation split) | Report both best-epoch and final-epoch |
| 3 | Calibrated softmax cross-entropy (Eq. 6) | The model outputs `sigmoid(logit)`, which is then fed to `BCEWithLogitsLoss` (sigmoid applied twice) | Kept for the replication |
| 4 | Task network hidden size 2048 | `configs.json` uses 512 for AWA2 | Run both (`awa2-*` and `awa2-*-upstream`) |
| 5 | 2-layer task network for aPY | Only a 3-layer model is defined | Added `tafenet_2layer` in [models.py](models.py) |
| 6 | MIT-States: 1,292 training pairs | Standard split has 1,262 | Use 1,262 |
| 7 | Compositional: word2vec (GoogleNews) | Loader uses GloVe 300d | GloVe |
| 8 | Table 4 headline row uses VGG-16 features | Configs are ResNet/DLA | ResNet-101; compare with Table 5's ResNet-101 row (AP 16.2, top-1 17.2) |

Small code fixes (no effect on training): `test()` in `zero-shot/train_zsl.py` unpacked three values
from `validate()`, which returns one; added `--seed`.

## 6. Repository layout

```
configs.json            upstream configs (compositional + AWA2)
configs_zsl.json        replication configs for all five ZSL datasets
models.py, tafenet.py   TAFE-Net model (meta learner + prediction network)
zero-shot/              ZSL data loader and training script
compositional-zs/       attribute-object composition loader and training script
tools/
  fetch_local.sh        download + checksum every public source
  check_data.py         data preflight (split sizes, GloVe coverage, feature files)
  build_compose_data.py rebuild MIT-States / StanfordVRD splits and metadata
  extract_features.py   extract ImageNet CNN features for the compositional task
  run_zsl.py            run ZSL configs over several seeds
  summarize_zsl.py      collect logs into the results table
results/                summarized results (committed)
```

## Citation

```bibtex
@inproceedings{wang2019tafe,
  title={TAFE-Net: Task-Aware Feature Embeddings for Low Shot Learning},
  author={Wang, Xin and Yu, Fisher and Wang, Ruth and Darrell, Trevor and Gonzalez, Joseph E},
  booktitle={CVPR}, pages={1831--1840}, year={2019}
}
```
