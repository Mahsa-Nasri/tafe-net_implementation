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

> Status: all five ZSL datasets run in both the ZSL and GZSL settings, 3 seeds each. None reaches the
> paper's numbers (see [2.3](#23-results)). The compositional (Table 4) pipeline is written but not yet run.

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

### 2.3 Results

All five datasets, both settings, mean ± std over seeds 0, 1, 2. Full table with every metric:
[results/zsl.md](results/zsl.md).

"Best epoch" is the epoch with the highest upstream metric on the test set (how the upstream code picks
its number). "Final epoch" is the last epoch, with no selection on the test set. "Standard" is per-class
accuracy over the whole test set (Xian et al.); "upstream" is the batched metric (difference 1 below).

**Conventional ZSL, top-1 (%)**

| config | task-net hidden | best epoch, upstream | best epoch, standard | final epoch, standard | paper |
|---|---|---|---|---|---|
| `sun-zsl` | 2048 | 11.6 ± 0.1 | 10.8 ± 0.1 | 10.6 ± 0.4 | 60.9 |
| `cub-zsl` | 2048 | 21.2 ± 0.7 | 20.1 ± 0.5 | 20.2 ± 0.6 | 56.9 |
| `awa1-zsl` | 2048 | 41.4 ± 5.9 | 38.9 ± 8.1 | 39.0 ± 7.0 | 70.8 |
| `awa2-zsl` | 2048 | 36.7 ± 3.2 | 32.5 ± 4.9 | 25.8 ± 3.7 | 69.3 |
| `awa2-zsl-upstream` | 512 | 48.7 ± 4.4 | 45.3 ± 3.9 | 33.0 ± 0.9 | 69.3 |
| `apy-zsl` (2-layer) | 2048 | 24.8 ± 4.6 | 28.5 ± 4.2 | 21.3 ± 3.3 | 42.2 |

**Generalized ZSL, standard metric (%)**: u = unseen, s = seen, H = harmonic mean

| config | best epoch u | best epoch s | best epoch H | final epoch H | paper u / s / H |
|---|---|---|---|---|---|
| `sun-gzsl` | 2.6 ± 0.5 | 0.9 ± 0.1 | 1.3 ± 0.1 | 1.2 ± 0.1 | 27.9 / 40.2 / 33.0 |
| `cub-gzsl` | 11.7 ± 1.2 | 2.6 ± 0.6 | 4.3 ± 0.9 | 3.8 ± 0.6 | 41.0 / 61.4 / 49.2 |
| `awa1-gzsl` | 17.6 ± 5.7 | 26.4 ± 26.1 | 15.3 ± 8.1 | 7.0 ± 0.7 | 50.5 / 84.4 / 63.2 |
| `awa2-gzsl` | 10.9 ± 6.7 | 23.0 ± 15.8 | 14.4 ± 9.4 | 0.5 ± 0.4 | 36.7 / 90.6 / 52.2 |
| `awa2-gzsl-upstream` | 22.2 ± 0.3 | 84.1 ± 1.3 | 35.1 ± 0.5 | 28.3 ± 10.0 | 36.7 / 90.6 / 52.2 |
| `apy-gzsl` (2-layer) | 7.6 ± 1.1 | 63.9 ± 6.2 | 13.5 ± 1.8 | 12.3 ± 2.9 | 24.3 / 75.4 / 36.8 |

What we observe:

- **No dataset reproduces the paper.** The gap is smallest for aPY ZSL (about 14 points) and largest for
  SUN (50 points in ZSL; GZSL H of 1.3 against 33.0). This holds even for the most generous number
  (best epoch chosen on the test set).
- **The upstream config's hidden size (512) beats the paper's (2048)** on AWA2: by 12 points in ZSL and by
  21 points of H in GZSL. It is the only GZSL config whose seen-class accuracy is near the paper's.
- **With hidden size 2048 the model underfits.** The running training top-1 at the last epoch (seed 0)
  is 16% on SUN, 26% on CUB and 56% on AWA2, against 91 to 93% on AWA1 and aPY and 99.6% for
  AWA2 with hidden size 512. On SUN and CUB GZSL, seen-class accuracy (0.9% and 2.6%) is even lower than
  unseen-class accuracy.
- **The classification loss sits at about 0.69 = ln 2 for the whole run.** That is the lowest value
  `BCEWithLogitsLoss` can reach when its input has already gone through a sigmoid (difference 3 below):
  the input is confined to (0, 1), so the loss can't fall below ln 2 for negative classes. We think this
  double sigmoid is the main cause of the gap, but we have not tested it yet (see [Extension](#4-extension)).
- **Where the model does fit, it overfits.** On AWA2, unseen-class accuracy peaks early (epochs 4 to 27)
  and then falls, while training accuracy with hidden size 512 reaches 99%.
- **GZSL is unstable on AWA1/AWA2 with hidden 2048.** Some epochs collapse to predicting only seen or only
  unseen classes. When u = s = 0 the upstream harmonic mean divided by zero; we added a small epsilon.
- Runs are deterministic: the same seed gives the same numbers.

A 60-epoch AWA2 run takes about 30 minutes on an RTX 3080 Laptop GPU; SUN GZSL is the slowest at
about 2.5 hours per run.

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
from `validate()`, which returns one; the GZSL harmonic mean divided by zero when u = s = 0 (now `+ 1e-12`,
which only changes epochs where both are 0); added `--seed`.

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
