
[![Dataset on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/alihassan1437/causalrec-bench)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

# CausalRec-Bench

**CausalRec-Bench: A Multi-Domain Semi-Synthetic Benchmark for Causal Recommendation Under Multiple Biases and Concept Drift**

Ali Hassan — RecSys 2026 CONSEQUENCES Workshop  
Collaborating with Dr. Yan Zhang, Charles Darwin University

---

## Quick Start — 3 Commands

```bash
git clone https://github.com/alihassan1437/causalrec-bench.git
cd causalrec-bench
pip install -r requirements.txt
python download_data.py          # downloads full dataset from Hugging Face
python benchmark/run_evaluation.py
```

> **Note:** The full dataset (2.4M interactions, 50k users, 4k items) and pre‑trained models are hosted on Hugging Face. Running `download_data.py` will fetch them into `data/` and `pretrained_models/`.

---

## What Is CausalRec-Bench?

The **first** semi‑synthetic benchmark for causal recommendation evaluation with **ground‑truth causal labels on every interaction**. Covers seven evaluation dimensions simultaneously:

- ✅ Cold‑start users  
- ✅ Warm users  
- ✅ Progressive difficulty levels (no confounders → all confounders)  
- ✅ Seasonal concept drift (4 seasons)  
- ✅ Cross‑domain generalisation (e‑commerce → streaming)  
- ✅ Position bias isolation  
- ✅ Causal‑aware vs. standard metric comparison  

| Statistic | Value |
|-----------|-------|
| Users | 50,000 |
| Items | 4,000 (2 domains) |
| Interactions | 2,468,985 |
| Domains | E‑commerce + Streaming |
| Confounders | 5 |
| Cold‑start users | 9,896 (19.8%) |
| Non‑genuine clicks | 70.0% |
| Evaluation splits | 18 |

---

## Key Findings (Verified on Colab)

| Finding | Scenario | Result |
|---------|----------|--------|
| Causal MF vs Standard MF | Cold‑Start | **+46.1%** CP@10 |
| Causal LightGCN vs Standard LightGCN | Level 3 Hard (warm users) | **+32.9%** CP@10 |
| Non‑genuine clicks | Entire benchmark | **70.0%** confounder‑driven |
| Position bias ratio (designed) | All interactions | 1.94× (position 1 vs 10) |
| Promotion bias ratio (measured) | All interactions | 1.55× |
| Graph methods cold‑start | Zero‑history users | **Identical** regardless of causal training |

---

## Five Confounders

| Confounder | Simulation Mechanism | Measured Effect |
|------------|----------------------|------------------|
| **Promotion bias** | +40% exposure; +15% click multiplier | 1.55× click rate |
| **Popularity bias** | Social proof by popularity tier | — |
| **Position bias (novel)** | +25% at pos 1 → +1% at pos 10 | **1.94×** (pos 1 vs 10) |
| **Seasonal concept drift** | Winter +15% books; Summer +15% outdoor | Validated |
| **New item penalty** | −20% exposure for new items | Validated |

---

## Causal Ground‑Truth Labels

Every interaction includes a `click_cause` label – **unavailable in any existing public recommendation dataset**:

| Label | Meaning |
|-------|---------|
| `genuine_preference` | Click reflects true user preference |
| `promotion_bias` | Click caused by promotional placement |
| `popularity_bias` | Click caused by social proof |
| `position_bias` | Click caused by display position |
| `mixed` | Multiple confounder effects |
| `no_click` | Item exposed but not clicked |

---

## Complete Results — All 5 Models, All Scenarios (Colab Verified)

### Category Precision@10 (CP@10)

| Model | Cold‑Start | Level 3 Hard | Level 1 Simple | Winter CS | Summer CS | E‑com CS | Stream CS |
|-------|------------|--------------|----------------|-----------|-----------|----------|-----------|
| Popularity | 0.2462 | 0.2463 | 0.1058 | 0.2230 | 0.2732 | 0.0000 | 0.3865 |
| Standard MF | 0.2835 | 0.2900 | 0.1183 | 0.2869 | 0.2853 | 0.1696 | 0.2679 |
| **Causal MF** | **0.4140** | 0.4241 | 0.1852 | 0.4156 | 0.4146 | 0.3853 | 0.3037 |
| Standard LightGCN | 0.5577 | 0.5450 | 0.2260 | 0.5645 | 0.5618 | 0.5045 | 0.4049 |
| **Causal LightGCN** | 0.5577 | **0.7245** | **0.3548** | 0.5645 | 0.5618 | 0.5045 | 0.4049 |

*CS = Cold‑Start, **bold** = best non‑oracle result per column.*

---

## 18 Evaluation Splits

| Category | Splits | Purpose |
|----------|--------|---------|
| Standard | `train` / `val` / `test` | Baseline evaluation |
| Cold‑start | `cold_start` | Zero‑history users |
| Difficulty | `level1_simple` / `level2_medium` / `level3_hard` | Progressive confounder complexity |
| Seasonal | `winter` / `summer` / `autumn` / `spring` cold‑start | Concept drift evaluation |
| Domain | `ecom_cold` / `stream_cold` | Cross‑domain generalisation |
| Position | `high_position` / `low_position` | Position bias isolation |

---

## Reproduce From Scratch

```bash
# Step 1 - Generate full benchmark dataset (~2 minutes)
python benchmark/generate_benchmark.py

# Step 2 - Train all models (~30 minutes on CPU)
python benchmark/train_models.py

# Step 3 - Run full evaluation (~15 minutes)
python benchmark/run_evaluation.py

# Step 4 - Generate publication charts
python benchmark/generate_charts.py
```

---

## Evaluate Your Own Model

```python
from evaluation.metrics import evaluate_model
import pandas as pd

# Load any evaluation split (after running download_data.py)
cold_start = pd.read_csv('data/cold_start.csv')
users = pd.read_csv('data/users.csv')
items = pd.read_csv('data/items.csv')

def my_recommender(user_id, user_info, items_df, k=10):
    # Your recommendation logic here
    return [item_id_1, item_id_2, ...]

results = evaluate_model(
    model_name='MyModel',
    model_func=my_recommender,
    test_data=cold_start,
    users_df=users,
    items_df=items,
    k=10
)

# Returns: precision@10, recall@10, ndcg@10,
#          genuine_p@10, category_p@10
print(results)
```

---

## Pre‑trained Models

Pre‑trained models are fetched by `download_data.py` into `pretrained_models/`:

| File | Description | Shape / Size |
|------|-------------|---------------|
| `pretrained_models/fmf_std_U.npy` | Standard MF user embeddings | 35,000 × 32 |
| `pretrained_models/fmf_std_V.npy` | Standard MF item embeddings | 4,000 × 32 |
| `pretrained_models/fmf_caus_U.npy` | Causal MF user embeddings | 35,000 × 32 |
| `pretrained_models/fmf_caus_V.npy` | Causal MF item embeddings | 4,000 × 32 |
| `pretrained_models/lgcn_std.pt` | Standard LightGCN weights | 2.5M parameters |
| `pretrained_models/lgcn_caus.pt` | Causal LightGCN weights | 2.5M parameters |

**Training details:**
- Standard models: 575,553 training clicks  
- Causal models: 172,870 genuine clicks (402,683 biased removed)  
- LightGCN: 50 epochs, embedding dim 64, 3 propagation layers  

---

## Project Structure (after running `download_data.py`)

```
causalrec-bench/
├── README.md
├── requirements.txt
├── download_data.py
├── data/                    # 18 CSV evaluation splits (fetched)
├── pretrained_models/       # 6 model files (fetched)
├── benchmark/
│   ├── generate_benchmark.py
│   ├── generate_charts.py
│   ├── run_evaluation.py
│   └── train_models.py
├── evaluation/
│   ├── __init__.py
│   └── metrics.py
├── models/
│   ├── __init__.py
│   └── fast_mf.py
├── figures/
│   ├── key_findings.png
│   ├── main_results.png
│   └── validation.png
└── results/
    └── final_results.csv
```

---

## Citation

```bibtex
@inproceedings{hassan2026causalrecbench,
  title     = {CausalRec-Bench: A Semi-Synthetic Benchmark for Evaluating
               Causal Cold-Start Recommendation Under Exposure Bias
               and Concept Drift},
  author    = {Hassan, Ali},
  booktitle = {Proceedings of the CONSEQUENCES Workshop at ACM RecSys 2026},
  year      = {2026},
  note      = {Collaborating with Dr. Yan Zhang, Charles Darwin University}
}
```

---

## License

- **Code:** MIT License  
- **Dataset:** CC BY 4.0  
- **Pre‑trained Models:** CC BY 4.0

---

## Links

- 📦 **Hugging Face Dataset:** [alihassan1437/causalrec-bench](https://huggingface.co/datasets/alihassan1437/causalrec-bench)
- 🐙 **GitHub Repository:** [alihassan1437/causalrec-bench](https://github.com/alihassan1437/causalrec-bench)
