## Hybrid retrieval
Training and evaluation of a two-stage retrieval pipeline: retriever top-n candidates → cross-encoder reranker reordering.

**approaches**:

| Stage | Variants |
| --- | --- |
| Retriever | dense <br> dense + BM25 <br> **dense + sparse (splade)** ← best |
| Reranker | cross-encoder |

<br>

**final setup**<br>
Retriever (dense) top-20  →  Reranker (cross-encoder)


**retriever**

| | |
| --- | --- |
| model | `deepvk/USER-bge-m3` |
| loss | `CachedMultipleNegativesRankingLoss` |
| batch sampler | `NO_DUPLICATES` |
| hard negatives | 2 per query: top-1 + 1 random from the ranking range<br>`range_min=5`, `range_max=35`, `num_negatives=30` |

**reranker**

| | |
| --- | --- |
| model | `BAAI/bge-reranker-v2-m3` |
| loss | `CachedMultipleNegativesRankingLoss` |

**evaluation (retriever → reranker)**:

| k | accuracy | precision | recall | ndcg |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.8973 | 0.8973 | 0.8973 | 0.8973 |
| 3 | 0.9821 | 0.3274 | 0.9821 | 0.9485 | 
| 5 | 0.9955 | 0.1991 | 0.9955 | 0.9541 | 
| 10 | 1.0000 | 0.1000 | 1.0000 | 0.9557 | 

`map@20 = 0.9406`, `mrr@10 = 0.9406`

<br>

**sparse (splade)**

| | |
| --- | --- |
| model | `opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1` |
| loss | `CachedSpladeLoss` over `SparseMultipleNegativesRankingLoss` |
| regularizer weight | `3e-3` for queries and documents |
| batch sampler | `NO_DUPLICATES` |

**evaluation**:

| k | accuracy | precision | recall | ndcg |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7813 | 0.7813 | 0.7813 | — |
| 3 | 0.9241 | 0.3080 | 0.9241 | — |
| 5 | 0.9554 | 0.1911 | 0.9554 | — |
| 10 | 0.9866 | 0.0987 | 0.9866 | 0.8856 |

`map@20 = 0.8531`, `map@100 = 0.8534`, `mrr@10 = 0.8528`

**sparsity**:

| | active dims | sparsity ratio |
| --- | ---: | ---: |
| query | 215.5 | 0.9980 |
| corpus | 361.7 | 0.9966 |

`avg_flops = 53.09`

**RRF (dense + splade)**:

| | map | mrr@10 | ndcg@10 |
| --- | ---: | ---: | ---: |
| dense | 0.8688 | 0.8685 | 0.8997 |
| sparse (splade) | 0.8534 | 0.8528 | 0.8856 |
| **fusion** | **0.8881** | **0.8881** | **0.9160** |

Learned term weights for a query:

```python
output = model_sparse_splade.encode(['нормы закрепления при штормовом ветре'])
model_sparse_splade.decode(output, top_k=10)
```

| token | weight |
| --- | ---: |
| `##ет` | 2.0054 |
| `##ете` | 1.9371 |
| `##ре` | 1.7489 |
| `##рм` | 1.2864 |
| `за` | 1.2845 |
| `но` | 1.2567 |
| `што` | 1.2246 |
| `##к` | 1.2002 |
| `##пления` | 1.1821 |
| `##пление` | 1.0149 |


## Installation

Clone the repo:

```bash
git clone https://github.com/ilyagvozdarev/hybrid-retrieval.git
cd hybrid-retrieval
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv\Scripts\Activate.ps1    # Windows
source venv/bin/activate            # Linux / macOS
```

Install the requirements:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```


## Run

### 1. Mine hard negatives

Produces the training dataset in n-tuple format (`anchor`, `positive`, `negative_1`, …).
Mining parameters live in `configs/mine_config.yaml`.

```bash
python mine_hard_negatives.py \
--model "deepvk/USER-bge-m3" \
--mine_config "configs/mine_config.yaml" \
--qrels "data/dataset/qrels.json"
```

`--qrels` expects a flat `{query_id: passage_id}` mapping. That file is not in the repo — only `inv_qrels.json`,
the inverted form used for evaluation. Skip this step and use the checked-in `hard_negatives/ds_h2.json`
to reproduce the runs below.

### 2. Train

train dense retriever:
```bash
python train_retriever_dense.py \
--model "deepvk/USER-bge-m3" \
--train_config "configs/train/train_config.yaml" \
--data_dir "./data/dataset" \
--inv_qrels "inv_qrels.json" \
--passages_splits "passages_splits2.json" \
--passages "passages.json" \
--queries "queries.json" \
--train_dataset "hard_negatives/ds_h2.json" \
--out_dir "result"
```

train sparse (splade) retriever and reranker (see the docstring for CLI args):
```bash
python train_retriever_splade.py
python train_reranker.py
```

## Data

Rules for the technical operation of railways<br>
**source**: https://www.tdesant.ru/info/item/316

synthetic generation:
- hierarchical chunking by document structure (total corpus - 918 passages)
- sampling ~1-3 passages from several sections (total ~100 passages)
- generating 5-10 queries for each passage (prompt: `data/prompts`, models: `nvidia/MiniMax-M3-NVFP4`, `Claude Sonnet 5`) with manual review
- split by passages, stratified by sections


All paths below are relative to `--data_dir`.

| File | Format | Size |
| --- | --- | --- |
| `queries.json` | `{"query": [...], "id": [...]}` | 639 queries |
| `passages.json` | `{"passage": [...], "id": [...]}` | 918 passages |
| `inv_qrels.json` | `{"passage_id": [...], "queries_ids": [[...], ...]}` | 96 passages with queries |
| `passages_splits2.json` | `{"train": [passage_id, ...], "test": [...]}` | 61 / 35 |
| `hard_negatives/ds_h2.json` | JSON Lines: `anchor`, `positive`, `negative_1`, `negative_2` | 415 rows |

Every query has exactly one relevant passage. Evaluation uses only the queries whose passage
falls into the `test` split.

## Layout

```
hybrid_retrieval/       shared code
  data.py               CLI arguments and data loading
  training.py           the training loop
  config.py             layered merging of training configs
  util/                 io, memory estimates, loss introspection
  eval.py               weighted RRF evaluator for dense + sparse fusion (used from notebooks)
  sampler.py            NoDuplicates variant that allows colliding negatives (experimental)

train_retriever_dense.py
train_retriever_splade.py
train_reranker.py
mine_hard_negatives.py
```
