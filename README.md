<div align="center">

# NexusRAG

<h3>Corpus-Guided Dual-Path Propagation for Graph Retrieval-Augmented Generation</h3>

[![arXiv](https://img.shields.io/badge/arXiv-2609.37661-B31B1B.svg)](https://arxiv.org/abs/2609.37661)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-2E7D32.svg)](LICENSE.txt)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776AB.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/Jacob-biu/NexusRAG?logo=github&label=stars)](https://github.com/Jacob-biu/NexusRAG)
[![Last commit](https://img.shields.io/github/last-commit/Jacob-biu/NexusRAG?logo=github)](https://github.com/Jacob-biu/NexusRAG)

</div>

---

<p align="center">
  <img src="figure/main_figure.svg" width="100%" alt="NexusRAG framework overview">
</p>

<p align="center">
  <sub><b>Figure 1.</b> NexusRAG in four stages. <b>(a) Graph construction</b> — a relation-free Tri-Graph over passages, entities, and sentences, built with no LLM relation extraction. <b>(b) Neighbor prior</b> — corpus-level proximity fuses structural entity co-occurrence with semantic entity similarity, and weight plus rank pruning produce a sparse neighbor prior <code>W</code>. <b>(c) Entity propagation</b> — semantic propagation with neighbor clamping runs alongside structural propagation via corpus-level neighbors, and their union becomes the next activation. <b>(d) Passage retrieval</b> — the propagated evidence weights initialize the passage scores that drive Personalized PageRank down to the retrieved passages.</sub>
</p>

---

## 🔍 Overview

Existing relation-free graph retrieval methods rely primarily on query–sentence similarity to search for evidence, which leads to two failure modes. **Query-gated cutoff** occurs when a sentence connecting relevant entities has low query similarity, weakening the activation signal and preventing propagation from reaching necessary bridging evidence; **spurious activation** occurs when a query-relevant sentence mentions incidental entities, letting activation spread to entities that do not support the answer.

NexusRAG is a **simple, effective, and efficient** approach that augments the relation-free Tri-Graph with a **corpus-guided** entity neighborhood prior `W` — fused from corpus-level co-occurrence and semantic similarity, and built **without any LLM-based relation extraction**. That prior drives a **dual-path propagation** mechanism which gates query-similar sentences with the structural prior and expands the resulting frontier along the neighbor structure. Indexing therefore consumes **no LLM tokens** — only a one-time neighbor precompute — while NexusRAG consistently outperforms existing approaches across three multi-hop QA benchmarks and a domain-specific GraphRAG-Bench subset.

<table>
  <tr>
    <td align="center" width="33%">
      <h3>🧩&nbsp; Relation-free</h3>
      <sub>No LLM relation extraction at indexing time, so the index costs <b>no tokens</b> and stays as linear as the base Tri-Graph.</sub>
    </td>
    <td align="center" width="33%">
      <h3>🔀&nbsp; Corpus-guided</h3>
      <sub>Co-occurrence and semantic similarity are fused into one sparse neighbor prior <b>W</b>; it clamps the query-gated <b>semantic</b> path and expands the <b>structural</b> path to partners no query-similar sentence can bridge.</sub>
    </td>
    <td align="center" width="33%">
      <h3>🎯&nbsp; Neighbor-aware</h3>
      <sub>Cumulative evidence weights <b>initialize the passage scores</b> that drive Personalized PageRank, instead of starting from uniform mass.</sub>
    </td>
  </tr>
</table>

---

## ✨ Highlights

- **Dual-path entity propagation: gating and direct expansion.** Entity activation proceeds over two complementary paths.
  - **Semantic propagation** walks from each active entity through its top-η query-similar sentences, where the neighbor clamp caps the transition by the fused neighbor weight `w_ij` and the pruning threshold δ sets the floor for non-neighbor transitions.
  - **Structural propagation** then expands that query-conditioned frontier along the precomputed neighbor structure, so a structurally supported partner stays reachable even without a query-similar bridging sentence.
  - The union of the two paths forms the next frontier, and every admitted transition accumulates into the target entity's cumulative evidence weight.
- **Corpus-guided propagation.** Neighbor weights are fused into a single matrix `W` controlled by `cooccur_alpha`, which balances co-occurrence evidence against semantic similarity. The cumulative evidence weights produced by dual-path propagation initialize the passage scores, which then drive Personalized PageRank down to the top-k passages.
- **Cheap by construction.** The neighbor precompute is a one-time cost of **2.6%** (5M tokens) and **2.8%** (10M) of NexusRAG's total indexing time — below 3% at both scales — and doubling the corpus grows total cost by **2.05×**, essentially the same rate as the LinearRAG base index. Against an LLM-extraction pipeline the gap is architectural: HippoRAG's total index time exceeds NexusRAG's by a factor of **17.5× at 5M** and **12.1× at 10M**.

---

## 📊 Results

<p align="center">
  <sub><b>Table 1.</b> Main results on HotpotQA, 2Wiki, MuSiQue, and Medical. Con. = containment, LLM. = LLM-judged accuracy, Avg. = their average. NexusRAG is compared against direct zero-shot LLM inference, vanilla retrieval-augmented generation, graph-based RAG methods, and linear graph RAG methods. Best per column in <b>bold</b>.</sub>
</p>

<p align="center">
  <img src="figure/main_result.png" width="88%" alt="Main results across four benchmarks">
</p>

<p align="center">
  <sub><b>Table 2.</b> Evidence recall and relevance by question category on the GraphRAG-Bench subset — fact retrieval, complex reasoning, contextual, and creative generation. NexusRAG records the highest evidence recall in every category. Best per column in <b>bold</b>, second best <u>underlined</u>.</sub>
</p>

<p align="center">
  <img src="figure/retrieval_quality.png" width="88%" alt="Evidence retrieval quality on the GraphRAG-Bench subset">
</p>

---

## 🛠️ Installation

### 1. Python packages

Python 3.9 is recommended.

```bash
pip install -r requirements.txt
```

<details>
<summary>What the stack is used for</summary>

| Stage | Libraries |
| :--- | :--- |
| Sentence & entity embeddings | `sentence-transformers`, `transformers`, `torch`, `huggingface-hub` |
| Entity recognition | `spacy` |
| Index & graph | `faiss-cpu`, `python-igraph` |
| LLM client (QA + evaluation) | `openai`, `httpx`, `python-dotenv` |
| Numerics & plots | `numpy`, `scipy`, `scikit-learn`, `pandas`, `pyarrow`, `matplotlib`, `seaborn`, `tqdm` |

</details>

### 2. SpaCy language model

```bash
python -m spacy download en_core_web_trf
```

> [!NOTE]
> `--spacy_model` accepts any SpaCy pipeline. For a biomedical corpus, install and pass the scientific model instead:

```bash
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_scibert-0.5.3.tar.gz
python run.py --dataset_name medical --spacy_model en_core_sci_scibert --cooccur_alpha 0.5
```

### 3. LLM service

The QA and evaluation stages talk to any OpenAI-compatible endpoint. Point the client at your own server through environment variables, or set it per run with `--llm_base_url`.

```bash
export OPENAI_API_KEY="your-api-key-here"
export OPENAI_BASE_URL="http://localhost:8009/v1"
```

> [!NOTE]
> `OPENAI_BASE_URL` falls back to `http://localhost:8009/v1` (a local vLLM-style deployment) when it is not set, and the `Authorization` header is only attached when `OPENAI_API_KEY` is non-empty.

### 4. Datasets

Place every dataset under `import/dataset/<dataset_name>/` with two files, `questions.json` and `chunks.json`:

```
import/dataset/
├── 2wikimultihop/{questions.json, chunks.json}
├── hotpotqa/{questions.json, chunks.json}
├── medical/{questions.json, chunks.json}
└── musique/{questions.json, chunks.json}
```

### 5. Embedding model

The default `--embedding_model` is a path relative to the project root:

```
model/all-mpnet-base-v2/
```

`scripts/run.sh` instead points at the absolute location `/mnt/model/sentence-transformers/all-mpnet-base-v2`, so either place the model accordingly or override `EMBEDDING_MODEL` in the script.

---

## ⚡ Quick start

```bash
EMBEDDING_MODEL="/mnt/model/sentence-transformers/all-mpnet-base-v2"
LLM_MODEL="gpt-4o-mini"
MAX_WORKERS=16

python run.py \
    --spacy_model "en_core_web_trf" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --dataset_name "2wikimultihop" \
    --llm_model "${LLM_MODEL}" \
    --max_workers "${MAX_WORKERS}" \
    --max_iterations 3 \
    --iteration_threshold 0.4 \
    --top_k_sentence 1 \
    --top_k_entity_cooccur 5 \
    --precompute_threshold 0.5 \
    --cooccur_alpha 0.5
```

Two optional flags can be appended to the same command:

```bash
    --use_vectorized_retrieval   # vectorized sparse-matrix propagation instead of BFS iteration
    --retrieval_only             # run retrieval only, skipping QA and evaluation
```

The command above is exactly what `scripts/run.sh` issues for one dataset. To run the full four-dataset sweep, use the script directly:

```bash
bash scripts/run.sh
```

Hyper-parameters baked into `scripts/run.sh`:

| Dataset | `--max_iterations` | `--iteration_threshold` | `--top_k_sentence` | `--precompute_threshold` | `--cooccur_alpha` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `2wikimultihop` | 3 | 0.4 | 1 | 0.5 | 0.5 |
| `hotpotqa` | 3 | 0.4 | 1 | 0.5 | 0.5 |
| `medical` | 3 | 0.5 | 3 | 0.5 | 0.5 |
| `musique` | 5 | 0.1 | 4 | 0.5 | 0.5 |

`--spacy_model en_core_web_trf`, `--top_k_entity_cooccur 5` and `--max_workers 16` are shared by all four runs.

> [!IMPORTANT]
> `scripts/run.sh` assumes the project is checked out at `/mnt/NexusRAG` and that the embedding model sits at `/mnt/model/sentence-transformers/all-mpnet-base-v2`. Adjust the `cd` target and `EMBEDDING_MODEL` at the top of the script for your own machine. `LLM_MODEL` is a placeholder there, so set it to a model name actually served by your endpoint.

Omitting `--cooccur_alpha` runs the full sweep over 21 values from 0 to 1 in steps of 0.05. A comma-separated list such as `--cooccur_alpha 0.0,0.5,1.0` runs only the given values.

---

## ⚙️ Key arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--dataset_name` | `novel` | Dataset folder name under `import/dataset/` |
| `--spacy_model` | `en_core_web_trf` | SpaCy pipeline used for entity recognition |
| `--embedding_model` | `model/all-mpnet-base-v2` | SentenceTransformer model path |
| `--llm_model` | `""` | QA model name; falls back to the client default |
| `--llm_base_url` | `""` | QA service URL; falls back to `OPENAI_BASE_URL` |
| `--eval_llm_model` | `""` | Dedicated evaluation model; reuses the QA model when empty |
| `--eval_llm_base_url` | `""` | Dedicated evaluation URL; reuses the QA URL when empty |
| `--cooccur_alpha` | `None` | Fuse weight between co-occurrence and similarity; sweep when unset |
| `--top_k_entity_cooccur` | `5` | Cap on co-occurring entities kept per entity when building `W` |
| `--top_k_sentence` | `3` | Top-k query-similar sentences (η) per active entity on the Sentence-Mediated Path |
| `--precompute_threshold` | `0.5` | Similarity threshold applied when precomputing the neighbor matrix |
| `--max_iterations` | `3` | Maximum number of propagation hops |
| `--iteration_threshold` | `0.4` | Pruning threshold δ; transitions below it are discarded, non-neighbor transitions are kept only at this floor |
| `--max_workers` | `16` | Worker count for QA and evaluation |
| `--use_vectorized_retrieval` | off | Use vectorized propagation instead of BFS iteration |
| `--no_resume` | off | Disable resume mode (resume is on by default) |
| `--resume_save_every` | `20` | Save `predictions.json` every N entries in resume mode |
| `--retrieval_only` | off | Run retrieval only, write `retrieval.json`, skip QA and evaluation |

---

## 🗂️ Project structure

```
NexusRAG/
├── run.py                      # Entry point: alpha sweep, resume logic, result CSV and plots
├── src/
│   ├── NexusRAG.py             # Core pipeline: indexing, neighbor matrix, dual-path propagation, retrieval
│   ├── config.py               # NexusRAGConfig dataclass
│   ├── embedding_store.py      # Embedding cache and cosine-similarity lookups
│   ├── evaluate.py             # Evaluator: EM and containment accuracy
│   ├── ner.py                  # SpacyNER wrapper
│   └── utils.py                # LLM_Model (OpenAI-compatible client) and logging setup
├── scripts/
│   └── run.sh                  # Four-dataset sweep ready to run
├── figure/
│   ├── main_figure.svg         # Figure 1: framework overview
│   ├── main_result.png         # Table 1: main results
│   └── retrieval_quality.png   # Table 2: evidence retrieval quality
├── requirements.txt
└── LICENSE.txt                 # GPL-3.0
```

---

## 📤 Outputs

Everything lands under a run-specific directory derived from the retrieval hyper-parameters:

```
results/alpha_sweep_s<>_k<top_k_entity_cooccur>_<llm_model>/
└── threshold_<precompute_threshold>/
    └── <dataset_name>/
        ├── alpha_sweep_results.csv      # per-alpha EM / Contain_Acc, resume source
        ├── alpha_sweep_lineplot.png     # alpha sensitivity, line plot
        ├── alpha_sweep_barplot.png      # alpha sensitivity, bar plot
        └── alpha_<alpha>_<timestamp>/   # per-alpha run
            ├── log.txt
            ├── retrieval.json
            ├── evaluation_results.json
            └── predictions.json
```

`alpha_sweep_results.csv` doubles as the resume ledger: any `cooccur_alpha` already present is skipped, and a run whose `retrieval.json` exists without a `predictions.json` is reused instead of recomputed. `--retrieval_only` writes only `retrieval.json` and leaves the CSV untouched.

---

## 📚 Citation

If you find NexusRAG useful in your research, please cite:

```bibtex
@article{liu2026nexusrag,
  title={Corpus-Guided Dual-Path Propagation for Graph Retrieval-Augmented Generation},
  author={Liu, Baoxian and Wei, Tong},
  journal={arXiv preprint arXiv:2609.37661},
  year={2026}
}
```

---

## 📄 License

Released under the **GNU General Public License v3.0 (GPL-3.0)**. See [`LICENSE.txt`](LICENSE.txt) for the full text.
