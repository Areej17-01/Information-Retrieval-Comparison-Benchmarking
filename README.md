# RAG Comparison Benchmarking

This repository is a small **information retrieval benchmark** on the **SciFact** dataset (BEIR format). It was prepared in the context of an R2L Lab onboarding quiz : Implemented several retrievers, export ranked document lists, and evaluate them with standard IR metrics. The focus is on **comparing retrieval strategies** (sparse, dense, hybrid), not on building a full retrieval augmented generation stack with a language model.

For a full writeup with figures, animated plots, and discussion, see **[report.md](report.md)**. For a PDF style document you can upload to **Overleaf**, see **[latex/RAG_Benchmark_Report.tex](latex/RAG_Benchmark_Report.tex)**.

## What this project does

1. **Dense retrieval** encodes the corpus with `sentence-transformers` (`all-MiniLM-L6-v2`), indexes embeddings with **FAISS** (inner product on L2 normalized vectors, cosine style similarity), and retrieves the top 100 documents per query.
2. **Sparse retrieval** builds a **BM25** index (`rank-bm25`) over simple tokenized text and retrieves the top 100 documents per query.
3. **Hybrid retrieval** combines BM25 and dense scores with a configurable **alpha** weight on BM25, after per query min max normalization (`hybrid_retrieval.py`).
4. **Hybrid with RRF** merges BM25 and dense rankings using **reciprocal rank fusion** with configurable **k** (`hybrid_retrieval_rff.py`).
5. **Evaluation** uses the **BEIR** `EvaluateRetrieval` helper on the SciFact **test** qrels (`evaluation.py`).

Committed **result JSON files** under `results/` correspond to runs you can re-evaluate or extend.

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3 |
| Dataset loader and metrics | [BEIR](https://github.com/beir-cellar/beir) (`GenericDataLoader`, `EvaluateRetrieval`) |
| Dense embeddings | [Sentence Transformers](https://www.sbert.net/) |
| Vector search | [FAISS](https://github.com/facebookresearch/faiss) (`IndexFlatIP`) |
| Sparse retrieval | [rank-bm25](https://github.com/dorianbrown/rank_bm25) (`BM25Okapi`) |
| Numerics and progress | NumPy, tqdm |
| Optional figures for the report | matplotlib, Pillow (see `scripts/generate_report_figures.py`) |

Dependencies are listed in the `requirements` file (unversioned; pin versions for stricter reproducibility if you need them).

## Quick start

```bash
pip install -r requirements
```

Ensure SciFact is available under `datasets/scifact/` in BEIR layout (`corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`, etc.). If you only have a partial tree, run `sparse_retrieval.py` once: it can download the dataset via BEIR utilities.

**Run retrievers (examples):**

```bash
python dense_retrieval.py
python sparse_retrieval.py
python hybrid_retrieval.py
python hybrid_retrieval_rff.py
```

**Evaluate a run:**

```bash
python evaluation.py datasets/scifact results/dense_results.json
```

**Regenerate report figures (PNG and GIF):**

```bash
pip install -r requirements-figures.txt
python scripts/generate_report_figures.py
```

## What can be enhanced

- **Reranking:** Add a cross encoder reranker on the top 50 dense or hybrid hits. This is the most common next step for measurable NDCG@10 gains.
- **Models:** Try domain suited encoders for scientific text instead of only MiniLM.
- **Tokenization:** Replace whitespace tokenization with a biomedical or general NLP tokenizer for BM25.
- **Scale:** Move from `IndexFlatIP` to approximate FAISS indexes when the corpus grows.
- **Experiment tracking:** Emit pure JSON score files (today some files under `scores/` mirror console text and are not valid JSON).
- **Dashboard:** Fix paths and data loading in `scores/dashboard.py` so it reads evaluation output reliably.
- **Packaging:** Add a `pyproject.toml` or pin versions in `requirements.txt` for repeatable installs.

## Documentation map

| Document | Contents |
|----------|----------|
| [report.md](report.md) | Detailed methodology, metric definitions, full results table, static and animated plots, limitations |
| [latex/RAG_Benchmark_Report.tex](latex/RAG_Benchmark_Report.tex) | LaTeX article version for Overleaf or local `pdflatex` |

### Overleaf

Create a new Overleaf project. Upload `latex/RAG_Benchmark_Report.tex` and the entire `report_assets` folder. Set the compiler to **pdfLaTeX**. The `.tex` file uses `\graphicspath{{../report_assets/}}` so it expects `report_assets` to sit **next to** the `latex` folder (same layout as this repo). If you prefer a flat Overleaf project with only one folder, move the `.tex` file to the project root next to `report_assets` and change that line to `\graphicspath{{report_assets/}}` as noted in the file header comment.

## License

See [LICENSE](LICENSE) (Apache 2.0 template in repository).
