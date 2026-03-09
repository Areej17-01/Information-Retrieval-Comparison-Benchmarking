# R2L Lab Onboarding Quiz

Welcome to the R2L Lab! This quiz is designed to be a challenging and educational experience that will test your ability to quickly ramp up on a new topic, synthesize information, and apply your knowledge to a practical coding problem. It covers fundamental concepts in information retrieval that are central to our work on Retrieval-Augmented Generation (RAG) and building generalist AI agents.

The quiz is divided into two parts. Please complete both.

---


## Part 1: Literature Review


### Objective

The goal of this exercise is to assess your ability to quickly learn a new domain, synthesize information from seminal works, and articulate complex ideas clearly and concisely. You will conduct a focused literature review on sparse retrieval, dense retrieval, and their role in Retrieval-Augmented Generation (RAG).


### Core Readings

You may seek out other works as you see fit, but your review **must** cover the following papers:

1.  **Sparse Retrieval (BM25):** Jones, K. S. (1976). [Relevance Weighting of Search Terms](https://www.staff.city.ac.uk/~sbrp622/papers/RSJ76.pdf). *Journal of documentation*.
2.  **Dense Retrieval (DPR):** Karpukhin, V., et al. (2020). [Dense Passage Retrieval for Open-Domain Question Answering](https://arxiv.org/abs/2004.04906). *EMNLP*.
3.  **Retrieval-Augmented Generation (RAG):** Lewis, P., et al. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). *NeurIPS*.


### Task

Write a literature review that addresses the following questions. Your review should be well-structured, with clear citations where relevant.

For each of the three topics (Sparse Retrieval, Dense Retrieval, and RAG), please answer the following:

*   **What are the authors trying to do?** Articulate their objectives.
*   **How was it done prior to their work, and what were the limits of current practice?**
*   **What is new in their approach, and why do they think it will be successful?**
*   **What are the mid-term and final “exams” to check for success?** (i.e., How is the method evaluated?)

Additionally, please provide a concluding section that synthesizes the information from all three papers:

*   **Who cares? What difference does the author's results make?** Discuss the broader impact of these technologies.
*   **What are the risks?** What are the potential failure modes or downsides of these approaches?
*   **Synthesis:** Briefly explain how these three technologies fit together. How do sparse and dense retrieval support the RAG framework? What are the pros and cons of using one retrieval method over the other in a RAG system?


### Deliverable

Please provide your literature review as a PDF document. There is no strict page limit, but we value conciseness and clarity. Aim for a submission that is thorough yet easy to digest.


---


## Part 2: Implementation Challenge


### Objective

The goal of this exercise is to evaluate your practical coding and problem-solving skills. You will implement and evaluate two fundamental information retrieval methods: a sparse retriever (BM25) and a dense retriever. This will give you hands-on experience with the trade-offs between these two approaches.


### Task

You are tasked with building two retrieval systems and evaluating them on a standard benchmark dataset. We recommend using the **SciFact** dataset, which is part of the BEIR benchmark. It is a good size for this exercise and has a clear fact-checking task.

Your implementation should be in Python. You are free to use any libraries you see fit, but we recommend the following for a straightforward implementation:
*   `beir`: For downloading the dataset.
*   `rank-bm25`: For the sparse retrieval implementation.
*   `sentence-transformers`: For the dense retrieval implementation.
*   `faiss-cpu` or `faiss-gpu`: For efficient similarity search in dense retrieval.


### Steps

1.  **Set up your environment:**
    *   Create a `requirements.txt` file listing all your dependencies.

2.  **Download the data:**
    *   Write a script to download the `scifact` dataset using the `beir` library.

3.  **Implement Sparse Retrieval:**
    *   Create a script that:
        *   Loads the SciFact corpus and queries.
        *   Indexes the corpus using BM25.
        *   For each query, retrieves the top 100 most relevant documents.
        *   Saves the results on the `test` split in the format expected by the `beir` evaluation tool (a JSON file where keys are query IDs and values are dictionaries of doc IDs and scores).

4.  **Implement Dense Retrieval:**
    *   Create a script that:
        *   Loads the SciFact corpus and queries.
        *   Uses a pre-trained `sentence-transformers` model (e.g., `all-MiniLM-L6-v2`) to embed the entire corpus.
        *   Builds a FAISS index for the corpus embeddings.
        *   For each query, embeds the query and uses the FAISS index to retrieve the top 100 most similar documents.
        *   Saves the results on the `test` split in the same format as the sparse retriever.

5.  **Evaluate the Retrievers:**
    *   We have provided a standardized evaluation script, `evaluation.py`. Use it to evaluate your two retrieval systems. This ensures that all candidates are evaluated in the same way.


### How to Run the Evaluation

Once you have generated the results files for your sparse and dense retrievers, you can run the evaluation script as follows:

```bash
# First, make sure you have downloaded the dataset.
# Let's assume it's in a directory called 'datasets/scifact'.

# Evaluate the sparse retriever
python evaluation.py datasets/scifact results/sparse_results.json

# Evaluate the dense retriever
python evaluation.py datasets/scifact results/dense_results.json
```
The script will print the evaluation scores (nDCG@10, Recall@100, and MRR) to the console.

### Deliverable

Please submit your solution as a **single zip file**. The zip file should contain:

*   All your Python scripts (for downloading data, sparse retrieval, and dense retrieval).
*   Your `requirements.txt` file.
*   A `README.md` file that:
    *   Briefly explains your project structure.
    *   Provides clear, step-by-step instructions on how to set up the environment and run your code to generate the retrieval results.
    *   Includes a brief discussion of your results. Which retriever performed better? Why do you think that is? What are the performance trade-offs (e.g., speed, memory, retrieval quality) you observed?

Do **not** include the downloaded dataset in your zip file. Your instructions should be clear enough that we can download it ourselves. We will run your scripts and then use the provided `evaluation.py` to verify your results.
