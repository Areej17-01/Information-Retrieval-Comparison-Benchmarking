from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from tqdm import tqdm
from beir import util

def load_data(data_path):
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels


def preprocess_corpus(corpus):
    """Prepare corpus for both BM25 and dense retrieval"""
    corpus_ids = []
    corpus_texts_raw = []
    corpus_texts_tokenized = []
    
    for doc_id, doc in corpus.items():
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        
        corpus_ids.append(doc_id)
        corpus_texts_raw.append(text)  # For dense retrieval
        corpus_texts_tokenized.append(text.lower().split())  # For BM25
    
    return corpus_ids, corpus_texts_raw, corpus_texts_tokenized

def build_bm25_index(corpus_texts_tokenized):
    """Build BM25 index"""
    print("Building BM25 index...")
    bm25 = BM25Okapi(corpus_texts_tokenized)
    return bm25

def build_dense_index(model, corpus_texts_raw):
    """Build FAISS index for dense retrieval"""
    print("Embedding corpus for dense retrieval...")
    embeddings = model.encode(
        corpus_texts_raw,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    return index, embeddings

def retrieve_bm25(bm25, corpus_ids, query_text, top_k=100):
    """Retrieve using BM25"""
    tokenized_query = query_text.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Get top-k indices
    top_k_indices = scores.argsort()[-top_k:][::-1]
    
    # Create results dictionary
    results = {}
    for idx in top_k_indices:
        doc_id = corpus_ids[idx]
        score = float(scores[idx])
        results[doc_id] = score
    
    return results

def retrieve_dense(model, index, corpus_ids, query_text, top_k=100):
    """Retrieve using dense retrieval"""
    # Embed query
    query_embedding = model.encode([query_text], convert_to_numpy=True)
    
    # Normalize
    faiss.normalize_L2(query_embedding)
    
    # Search
    scores, indices = index.search(query_embedding, top_k)
    
    # Create results dictionary
    results = {}
    for score, idx in zip(scores[0], indices[0]):
        doc_id = corpus_ids[idx]
        results[doc_id] = float(score)
    
    return results

def reciprocal_rank_fusion(bm25_results, dense_results, k=60):
    """
    Reciprocal Rank Fusion (RRF) - combines rankings from multiple retrievers
    
    Formula: RRF_score(doc) = Σ [1 / (k + rank(doc))]
    
    Args:
        bm25_results: Dictionary of {doc_id: score} from BM25
        dense_results: Dictionary of {doc_id: score} from dense retrieval
        k: Constant for RRF (default=60, recommended in literature)
           Higher k = less emphasis on top ranks
           Lower k = more emphasis on top ranks
    
    Returns:
        Dictionary of {doc_id: rrf_score}
    """
    # Sort both result sets by score to get rankings
    bm25_ranked = sorted(bm25_results.items(), key=lambda x: x[1], reverse=True)
    dense_ranked = sorted(dense_results.items(), key=lambda x: x[1], reverse=True)
    
    # Calculate RRF scores
    rrf_scores = {}
    
    # Add BM25 ranks
    # rank starts at 1 (not 0) as per RRF formula
    for rank, (doc_id, _) in enumerate(bm25_ranked, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    # Add dense ranks
    for rank, (doc_id, _) in enumerate(dense_ranked, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    
    return rrf_scores

def retrieve_hybrid_rrf(bm25, model, index, corpus_ids, queries, k=60, top_k=100):
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF)
    
    Args:
        k: RRF constant (default=60)
           - k=60: Standard value (recommended)
           - k=10: More emphasis on top-ranked docs
           - k=100: Less emphasis on top-ranked docs
        top_k: Number of documents to retrieve
    """
    results = {}
    
    print(f"Retrieving with RRF (k={k})...")
    for query_id, query_text in tqdm(queries.items()):
        # Get BM25 results
        bm25_results = retrieve_bm25(bm25, corpus_ids, query_text, top_k=top_k)
        
        # Get dense results
        dense_results = retrieve_dense(model, index, corpus_ids, query_text, top_k=top_k)
        
        # Combine with RRF
        rrf_scores = reciprocal_rank_fusion(bm25_results, dense_results, k=k)
        
        # Sort by RRF score and get top-k
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # Store results
        results[query_id] = {doc_id: score for doc_id, score in sorted_results}
    
    return results

def save_results(results, output_path):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main():
    # Load data
    data_path = "datasets/scifact"
    corpus, queries, qrels = load_data(data_path)
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries")
    
    # Preprocess corpus
    print("Preprocessing corpus...")
    corpus_ids, corpus_texts_raw, corpus_texts_tokenized = preprocess_corpus(corpus)
    
    # Build BM25 index
    bm25 = build_bm25_index(corpus_texts_tokenized)
    
    # Load sentence transformer model
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Build dense index
    index, embeddings = build_dense_index(model, corpus_texts_raw)
    
    # Try different k values for RRF
    k_values = [10, 60, 100]  # Different RRF constants
    
    for k in k_values:
        print(f"\n{'='*60}")
        print(f"Running hybrid retrieval with RRF (k={k})")
        print(f"RRF Formula: score = 1/(k + rank)")
        print(f"{'='*60}")
        
        # Retrieve using RRF
        results = retrieve_hybrid_rrf(
            bm25, model, index, corpus_ids, queries, 
            k=k, top_k=100
        )
        
        # Save results
        output_path = f"results/hybrid_rrf_k_{k}.json"
        save_results(results, output_path)
    
    print("\n" + "="*60)
    print("Hybrid retrieval with RRF complete!")
    print("="*60)
    print("\nNow evaluate each k value:")
    for k in k_values:
        print(f"python evaluation.py datasets/scifact results/hybrid_rrf_k_{k}.json")

if __name__ == "__main__":
    main()