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

def normalize_scores(scores):
    """Normalize scores to 0-1 range using min-max normalization"""
    if len(scores) == 0:
        return scores
    
    scores_array = np.array(list(scores.values()))
    
    # Avoid division by zero
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score - min_score == 0:
        return {k: 1.0 for k in scores.keys()}
    
    normalized = {}
    for doc_id, score in scores.items():
        normalized[doc_id] = (score - min_score) / (max_score - min_score)
    
    return normalized

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

def hybrid_fusion(bm25_results, dense_results, alpha=0.5):
    """
    Combine BM25 and dense retrieval scores
    
    Args:
        bm25_results: Dictionary of {doc_id: score} from BM25
        dense_results: Dictionary of {doc_id: score} from dense retrieval
        alpha: Weight for BM25 (0 to 1). Dense weight = 1 - alpha
               alpha=0.5 means equal weight
               alpha=0.7 means more weight to BM25
               alpha=0.3 means more weight to dense
    
    Returns:
        Dictionary of {doc_id: combined_score}
    """
    # Normalize scores to 0-1 range
    bm25_normalized = normalize_scores(bm25_results)
    dense_normalized = normalize_scores(dense_results)
    
    # Get all unique document IDs from both results
    all_doc_ids = set(bm25_normalized.keys()) | set(dense_normalized.keys())
    
    # Combine scores
    hybrid_scores = {}
    for doc_id in all_doc_ids:
        bm25_score = bm25_normalized.get(doc_id, 0.0)
        dense_score = dense_normalized.get(doc_id, 0.0)
        
        # Weighted combination
        hybrid_scores[doc_id] = alpha * bm25_score + (1 - alpha) * dense_score
    
    return hybrid_scores

def retrieve_hybrid(bm25, model, index, corpus_ids, queries, alpha=0.5, top_k=100):
    """
    Hybrid retrieval combining BM25 and dense retrieval
    
    Args:
        alpha: Weight for BM25 (0=only dense, 1=only BM25, 0.5=equal weight)
    """
    results = {}
    
    print(f"Retrieving with hybrid approach (alpha={alpha})...")
    for query_id, query_text in tqdm(queries.items()):
        # Get BM25 results
        bm25_results = retrieve_bm25(bm25, corpus_ids, query_text, top_k=top_k)
        
        # Get dense results
        dense_results = retrieve_dense(model, index, corpus_ids, query_text, top_k=top_k)
        
        # Combine with hybrid fusion
        hybrid_scores = hybrid_fusion(bm25_results, dense_results, alpha=alpha)
        
        # Sort and get top-k
        sorted_results = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
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
    
    # Try different alpha values
    alphas = [0.3, 0.5, 0.7]  # Different weights for BM25 vs Dense
    
    for alpha in alphas:
        print(f"\n{'='*60}")
        print(f"Running hybrid retrieval with alpha={alpha}")
        print(f"(BM25 weight: {alpha}, Dense weight: {1-alpha})")
        print(f"{'='*60}")
        
        # Retrieve
        results = retrieve_hybrid(
            bm25, model, index, corpus_ids, queries, 
            alpha=alpha, top_k=100
        )
        
        # Save results
        output_path = f"results/hybrid_results_alpha_{alpha}.json"
        save_results(results, output_path)
    
    print("\n" + "="*60)
    print("Hybrid retrieval complete!")
    print("="*60)
    print("\nNow evaluate each alpha value:")
    for alpha in alphas:
        print(f"python evaluation.py datasets/scifact results/hybrid_results_alpha_{alpha}.json")

if __name__ == "__main__":
    main()