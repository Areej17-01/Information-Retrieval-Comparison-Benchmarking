from beir.datasets.data_loader import GenericDataLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from tqdm import tqdm

def load_data(data_path):

    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels

def prepare_corpus(corpus):

    corpus_ids = []
    corpus_texts = []
    
    for doc_id, doc in corpus.items():

        text = doc.get('title', '') + ' ' + doc.get('text', '')
        corpus_ids.append(doc_id)
        corpus_texts.append(text)
    
    return corpus_ids, corpus_texts

def embed_corpus(model, corpus_texts, batch_size=32):

    print("Embedding corpus")
    embeddings = model.encode(
        corpus_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    return embeddings

def build_faiss_index(embeddings):

    print("Building FAISS index")
    
    dimension = embeddings.shape[1]
    
    index = faiss.IndexFlatIP(dimension)
    
    faiss.normalize_L2(embeddings)
    
    index.add(embeddings)
    
    print(f"FAISS index built with {index.ntotal} vectors")
    return index

def retrieve_dense(model, index, corpus_ids, queries, top_k=100):
    """Retrieve top-k documents for each query using dense retrieval"""
    
    results = {}
    
    print(f"Retrieving top-{top_k} documents for each query...")
    for query_id, query_text in tqdm(queries.items()):
        query_embedding = model.encode([query_text], convert_to_numpy=True)
        
        faiss.normalize_L2(query_embedding)
        
        scores, indices = index.search(query_embedding, top_k)
        
        results[query_id] = {}
        for score, idx in zip(scores[0], indices[0]):
            doc_id = corpus_ids[idx]
            results[query_id][doc_id] = float(score)
    
    return results

def save_results(results, output_path):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")

def main():
    data_path = "datasets/scifact"
    corpus, queries, qrels = load_data(data_path)
    
    print(f"Loaded {len(corpus)} documents and {len(queries)} queries")
    
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    
    corpus_ids, corpus_texts = prepare_corpus(corpus)
    
    corpus_embeddings = embed_corpus(model, corpus_texts)
    
    # Build FAISS index
    index = build_faiss_index(corpus_embeddings)
    
    # Retrieve
    results = retrieve_dense(model, index, corpus_ids, queries, top_k=100)
    
    # Save results
    output_path = "results/dense_results.json"
    save_results(results, output_path)
    
    print("Dense retrieval complete!")

if __name__ == "__main__":
    main()