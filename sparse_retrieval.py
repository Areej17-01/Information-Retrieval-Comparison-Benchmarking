from beir.datasets.data_loader import GenericDataLoader
from rank_bm25 import BM25Okapi
import json
import os
from tqdm import tqdm
from beir import util




def load_data(): 
    #downloading the dataset using beir

    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    
    data_path = util.download_and_unzip(url, "datasets")
    
    print(f"Dataset downloaded to: {data_path}")
    
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    
    return corpus, queries, qrels

def preprocess_corpus(corpus):
    
    #combine title and text
    corpus_ids = []
    corpus_texts = []
    
    for doc_id, doc in corpus.items():
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        
        tokenized = text.lower().split()
        
        corpus_ids.append(doc_id)
        corpus_texts.append(tokenized)
    
    return corpus_ids, corpus_texts


def retrieve_bm25(corpus_ids, corpus_texts, queries, top_k=100):
    #retrieve top 100 docs using BM25 
    
    print("Building BM25 index")
    bm25 = BM25Okapi(corpus_texts)
    
    results = {}
    
    print(f"Retrieving top-{top_k} documents")
    for query_id, query_text in tqdm(queries.items()):

        tokenized_query = query_text.lower().split()
        
        scores = bm25.get_scores(tokenized_query)
        top_k_values = scores.argsort()[-top_k:][::-1]

        

        results[query_id] = {}
        for idx in top_k_values:  
            doc_id = corpus_ids[idx]
            score = float(scores[idx])
            results[query_id][doc_id] = score
    
    return results

def save_results(results, output_path):
    #save results to json file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")


corpus, queries, qrels = load_data()

print(f"Loaded {len(corpus)} documents and {len(queries)} queries")

corpus_ids, corpus_texts = preprocess_corpus(corpus)

results = retrieve_bm25(corpus_ids, corpus_texts, queries, top_k=100)


# Saving results in results folder
output_path = "results/sparse_results.json"
save_results(results, output_path)

print("complete!")

