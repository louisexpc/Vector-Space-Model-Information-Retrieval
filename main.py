import argparse
import os
from pathlib import Path
from query import Query
from similarity import concept_tokenize,cosine_similarity_sparse,tokenize,compute_BM25
from vsm_model import VSM
from scipy.sparse import lil_matrix, csr_matrix
import pandas as pd
import numpy as np
from collections import Counter

def generate_ranked_list(query_ids: list, retrieved_docs: list, path: str):
    if len(query_ids) != len(retrieved_docs):
        print(f"The length of query_ids and retrieved_docs doesn't match, output file failed")
        return
    
    df = pd.DataFrame({"query_id": query_ids, "retrieved_docs": retrieved_docs})
    
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise ValueError(f"Can't save ranked file at {path}: {e}")

def retrieve(score:np.array,top_k:int, file_names)->list:
    """
    based on unsorted score to genereate retrieved_docs string
    Parameter:
    - score: np.array, 

    Return:
    - 
    """
    top_indices = score.argsort()[::-1][:top_k]
    docs_string =""
    for rank, idx in enumerate(top_indices, start=1):
        sim_score = score[idx]
        if sim_score == 0:
            continue 
        file_name =  file_names[idx].split("/")[-1].lower()
        print(f"{rank}. 檔案名稱：{file_name}，相似度：{sim_score:.4f}")
        docs_string+= (file_name+" ")
    return docs_string

def pseudo_rocchio_feedback(q_vec , tfidf_matrix, alpha=1.0, beta=0.75, top_k=10):
    """
    Update query vector by rocchio feedback algorithm (suppose gamma is 0 since unrrelevent docs is unknown)

    Parameters:
    - q_vec : lil_matrix, original query vector
    - tfidf_matrix
    - alpah, beta: parameters of feedback
    - top_k: number of relevent docs

    Return:
    - new_q_vec : lil_matrix (1 x vocab_size), updated query vector

    """
    tfidf_matrix = tfidf_matrix.tocsr()
    similarities = cosine_similarity_sparse(q_vec,tfidf_matrix)

    top_k_indices = similarities.argsort()[::-1][:top_k]
    centroid = tfidf_matrix[top_k_indices].mean(axis = 0)

    # 維持輸出 data type lil_matrix
    centroid_lil = lil_matrix(centroid)
    new_q_vec = q_vec.copy()
    new_q_vec *= alpha
    new_q_vec += beta * centroid_lil

    return new_q_vec

def pseudo_bm25_query_extension(query_term_indices: list, idf: np.array, docs_length: list, doc_term_freqs: list, K1: float, b: float,top_k:int = 10,expand_n:int=5):
    bm25_score = compute_BM25(query_term_indices,idf,docs_length,doc_term_freqs,K1,b)
    top_k_indices = bm25_score.argsort()[::-1][:top_k]

    term_counter = Counter()
    for doc_id in top_k_indices:
        term_counter.update(doc_term_freqs[doc_id])
    
    original_set = set(query_term_indices)

    #counter: {term_index , term_frequency}
    scored_terms = {
        t:tf *idf[t] for t,tf in term_counter.items() if t not in original_set
    }
    expansion_terms = sorted(scored_terms, key=scored_terms.get, reverse=True)[:expand_n]
    expanded_query_terms = query_term_indices + expansion_terms

    return expanded_query_terms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",action="store_true",help="Enable relevance feedback")
    parser.add_argument("-i",required=True,help="Input query file")
    parser.add_argument("-o",required=True,help="Output ranked list file")
    parser.add_argument("-m",required=True,help="Model directory path")
    parser.add_argument("-d",required=True,help="NTCIR document directory")
    parser.add_argument("-b",action="store_true",help="Change score as BM25 not similarity")

    args = parser.parse_args()
    
    model = VSM(args.m)
    q = Query(args.i)
    top_k = 100
    K1 = 1.2
    b = 0.75
    retrieved_docs = []
    for topic in q.topics:
        if not args.b:
            """Similarity"""
            print("Using Similarity")
            q_vec = concept_tokenize(topic['concepts'],model.term_to_index, model.idf_vector)
            if args.r:
                new_q_vec = pseudo_rocchio_feedback(q_vec,model.tfidf_matrix)
            else:
                new_q_vec = q_vec
            similarities = cosine_similarity_sparse(new_q_vec,model.tfidf_matrix)
            docs_string = retrieve(similarities,top_k,model.file_names)
        else:
            """BM25"""
            print("Using BM25")
            query_term_indices = tokenize(topic['concepts'],model.term_to_index)
            if args.r:
                new_query_term_indices = pseudo_bm25_query_extension(query_term_indices,model.idf_vector,model.docs_length,model.doc_term_freqs,K1,b)
            else:
                new_query_term_indices=query_term_indices
            bm25_score = compute_BM25(new_query_term_indices,model.idf_vector,model.docs_length,model.doc_term_freqs,K1,b)
            docs_string = retrieve(bm25_score,top_k,model.file_names)

        retrieved_docs.append(docs_string.strip())

    generate_ranked_list(q.query_ids,retrieved_docs,args.o)


if __name__=="__main__":
    main()