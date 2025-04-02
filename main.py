import argparse
import os
from pathlib import Path
from query import Query
from similarity import concept_tokenize,cosine_similarity_sparse,tokenize,compute_BM25
from vsm_model import VSM
import pandas as pd
import numpy as np

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
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r",action="store_true",help="Enable relevance feedback")
    parser.add_argument("-i",required=True,help="Input query file")
    parser.add_argument("-o",required=True,help="Output ranked list file")
    parser.add_argument("-m",required=True,help="Model directory path")
    parser.add_argument("-d",required=True,help="NTCIR document directory")

    args = parser.parse_args()
    
    model = VSM(args.m)
    q = Query(args.i)
    top_k = 50
    K1 = 1.2
    b = 0.75
    retrieved_docs = []
    for topic in q.topics:
        """Similarity"""
        #q_vec = concept_tokenize(topic['concepts'],model.term_to_index, model.idf_vector)
        #similarities = cosine_similarity_sparse(q_vec,model.tfidf_matrix)

        """BM25"""
        query_term_indices = tokenize(topic['concepts'],model.term_to_index)
        bm25_score = compute_BM25(query_term_indices,model.idf_vector,model.docs_length,model.doc_term_freqs,K1,b)
        docs_string = retrieve(bm25_score,top_k,model.file_names)
        retrieved_docs.append(docs_string.strip())

    generate_ranked_list(q.query_ids,retrieved_docs,args.o)


if __name__=="__main__":
    main()