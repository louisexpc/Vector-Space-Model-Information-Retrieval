import argparse
import os
from pathlib import Path
from query import Query
from similarity import concept_tokenize,cosine_similarity_sparse
from vsm_model import VSM
import pandas as pd

def generate_ranked_list(query_ids: list, retrieved_docs: list, path: str):
    if len(query_ids) != len(retrieved_docs):
        print(f"The length of query_ids and retrieved_docs doesn't match, output file failed")
        return
    
    df = pd.DataFrame({"query_id": query_ids, "retrieved_docs": retrieved_docs})
    
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        raise ValueError(f"Can't save ranked file at {path}: {e}")

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

    retrieved_docs = []
    for topic in q.topics:
        q_vec = concept_tokenize(topic['concepts'],model.term_to_index, model.idf_vector)
        similarities = cosine_similarity_sparse(q_vec,model.tfidf_matrix)
        top_indices = similarities.argsort()[::-1][:top_k]
        """
        feedback
        """
        docs_string =""
        for rank, idx in enumerate(top_indices, start=1):
            sim_score = similarities[idx]
            if sim_score == 0:
                continue 
            file_name =  model.file_names[idx].split("/")[-1].lower()
            print(f"{rank}. 檔案名稱：{file_name}，相似度：{sim_score:.4f}")
            docs_string+= (file_name+" ")
        retrieved_docs.append(docs_string.strip())

    generate_ranked_list(q.query_ids,retrieved_docs,args.o)


if __name__=="__main__":
    main()