import argparse
import os
from pathlib import Path
from query import Query
from similarity import concept_tokenize,cosine_similarity_sparse
from vsm_model import VSM
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
    top_k = 10
    for topic in q.topics[0:1]:
        q_vec = concept_tokenize(topic['concepts'],model.term_to_index, model.idf_vector)
        similarities = cosine_similarity_sparse(q_vec,model.tfidf_matrix)
        top_indices = similarities.argsort()[::-1][:top_k]
        for rank, idx in enumerate(top_indices, start=1):
            sim_score = similarities[idx]
            if sim_score == 0:
                continue  
            print(f"{rank}. 檔案名稱：{model.file_names[idx]}，相似度：{sim_score:.4f}")


if __name__=="__main__":
    main()