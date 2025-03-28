import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix, save_npz
def concept_tokenize(concept:list,term_to_index:dict,idf : np.array):
    """
    將 Query 中的concept 做tokenized 建立出 query vector
    """
    term_freq=defaultdict(int)
    for keyword in concept:
        #unigram
        for i in range(len(keyword)):
            unigram =  keyword[i]
            if unigram in term_to_index:
                term_freq[term_to_index[unigram]] +=1
        
        #bigram
        for i in range(len(keyword)-1):
            bigram = keyword[i] +"_"+keyword[i+1]
            if bigram in term_to_index:
                term_freq[term_to_index[bigram]]+=1

    q_vec = lil_matrix((1,len(idf)), dtype=np.float32)
    for idx,freq in term_freq.items():
        q_vec[0,idx] = idf[idx] * freq
    return q_vec

def cosine_similarity_sparse(q_vec, tfidf_matrix):
    """
    計算查詢向量與每份文件的 cosine similarity（支援 sparse）
    
    q_vec: 1 x vocab_size 稀疏向量
    tfidf_matrix: N x vocab_size 稀疏矩陣
    回傳：長度為 N 的 numpy array（每筆相似度）
    """
    # 1. 計算內積（查詢向量 ⋅ 每份文件）
    dot_products = tfidf_matrix.dot(q_vec.T).toarray().flatten()  # shape: (N,)

    # 2. 計算查詢向量的 L2 norm
    q_norm = np.sqrt(q_vec.multiply(q_vec).sum())

    # 3. 計算每份文件向量的 L2 norm（axis=1）
    doc_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1  # shape: (N,)

    # 4. 避免除以 0（加一個 epsilon）
    epsilon = 1e-10
    similarities = dot_products / (doc_norms * q_norm + epsilon)

    return similarities
