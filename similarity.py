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

def tokenize(concepts:list , term_to_index:dict):
    """
    將 concept 的 key word 做 term to index 的mapping
    Steps: For concept key words :[流浪狗、流浪犬、動物保護、動保法、保育、人道]
    - 提取 unigram and bigram，並去除重複的gram
    - 轉換成 qurey term to index

    Returns:
        List[int] - 查詢詞對應的 index 列表
    """
    query_terms = []
    for keyword in concepts:
        # unigram
        query_terms.extend(keyword)  # 等同於逐字加入
        # bigram
        query_terms.extend([keyword[i] + "_" + keyword[i+1] for i in range(len(keyword) - 1)])

    
    query_terms = list(set(query_terms))

    query_term_indices = [term_to_index[w] for w in query_terms if w in term_to_index]
    return query_term_indices
    



def compute_BM25(query_term_indices: list, idf: np.array, docs_length: list, doc_term_freqs: list, K1: float, b: float) -> np.array:
    """
    Parameters:
    - query_term_indices: 查詢詞對應的 index 列表
    - idf: smooth document frequency (term_i 總共出現在多少個 docs)
    - docs_length: the lengths of each doc
    - doc_term_freqs: list(dict)
    - k1: BM25 參數
    - b: BM25 參數

    Return:
    - bm25_score
    """
    N = len(docs_length)
    avgDL = sum(docs_length) / N
    scores = np.zeros(N, dtype=np.float32)

    for doc_id in range(N):
        doc_len = docs_length[doc_id]
        term_freq_dict = doc_term_freqs[doc_id]
        for term_idx in query_term_indices:
            f = term_freq_dict.get(term_idx, 0)
            if f == 0:
                continue
            const = K1 * (1 - b + b * doc_len / avgDL)
            score = idf[term_idx] * f * (K1 + 1) / (f + const)
            scores[doc_id] += score

    return scores

    pass

if __name__=="__main__":
    compute_BM25()