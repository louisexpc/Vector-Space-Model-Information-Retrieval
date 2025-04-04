import numpy as np
from collections import defaultdict
from scipy.sparse import lil_matrix, csr_matrix, save_npz
def query_vector_transformation(query_input_list:list,term_to_index:dict,idf : np.array):
    """
    將query input list 內的keyword 做tokenized (unigram + bigram )，建立出 term_freq 後計算出 query_vector

    Parameters:
    - query_input_list: list[str]，處理過後的key word list
    - term_to_index: dict, 將query_input_list 轉換為 term_freq所需
    - idf: np.array，smoothed idf，將 term_freq 轉換為 query vector 所需

    Return:
    - q_vec : lil_matrix，稀疏矩陣
    """
    term_freq=defaultdict(int)
    for keyword in query_input_list:
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

def compute_cosine_similarity(q_vec, tfidf_matrix):
    """
    計算query vector 與 docs 的 cosine similarity

    Parameters:
    - q_vec: lil_matrix, (1 x vacb_size)
    - tfidf_matrix: lil_matrix, (docs_size x vacb_size)

    Return:
    - similarities: np.array
    """
    dot_products = tfidf_matrix.dot(q_vec.T).toarray().flatten()

    q_norm = np.sqrt(q_vec.multiply(q_vec).sum())
    doc_norms = np.sqrt(tfidf_matrix.multiply(tfidf_matrix).sum(axis=1)).A1 
    epsilon = 1e-10
    similarities = dot_products / (doc_norms * q_norm + epsilon)

    return similarities

def query_term_indices_transformation(query_input_list:list , term_to_index:dict):
    """
    將query input list 內的keyword 做 term to index 的mapping，轉換為 term indices 格式
    - 提取 unigram and bigram
    - 轉換成 qurey term to index
    
    Parameters:
    - query_input_list: list[str]，處理過後的key word list
    - term_to_index: dict, 將query_input_list 轉換為 term_freq所需

    Returns:
    - query_term_indices: list[int], query term 對應之 index
    """
    query_terms = []
    for keyword in query_input_list:
        # unigram
        query_terms.extend(keyword)  # 等同於逐字加入
        # bigram
        query_terms.extend([keyword[i] + "_" + keyword[i+1] for i in range(len(keyword) - 1)])

    
    #query_terms = list(set(query_terms))

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
    - bm25_score : np.array , (1 x doc_size(N))
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

    

if __name__=="__main__":
    compute_BM25()