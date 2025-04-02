import numpy as np
import os
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
import pickle

class VSM:
    _instance = None

    def __new__(cls,model_dir):
        """
        Parameters
        - model_dir: args.m 
        """
        if cls._instance is None:
            cls._instance = super(VSM,cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self,model_dir):
        if self._initialized:
            return 
        #Svae model path
        self.model_dir = model_dir
        self.vocab_file = os.path.join(model_dir, "vocab.all")
        self.file_list_file = os.path.join(model_dir, "file-list")
        self.inverted_file = os.path.join(model_dir, "inverted-file")

        #Save output file path
        self.output_files_file = os.path.join(model_dir,"files_name.npy")
        self.output_tf_idf_file = os.path.join(model_dir,"tf_idf.npz")
        self.output_term_to_index_file = os.path.join(model_dir,"term_to_index.pkl")
        self.output_idf_vector_file = os.path.join(model_dir,"idf_vector.npy")
        self.output_docs_length = os.path.join(model_dir,"docs_lengths.npy")
        self.output_df = os.path.join(model_dir,"df.npy")
        self.output_doc_term_freqs = os.path.join(model_dir,"doc_term_freq.pkl")

        #Validation model files
        if not os.path.exists(self.inverted_file):
            raise ValueError(f"Inverted file 不存在，請檢查文件路徑是否在 {self.inverted_file}")
        if not os.path.exists(self.vocab_file):
            raise ValueError(f"Vocab file 不存在，請檢查文件路徑是否在 {self.vocab_file}")
        if not os.path.exists(self.file_list_file):
            raise ValueError(f"File list 不存在，請檢查文件路徑是否在 {self.file_list_file}")
        
        #Validation Output files

        if not os.path.exists(self.output_files_file) or \
            not os.path.exists(self.output_tf_idf_file) or \
            not os.path.exists(self.output_term_to_index_file) or \
            not os.path.exists(self.output_idf_vector_file) or \
            not os.path.exists(self.output_docs_length) or \
            not os.path.exists(self.output_df) or \
            not os.path.exists(self.output_doc_term_freqs):
            #Renerate necessacry files
            self.term_to_index,\
            self.idf_vector,\
            self.file_names,\
            self.tfidf_matrix,\
            self.df, \
            self.docs_length, \
            self.doc_term_freqs \
            =self._load_model()
        
        else:
            print(f"files exist, loading")
            self.term_to_index = self._load_pickle(self.output_term_to_index_file)
            self.idf_vector = self._load_npy(self.output_idf_vector_file)
            self.file_names = self._load_npy(self.output_files_file)
            self.tfidf_matrix = self._load_npz(self.output_tf_idf_file)
            self.df = self._load_npy(self.output_df)
            self.docs_length = self._load_npy(self.output_docs_length)
            self.doc_term_freqs = self._load_pickle(self.output_doc_term_freqs)

        self._initialized = True
    def _load_pickle(self,path):
        try:
            with open(path,'rb') as f:
                target = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Loading pkl failed: {e}")
        return target
    
    def _load_npy(self, path):
        try:
            target =  np.load(path)
        except Exception as e:
            raise ValueError(f"Loading npy failed: {e}")
        return target

    def _load_npz(self,path):
        try:
            target =  load_npz(path)
        except Exception as e:
            raise ValueError(f"Loading npz failed: {e}")
        return target
    
    def _load_model(self):
        
        """Compute Term Freqency"""
        #read vacab
        with open(self.vocab_file,'r',encoding='UTF-8') as file:
            vocab = [line.strip() for line in file.readlines()]
        vocab_size = len(vocab)

        #read file-list
        with open(self.file_list_file, 'r',encoding='UTF-8') as file:
            file_list = [line.strip() for line in file.readlines()]
        file_list_size = len(file_list)
        
        #read inverted file
        with open(self.inverted_file,'r',encoding='UTF-8') as file:
            lines = file.readlines()

        term_to_index = {}
        doc_term_freqs = [{} for _ in range(file_list_size)] 

        index = 0
        while index < len(lines):
            title = lines[index].strip()
            if not title:
                index+=1
                continue

            """
            - format: 
            vocab_id_1 vocab_id_2 N
            - hint: 
            1. bigram when vocab_id_2 != -1
            """
            v_1,v_2,n = map(int,title.split()) 
        
            #classify unigram or bigram
            if v_2 != -1: #bigram
                term = vocab[v_1]+ "_" + vocab[v_2]
            else:
                term = vocab[v_1]
            
            # indexing the term
            if term not in term_to_index:
                term_idx = len(term_to_index)
                term_to_index[term] = term_idx
                
            
            term_idx = term_to_index[term]

            #record file id
            for j in range(1,n+1):
                file_id,count = map(int,lines[index+j].strip().split())
                doc_term_freqs[file_id][term_idx] = count
            
            #jump to next title
            index += n+1
        
        try:
            with open(self.output_term_to_index_file, "wb") as f:
                pickle.dump(term_to_index, f)
            np.save(self.output_files_file,np.array(file_list))
        except Exception as e:
            print(f"Save file error: {e}")

        """Compute TF-IDF Matrix"""
        doc_count = len(doc_term_freqs)
        term_size = len(term_to_index)
        tf_matrix = lil_matrix((doc_count, term_size), dtype=np.float32)

        #create tf_matrix
        for doc_id, term_freqs in enumerate(doc_term_freqs):
            for term_idx,count in term_freqs.items():
                tf_matrix[doc_id, term_idx] = count

        #compute df(document freqency): how many freqs have the term showing in the docs(axis = 0)
        df = (tf_matrix > 0).sum(axis=0).A1  
        #compute idf with smoothing
        idf = np.log((doc_count + 1) / (df + 1)) + 1
        tf_idf_matrix = tf_matrix.multiply(idf)
        try:
            save_npz(self.output_tf_idf_file, tf_idf_matrix)
            np.save(self.output_idf_vector_file, idf)
        except Exception as e:
            print(f"Save file error: {e}")
        
        """add: compute docs_length(BM25)"""
        doc_lengths = [sum(term_freqs.values()) for term_freqs in doc_term_freqs]
        try:
            np.save(self.output_docs_length,np.array(doc_lengths))
            np.save(self.output_df,np.array(df))
            with open(self.output_doc_term_freqs, "wb") as f:
                pickle.dump(doc_term_freqs, f)
        except Exception as e:
            print(f"Save file error: {e}")
        return term_to_index,idf,file_list,tf_idf_matrix,df,doc_lengths,doc_term_freqs

if __name__=="__main__":
    vsm = VSM("model")
    print(len(vsm.term_to_index))