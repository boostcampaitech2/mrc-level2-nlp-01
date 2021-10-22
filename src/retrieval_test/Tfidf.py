from typing import List, Tuple, NoReturn, Any, Optional, Union
import pandas as pd
import json
from sklearn.feature_extraction.text import TfidfVectorizer

class SparseRetrieval:
    def __init__(self, corpus_data_path, tokenizer):
        self.wiki_path = corpus_data_path
        
        self.tokenizer = tokenizer #tokeinzer 지정함
        with open(self.wiki_path) as data_file:
            local = json.load(data_file)
        
        wiki_corpus =[]
        for i in range(len(local)):
            wiki_corpus.append(local[str(i)])
        
        self.wiki_data = pd.DataFrame(wiki_corpus) #DataFrame 형태로 저장된 wiki_data
        
        self.vectorizer = TfidfVectorizer(tokenizer=self.tokenizer, 
                                     ngram_range=(1,2))
        
        self.sp_matrix = self.get_sp_matrix() #SparseRetrival 객체 생성시 sp_matrix 구함
    
    def get_sp_matrix(self):
        
        self.vectorizer.fit(self.wiki_data['text'].values)
        sp_matrix = self.vectorizer.transform(self.wiki_data['text'].values)
        
        print('passage의 matrix는 {}'.format(sp_matrix.shape))
        
        return sp_matrix
    
    
    def get_relevant_doc(query, top_k): #query : str #dataset['validation']
        #하나의 query가 들어오면 계산하도록 함.
        
        query_vec = self.vectorizer.transform([query]) 
        print('query의 matirx는 {}'.format(query_vec.shape))
        
        result = query_vec * self.sp_matrix.T
        print('결과 matrix는 {}'.format(result.shape))
        
        sorted_result = np.argsort(-result.data)
        doc_scores = result.data[sorted_result]
        doc_ids = result.indices[sorted_result]
        
        result_docscores, result_docids = doc_scores[:top_k], doc_ids[:top_k]

        return result_docscores, result_docids