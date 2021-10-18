import json
import numpy as np

from rank_bm25 import BM25Okapi


class BM25Retrieval:
    def __init__(self, tokenize_fn, data_path):
        """
        bm25 모듈 생성
        """
        with open(data_path, "r") as json_data:
            wiki_data = json.load(json_data)

        wiki_data_keys = list(wiki_data.keys())
        self.contexts = [
            wiki_data[key]["text"] for key in wiki_data_keys
        ]  # 여기 keys값의 text값을 받도록하기
        self.tokenize_fn = tokenize_fn
        self.__get_sparse_embedding()

    def __get_sparse_embedding(self):
        """
        주어진 passage를 fit 해준다.
        """
        tokenized_contexts = list(map(self.tokenize_fn, self.contexts))
        self.bm25 = BM25Okapi(tokenized_contexts)

    def get_relevant_doc(self, query, k=1):
        """
        질문을 입력받고, k 만큼 랭킹을 가져온다.
        """
        tokenized_query = self.tokenize_fn(query)
        raw_doc_scores = self.bm25.get_scores(tokenized_query)

        doc_scores_index_desc = np.argsort(-raw_doc_scores)
        doc_scores = raw_doc_scores[doc_scores_index_desc]

        doc_list = self.bm25.get_top_n(tokenized_query, self.contexts, k)

        return doc_scores[:k], doc_list
