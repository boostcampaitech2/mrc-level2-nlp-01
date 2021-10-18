import json
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfRetrieval:
    def __init__(self, tokenize_fn, data_path):

        """
        데이터를 불러오고 토크나이즈 작업을 해준다.
        """
        with open(data_path, "r") as json_data:
            wiki_data = json.load(json_data)

        wiki_data_keys = list(wiki_data.keys())
        self.contexts = [
            wiki_data[key]["text"] for key in wiki_data_keys
        ]  # 여기 keys값의 text값을 받도록하기
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=100000,
        )
        self.__get_sparse_embedding()

    def __get_sparse_embedding(self):

        """
        주어진 passage를 fit 해준다.
        """
        self.vectorizer.fit(self.contexts)
        self.sp_matrix = self.vectorizer.transform(self.contexts)

    def get_relevant_doc(self, query, k=1):

        """
        질문을 입력받고, k 만큼 랭킹을 가져온다.
        """
        query_vec = self.vectorizer.transform([query])
        result = query_vec * self.sp_matrix.T
        sorted_result = np.argsort(-result.data)
        doc_scores = result.data[sorted_result]
        doc_ids = result.indices[sorted_result]
        return doc_scores[:k], doc_ids[:k]
