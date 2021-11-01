import os
import time

# import faiss
import pickle
import numpy as np
import pandas as pd

from rank_bm25 import BM25Plus, BM25Okapi, BM25L
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Optional, Union
from datasets import Dataset, load_from_disk


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25SparseRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "/opt/ml/data/",
        context_path: Optional[str] = "wiki_preprocessed_droped",
        **kwargs,
    ) -> NoReturn:
        self.data_path = data_path
        wiki_datasets = load_from_disk(os.path.join(data_path, context_path))

        self.contexts = [wiki["text"] for wiki in wiki_datasets]
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.bm25 = None

    def tokenizer_fn(self, context) -> List[str]:
        raise "토크나이저를 설정해 주세요"

    def get_sparse_embedding(
        self,
        pickle_name="bm25.bin",
        type="Plus",
        k1=1.6,
        b=0.3,
        ep=0.25,
        delta=0.7,
        **kwargs,
    ) -> NoReturn:
        # 논문기준 가장 큰값을 기본값으로 사용 http://www.cs.otago.ac.nz/homepages/andrew/papers/2014-2.pdf

        pickle_name = f"{k1}_{b}_{ep}_{delta}_{type}_{pickle_name}"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            if type not in ["Okapi", "Plus", "L"]:
                raise "올바른 type을 입력해주세요. Okapi | Plus | L"
            print("Build passage embedding")
            tokenized_contexts = list(map(self.tokenizer_fn, tqdm(self.contexts)))
            if type == "Okapi":
                self.bm25 = BM25Okapi(tokenized_contexts, k1=k1, b=b, epsilon=ep)
            elif type == "Plus":
                self.bm25 = BM25Plus(tokenized_contexts, k1=k1, b=b, delta=delta)
            else:
                self.bm25 = BM25L(tokenized_contexts, k1=k1, b=b, delta=delta)
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        assert self.bm25 is not None, "get_sparse_embedding() 메소드를 먼저 수행해줘야합니다."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print(f"Top-{i+1} passage with score {doc_scores[i]:4f}")
                print(self.contexts[doc_indices[i]])

            return (doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)])

        elif isinstance(query_or_dataset, Dataset):

            # Retrieve한 Passage를 pd.DataFrame으로 반환합니다.
            total = []
            doc_scores, doc_indices = [], []
            with timer("query exhaustive search"):
                for question in tqdm(query_or_dataset["question"]):
                    doc_score, doc_indice = self.get_relevant_doc(question, k=topk)
                    doc_scores.append(doc_score)
                    doc_indices.append(doc_indice)
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    # Query와 해당 id를 반환합니다.
                    "question": example["question"],
                    "id": example["id"],
                    # Retrieve한 Passage의 id, context를 반환합니다.
                    "context_id": doc_indices[idx],
                    "context": " ".join(doc_indices[idx]),
                }
                if "context" in example.keys() and "answers" in example.keys():
                    # validation 데이터를 사용하면 ground_truth context와 answer도 반환합니다.
                    tmp["original_context"] = example["context"]
                    tmp["answers"] = example["answers"]
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:

        """
        Arguments:
            query (str):
                하나의 Query를 받습니다.
            k (Optional[int]): 1
                상위 몇 개의 Passage를 반환할지 정합니다.
        Note:
            vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        tokenized_query = self.tokenizer_fn(query)
        raw_doc_scores = self.bm25.get_scores(tokenized_query)

        doc_scores_index_desc = np.argsort(-raw_doc_scores)
        doc_scores = raw_doc_scores[doc_scores_index_desc]

        doc_list = self.bm25.get_top_n(tokenized_query, self.contexts, k)

        return doc_scores[:k], doc_list
