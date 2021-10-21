# import faiss
import os
import json
import time
import re
import pickle
import numpy as np
import pandas as pd

from konlpy.tag import Komoran, Okt
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm
from contextlib import contextmanager
from typing import List, Tuple, NoReturn, Any, Optional, Union


from datasets import Dataset


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BM25SparseRetrieval:
    def __init__(
        self,
        tokenize_fn,
        data_path: Optional[str] = "../data/",
        context_path: Optional[str] = "wikipedia_tag_docs.json",
        question_tokenize_fn=None,
    ) -> NoReturn:

        """
        Arguments:
            tokenize_fn:
                기본 text를 tokenize해주는 함수입니다.
                아래와 같은 함수들을 사용할 수 있습니다.
                - lambda x: x.split(' ')
                - Huggingface Tokenizer
                - konlpy.tag의 Mecab

            data_path:
                데이터가 보관되어 있는 경로입니다.

            context_path:
                Passage들이 묶여있는 파일명입니다.

            data_path/context_path가 존재해야합니다.

        Summary:
            Passage 파일을 불러오고 TfidfVectorizer를 선언하는 기능을 합니다.
        """

        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r", encoding="utf-8") as f:
            wiki = json.load(f)

        self.contexts = [v["text"] for v in wiki.values()]
        self.okt = [v["okt"] for v in wiki.values()]
        self.komoran = [v["komoran"] for v in wiki.values()]
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        self.tokenize_fn = tokenize_fn
        self.question_tokenize_fn = (
            question_tokenize_fn if question_tokenize_fn is not None else tokenize_fn
        )
        self.bm25 = None
        self.indexer = None  # build_faiss()로 생성합니다.

    def get_sparse_embedding(self, pickle_name="bm25api.bin") -> NoReturn:

        """
        Summary:
            Passage Embedding을 만들고
            TFIDF와 Embedding을 pickle로 저장합니다.
            만약 미리 저장된 파일이 있으면 저장된 pickle을 불러옵니다.
        """

        # Pickle을 저장합니다.
        pickle_name = pickle_name
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            tokenized_contexts = list(map(self.tokenize_fn, self.komoran, self.okt))
            self.bm25 = BM25Okapi(tokenized_contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve(
        self, query_or_dataset: Union[str, Dataset], topk: Optional[int] = 1
    ) -> Union[Tuple[List, List], pd.DataFrame]:

        """
        Arguments:
            query_or_dataset (Union[str, Dataset]):
                str이나 Dataset으로 이루어진 Query를 받습니다.
                str 형태인 하나의 query만 받으면 `get_relevant_doc`을 통해 유사도를 구합니다.
                Dataset 형태는 query를 포함한 HF.Dataset을 받습니다.
                이 경우 `get_relevant_doc_bulk`를 통해 유사도를 구합니다.
            topk (Optional[int], optional): Defaults to 1.
                상위 몇 개의 passage를 사용할 것인지 지정합니다.

        Returns:
            1개의 Query를 받는 경우  -> Tuple(List, List)
            다수의 Query를 받는 경우 -> pd.DataFrame: [description]

        Note:
            다수의 Query를 받는 경우,
                Ground Truth가 있는 Query (train/valid) -> 기존 Ground Truth Passage를 같이 반환합니다.
                Ground Truth가 없는 Query (test) -> Retrieval한 Passage만 반환합니다.
        """

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
                for question in query_or_dataset["question"]:
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
                    "context": " ".join(
                        [self.contexts[pid] for pid in doc_indices[idx]]
                    ),
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
        tokenized_query = self.question_tokenize_fn(query)
        with timer("transform"):
            raw_doc_scores = self.bm25.get_scores(tokenized_query)

        doc_scores_index_desc = np.argsort(-raw_doc_scores)
        doc_scores = raw_doc_scores[doc_scores_index_desc]

        doc_list = self.bm25.get_top_n(tokenized_query, self.contexts, k)

        return doc_scores[:k], doc_list


class konlpy_tokenize:
    def __init__(self, nnp_score=3, nng_score=2, verb_score=1):
        self.noun_collector = Komoran()
        self.verb_collector = Okt()
        self.nnp_score = nnp_score
        self.nng_score = nng_score
        self.verb_score = verb_score

    def __pre_regex(self, context):
        re_compile = re.compile("[^a-zA-Z0-9ㄱ-ㅣ가-힣\s\(\)\[\]?!.,\@\*\{\}\-\_\=\+]")
        context = re.sub("\s", " ", context)
        re_context = re_compile.sub(" ", context)
        return re_context

    def __pre_devide(self, context):
        if len(context) < 3000:
            return [context]
        else:
            return re.split(".\s|.\\n", context)

    def __context_tokenize(self, context):
        tokenized_list = []
        context = context.strip()
        if context == "":
            return tokenized_list
        noum_tokenize = self.noun_collector.pos(context)
        verb_tokenize = self.verb_collector.pos(context, norm=True, stem=True)
        for word, tag in noum_tokenize:
            if tag == "NNG":
                tokenized_list.extend([word] * self.nng_score)
            elif tag == "NNP":
                tokenized_list.extend([word] * self.nnp_score)
        for word, tag in verb_tokenize:
            if tag == "Verb":
                tokenized_list.extend([word] * self.verb_score)
        return tokenized_list

    def tokenize_fn(self, context):
        context = self.__pre_regex(context)
        tokenized_list = []
        context_list = self.__pre_devide(context)
        for context in context_list:
            tokenized_list.extend(self.__context_tokenize(context))
        return tokenized_list

    def tokenize_fn_without_tagging(self, noum_list, verb_list):
        tokenized_list = []
        for word, tag in noum_list:
            if tag == "NNG":
                tokenized_list.extend([word] * self.nng_score)
            elif tag == "NNP":
                tokenized_list.extend([word] * self.nnp_score)
        for word, tag in verb_list:
            if tag == "Verb":
                tokenized_list.extend([word] * self.verb_score)
        return tokenized_list

    def question_tokenize(self, question):
        question = self.__pre_regex(question)
        tokenized_list = []
        noum_tokenize = self.noun_collector.pos(question)
        verb_tokenize = self.verb_collector.pos(question, norm=True, stem=True)
        for word, tag in noum_tokenize:
            if tag == "NNG":
                tokenized_list.append(word)
            elif tag == "NNP":
                tokenized_list.append(word)
        for word, tag in verb_tokenize:
            if tag == "Verb":
                tokenized_list.append(word)
        return tokenized_list
