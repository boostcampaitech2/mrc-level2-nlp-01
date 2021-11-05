import os

from typing import Optional, Tuple, List
from datasets import load_from_disk


class ElasticSearchRetrieval:
    def __init__(
        self,
        data_path: Optional[str] = "/opt/ml/data/",
        context_path: Optional[str] = "wiki_preprocessed_droped",
        **kwargs,
    ):
        self.wiki_datasets = load_from_disk(os.path.join(data_path, context_path))
        es_index_name = "wikipedia_contexts"  # name of the index in ElasticSearch

        self.wiki_datasets.load_elasticsearch_index(
            "text", host="localhost", port="9200", es_index_name=es_index_name
        )

    def get_sparse_embedding(self, *args, **kwargs):
        pass

    def get_relevant_doc(self, query: str, k: Optional[int] = 1) -> Tuple[List, List]:
        scores, contexts = self.wiki_datasets.get_nearest_examples("text", query, k=k)
        return scores, contexts["text"]
