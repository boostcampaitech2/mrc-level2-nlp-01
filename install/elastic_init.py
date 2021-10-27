from elasticsearch import Elasticsearch
from datasets import Dataset
import pandas as pd
from datasets import load_from_disk, DatasetDict

wekipedia = pd.read_json("/opt/ml/data/wikipedia_documents.json")
wekipedia = wekipedia.T
wekipedia = wekipedia.loc[:, ['document_id', 'title', 'text']]
wekipedia.drop_duplicates('text', inplace=True)

wiki_datasets = Dataset.from_pandas(wekipedia)
wiki_datasets.save_to_disk("/opt/ml/data/wiki_datasets")

es_client = Elasticsearch([{"host": "localhost", "port": "9200"}])  # default client

es_config = {
    "settings": {
        "analysis": {
            "filter": {
                "stop_word_filter": {
                    "type": "stop",
                    "stopwords_path": "stop_words.txt",
                }
            },
            "analyzer": {
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    "filter": ["stop_word_filter"],
                }
            },
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "text": {"type": "text", "analyzer": "nori_analyzer", "similarity": "BM25"}
        },
    },
}  # default config

es_index_name = "wikipedia_contexts"  # name of the index in ElasticSearch
wiki_datasets.add_elasticsearch_index(
    "text",
    es_client=es_client,
    es_index_config=es_config,
    es_index_name=es_index_name,
)