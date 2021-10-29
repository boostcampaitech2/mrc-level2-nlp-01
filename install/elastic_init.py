from elasticsearch import Elasticsearch
from datasets import load_from_disk

wiki_datasets = load_from_disk("/opt/ml/data/wiki_preprocessed_droped")

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
                    #   "filter": ["stop_word_filter"],
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
