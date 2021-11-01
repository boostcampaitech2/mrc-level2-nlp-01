from elasticsearch import Elasticsearch
from datasets import load_from_disk

wiki_datasets = load_from_disk("/opt/ml/data/wiki_preprocessed_droped")

es_client = Elasticsearch([{"host": "localhost", "port": "9200"}])  # default client

es_config = {
    "settings": {
        "analysis": {
            "tokenizer": {
                "nori_user_dict": {
                    "type": "nori_tokenizer",
                    "decompound_mode": "mixed",
                    # "user_dictionary": "komoran_nnp_len5.txt"
                }
            },
            "filter": {
                "stop_word_filter": {
                    "type": "stop",
                    "stopwords_path": "stop_words.txt",
                },
                "my_posfilter": {
                    "type": "nori_part_of_speech",
                    "stoptags": [
                        "E",
                        "IC",
                        "J",
                        "MAG", "MAJ", "MM",
                        "SP", "SSC", "SSO", "SC", "SE", "SF", "SY", "SH",
                        "XPN", "XSA", "XSN", "XSV",
                        "UNA", "NA", "VSV"
                    ]
                }
            },
            "analyzer": {
                "nori_analyzer": {
                    "type": "custom",
                    "tokenizer": "nori_user_dict",
                    #   "filter": ["stop_word_filter"],
                    "filter": ["my_posfilter"]
                },
            },
        },
        "similarity": {
            "DFR_custom": {
                "type": "DFR",
                "basic_model": "g",
                "after_effect": "l",
                "normalization": "h2",
                "normalization.h2.c": "3.0"
            },
            "DFI_custom": {
                "type": "DFI",
                "independence_measure": "standardized",
            },
            "BM25_custom": {
                "type": "BM25",
                "b": 0.3,
                "k1": 1.1
            },
        }
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "text": {"type": "text", "analyzer": "nori_analyzer", "similarity": "BM25_custom"}
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
