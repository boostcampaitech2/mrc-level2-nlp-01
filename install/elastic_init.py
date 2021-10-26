import json

from elasticsearch import Elasticsearch
from datasets import Dataset

with open("/opt/ml/data/wikipedia_documents.json", "r", encoding="UTF-8") as json_data:
    wiki_data = json.load(json_data)
wiki_keys = wiki_data.keys()
wiki_texts = []

for key in wiki_keys:
    wiki_texts.append(wiki_data[key]["text"])

data = {"document_id": [], "title": [], "context": []}

for key in wiki_keys:
    data["document_id"].append(wiki_data[key]["document_id"])
    data["title"].append(wiki_data[key]["title"])
    data["context"].append(wiki_data[key]["text"])

wiki_datasets = Dataset.from_dict(data)
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
    "context",
    es_client=es_client,
    es_index_config=es_config,
    es_index_name=es_index_name,
)
