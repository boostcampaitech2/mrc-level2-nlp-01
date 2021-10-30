import os

from datasets import Dataset, DatasetDict, load_from_disk


def initializeES(es_cfg):
    wiki_datasets = load_from_disk(os.path.join(es_cfg.root, es_cfg.wiki_path))
    es_index_name = "wikipedia_contexts"  # name of the index in ElasticSearch
    
    wiki_datasets.load_elasticsearch_index(
        "text", host="localhost", port="9200", es_index_name=es_index_name
    )

    # test 데이터셋 불러오기
    test_dataset = load_from_disk(os.path.join(es_cfg.root, es_cfg.data_path))
    if isinstance(test_dataset, DatasetDict):
        test_valid = test_dataset["validation"] # 데이터셋 형태마다 달라질 것 같아요
    else:
        test_valid = test_dataset

    # top_k개 만큼 문서 검색
    context_list, context_string, scores_list = [], [], []

    for idx in range(test_valid.num_rows):
        scores, context = wiki_datasets.get_nearest_examples(
            "text", test_valid["question"][idx], k=es_cfg.top_k
        )
        context_list.append(context["text"])
        context_string.append(" ".join(context["text"]))
        scores_list.append(scores)

    # 데이터프레임 형태로 불러온 문서와 유사도 점수 컬럼 추가
    test_pd = test_valid.to_pandas()
    test_pd["context"] = context_string
    test_pd["context_list"] = context_list
    test_pd["score"] = scores_list
    test_valid = Dataset.from_pandas(test_pd)
    test_valid.save_to_disk(os.path.join(es_cfg.root, f"retrival_test_{es_index_name}_{es_cfg.top_k}"))