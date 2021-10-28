from datasets import Dataset, load_from_disk
import os

def initializeES(es_cfg):
    wiki_datasets = load_from_disk(os.path.join(es_cfg.root, "wiki_datasets"))
    es_index_name = "wikipedia_contexts"  # name of the index in ElasticSearch
    wiki_datasets.load_elasticsearch_index(
        "text", host="localhost", port="9200", es_index_name=es_index_name
    )

    # test 데이터셋 불러오기
    test_dataset = load_from_disk(os.path.join(es_cfg.root, es_cfg.data_path))
    test_valid = test_dataset["validation"]

    # top_k개 만큼 문서 검색
    context_list = []
    scores_list = []
    for idx in range(test_valid.num_rows):
        scores, context = wiki_datasets.get_nearest_examples(
            "text", test_valid["question"][idx], k=es_cfg.top_k
        )
        context_list.append(context["text"])
        scores_list.append(scores)

    # 데이터프레임 형태로 불러온 문서와 유사도 점수 컬럼 추가
    test_pd = test_valid.to_pandas()
    test_pd["context"] = context_list
    test_pd["score"] = scores_list
    test_valid = Dataset.from_pandas(test_pd)
    test_valid.save_to_disk(os.path.join(es_cfg.root, f"retrival_test_{es_index_name}_{es_cfg.top_k}"))
