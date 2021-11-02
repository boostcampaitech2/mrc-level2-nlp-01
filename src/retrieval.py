import os

from tqdm import tqdm
from importlib import import_module
from datasets import Dataset, DatasetDict, load_from_disk
from dataclasses import asdict

from src.arguments import RetrievalArguments


def make_retrieval_datasets(retrieval_cfg: RetrievalArguments):
    os.environ["WANDB_DISABLED"] = "true"
    retrieval_cfg = RetrievalArguments(**retrieval_cfg)
    # test 데이터셋 불러오기
    result_dataset_dict = {}
    target_dataset = load_from_disk(
        os.path.join(retrieval_cfg.data_path, retrieval_cfg.target_dataset)
    )
    dataset_key = target_dataset.keys()

    # top_k개 만큼 문서 검색
    retrieval_module = getattr(
        import_module("src.retrievals.sparse"),
        retrieval_cfg.retrieval_class,
    )
    retrieval_cfg_dict = asdict(retrieval_cfg)
    retrieval = retrieval_module(**retrieval_cfg_dict)
    retrieval.get_sparse_embedding(**retrieval_cfg_dict)

    for key in dataset_key:
        if key == "train":
            pass  # 트레인용을 만들어라!
        else:
            datas = {"scores": [], "context": [], "context_list": []}
            for data_key in target_dataset[key].column_names:
                datas[data_key] = []
            for idx in tqdm(range(target_dataset[key].num_rows)):
                scores, context = retrieval.get_relevant_doc(
                    target_dataset[key]["question"][idx], k=retrieval_cfg.k
                )
                for data_key in target_dataset[key].column_names:
                    datas[data_key].append(target_dataset[key][data_key][idx])
                datas["scores"].append(scores)
                datas["context"].append(" ".join(context))
                datas["context_list"].append(context)
            result_dataset_dict[key] = Dataset.from_dict(datas)

    new_dataset_dict = DatasetDict(result_dataset_dict)
    new_dataset_dict.save_to_disk(
        os.path.join(retrieval_cfg.data_path, retrieval_cfg.output_name)
    )
    print(
        f"Success! save at {os.path.join(retrieval_cfg.data_path, retrieval_cfg.output_name)}"
    )
