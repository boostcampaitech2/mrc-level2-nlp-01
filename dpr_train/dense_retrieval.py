import faiss
import random
import os
import torch
import numpy as np
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler
from datasets import load_from_disk
from transformers import AdamW, get_linear_schedule_with_warmup


class DenseRetrieval_with_Faiss:
    def __init__(
        self,
        args,
        dataset,
        tokenizer,
        q_encoder,
        p_encoder=None,
        num_neg=5,
        is_trained=False,
        wiki_path="/opt/ml/data/wiki_preprocessed_droped",
    ):
        self.args = args
        self.dataset = dataset
        self.num_neg = num_neg

        self.tokenizer = tokenizer
        self.p_encoder = p_encoder
        self.q_encoder = q_encoder

        if torch.cuda.is_available():
            if p_encoder is not None:
                self.p_encoder.cuda()
            self.q_encoder.cuda()

        self.wiki_dataset = load_from_disk(wiki_path)
        self.search_corpus = [example["text"] for example in self.wiki_dataset]

        if not is_trained:
            self.prepare_in_batch_negative(num_neg=num_neg)

    def prepare_in_batch_negative(self, dataset=None, num_neg=5, tokenizer=None):
        wiki_datasets = self.wiki_dataset
        wiki_datasets.load_elasticsearch_index(
            "text", host="localhost", port="9200", es_index_name="wikipedia_contexts"
        )
        if dataset is None:
            dataset = self.dataset

        if tokenizer is None:
            tokenizer = self.tokenizer

        # 1. In-Batch-Negative 만들기
        p_with_neg = []

        for c in tqdm(dataset):
            p_with_neg.append(c["context"])
            query = c["question"]
            p_neg = []
            _, retrieved_examples = wiki_datasets.get_nearest_examples(
                "text", query, k=num_neg * 10
            )
            for index in range(num_neg * 10):
                if retrieved_examples["document_id"][index] == c["document_id"]:
                    continue
                p_neg.append(retrieved_examples["text"][index])
            p_with_neg.extend(random.sample(p_neg, num_neg))
            assert len(p_with_neg) % (num_neg + 1) == 0, "데이터가 잘못 추가되었습니다."

        # 2. (Question, Passage) 데이터셋 만들어주기
        q_seqs = tokenizer(
            dataset["question"],
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        p_seqs = tokenizer(
            p_with_neg, padding="max_length", truncation=True, return_tensors="pt"
        )

        max_len = p_seqs["input_ids"].size(-1)
        p_seqs["input_ids"] = p_seqs["input_ids"].view(-1, num_neg + 1, max_len)
        p_seqs["attention_mask"] = p_seqs["attention_mask"].view(
            -1, num_neg + 1, max_len
        )
        p_seqs["token_type_ids"] = p_seqs["token_type_ids"].view(
            -1, num_neg + 1, max_len
        )

        train_dataset = TensorDataset(
            p_seqs["input_ids"],
            p_seqs["attention_mask"],
            p_seqs["token_type_ids"],
            q_seqs["input_ids"],
            q_seqs["attention_mask"],
            q_seqs["token_type_ids"],
        )

        self.train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=self.args.per_device_train_batch_size,
        )

    def build_faiss(
        self,
        index_file_path="/opt/ml/data/wiki.index",
        use_gpu=True,
        del_p_encoder=True,
    ):
        if os.path.isfile(index_file_path):
            self.indexer = faiss.read_index(index_file_path)
            if use_gpu:
                self.indexer = faiss.index_cpu_to_all_gpus(self.indexer)
        else:
            self.__make_faiss_index(index_file_path, use_gpu, del_p_encoder)

    def __make_faiss_index(
        self,
        index_file_path="/opt/ml/data/wiki.index",
        use_gpu=True,
        del_p_encoder=True,
    ):
        eval_batch_size = 8

        # Construt dataloader
        valid_p_seqs = self.tokenizer(
            self.search_corpus,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        valid_dataset = TensorDataset(
            valid_p_seqs["input_ids"],
            valid_p_seqs["attention_mask"],
            valid_p_seqs["token_type_ids"],
        )
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader = DataLoader(
            valid_dataset, sampler=valid_sampler, batch_size=eval_batch_size
        )

        # Inference using the passage encoder to get dense embeddeings
        p_embs = []

        with torch.no_grad():

            epoch_iterator = tqdm(
                valid_dataloader, desc="Iteration", position=0, leave=True
            )
            self.p_encoder.eval()

            for _, batch in enumerate(epoch_iterator):
                batch = tuple(t.cuda() for t in batch)

                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                outputs = self.p_encoder(**p_inputs).to("cpu").numpy()
                p_embs.extend(outputs)

        # 이제 p encoder 쓸 일이 없을경우 삭제
        if del_p_encoder:
            del self.p_encoder

        p_embs = np.array(p_embs)
        emb_dim = p_embs.shape[-1]

        self.indexer = faiss.IndexFlatL2(emb_dim)  # Flat에 GPU 사용
        if use_gpu:
            self.indexer = faiss.index_cpu_to_all_gpus(self.indexer)
        self.indexer.add(p_embs)
        faiss.write_index(
            faiss.index_gpu_to_cpu(self.indexer) if use_gpu else self.indexer,
            index_file_path,
        )

    def train(self, args=None):
        if args is None:
            args = self.args
        batch_size = args.per_device_train_batch_size

        # Optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.p_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.q_encoder.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
        )
        t_total = (
            len(self.train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

        # Start training!
        self.p_encoder.train()
        self.q_encoder.train()
        global_step = 0

        self.p_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(args.num_train_epochs)), desc="Epoch")
        # for _ in range(int(args.num_train_epochs)):
        for _ in train_iterator:

            with tqdm(self.train_dataloader, unit="batch") as tepoch:
                for batch in tepoch:

                    targets = torch.zeros(
                        batch_size
                    ).long()  # positive example은 전부 첫 번째에 위치하므로
                    targets = targets.to(args.device)

                    p_inputs = {
                        "input_ids": batch[0]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                        "attention_mask": batch[1]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                        "token_type_ids": batch[2]
                        .view(batch_size * (self.num_neg + 1), -1)
                        .to(args.device),
                    }

                    q_inputs = {
                        "input_ids": batch[3].to(args.device),
                        "attention_mask": batch[4].to(args.device),
                        "token_type_ids": batch[5].to(args.device),
                    }

                    # (batch_size*(num_neg+1), emb_dim)
                    p_outputs = self.p_encoder(**p_inputs)
                    # (batch_size*, emb_dim)
                    q_outputs = self.q_encoder(**q_inputs)

                    # Calculate similarity score & loss
                    p_outputs = p_outputs.view(batch_size, self.num_neg + 1, -1)
                    p_outputs = torch.transpose(p_outputs, 1, 2)
                    q_outputs = q_outputs.view(batch_size, 1, -1)

                    sim_scores = torch.bmm(
                        q_outputs, p_outputs
                    ).squeeze()  # (batch_size, num_neg + 1)
                    sim_scores = sim_scores.view(batch_size, -1)
                    sim_scores = F.log_softmax(sim_scores, dim=1)

                    loss = F.nll_loss(sim_scores, targets)
                    tepoch.set_postfix(loss=f"{str(loss.item())}")

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    self.p_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                    global_step += 1

                    torch.cuda.empty_cache()

                    del p_inputs, q_inputs

        print("save model & tokenizer")
        self.q_encoder.save_pretrained(args.output_dir)
        self.tokenizer.save_pretrained(args.output_dir)

    def get_relevant_doc(self, query, k=1):
        valid_q_seqs = self.tokenizer(
            query, padding="max_length", truncation=True, return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            self.q_encoder.eval()
            q_emb = self.q_encoder(**valid_q_seqs).to("cpu").numpy()
            del valid_q_seqs

        q_emb = q_emb.astype(np.float32)
        D, I = self.indexer.search(q_emb, k)
        distances, index = D.tolist()[0], I.tolist()[0]

        distance_list, doc_list = [], []
        for d, i in zip(distances, index):
            distance_list.append(d)
            doc_list.append(self.search_corpus[i])

        return distance_list, doc_list
