from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

query_model = "klue/roberta-large"
passage_model = "klue/roberta-large"

retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
    query_embedding_model=query_model,
    passage_embedding_model=passage_model,
    max_seq_len_query=64,
    max_seq_len_passage=512
)

retriever.train(
    data_dir='./',
    train_filename='train.json',
    dev_filename='valid.json',
    test_filename='valid.json',
    n_epochs=20,
    batch_size=2,
    grad_acc_steps=16,
    save_dir='./save',
    evaluate_every=3000,
    embed_title=True,
    num_positives=1,
    num_hard_negatives=2
)