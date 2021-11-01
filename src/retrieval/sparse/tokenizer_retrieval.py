from typing import List, NoReturn, Optional
from transformers import AutoTokenizer

from src.retrieval.sparse.bm25_retrieval_base import BM25SparseRetrieval

class TokenizerRetrieval(BM25SparseRetrieval):
  def __init__(self, tokenizer_name,data_path: Optional[str] = "/opt/ml/data/", context_path: Optional[str] = "wiki_preprocessed_droped") -> NoReturn:
      super().__init__(data_path=data_path, context_path=context_path)
      self.tokenizer_name = tokenizer_name
      self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
  def tokenizer_fn(self, context) -> List[str]:
      return self.tokenizer.tokenize(context)
  def get_sparse_embedding(self, pickle_name="bm25.bin", type='Plus', k1=1.6, b=0.3, ep=0.25, delta=0.7) -> NoReturn:
      pickle_name = f"{self.tokenizer_name}_{pickle_name}"
      return super().get_sparse_embedding(pickle_name=pickle_name, type=type, k1=k1, b=b, ep=ep, delta=delta)