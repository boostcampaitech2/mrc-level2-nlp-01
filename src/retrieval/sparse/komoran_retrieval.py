from typing import List, NoReturn, Optional
from konlpy.tag import Komoran

from src.retrieval.sparse.bm25_retrieval_base import BM25SparseRetrieval

class KomoranRetrieval(BM25SparseRetrieval):
  def __init__(self, data_path: Optional[str] = "/opt/ml/data/", context_path: Optional[str] = "wiki_preprocessed_droped") -> NoReturn:
      super().__init__(data_path=data_path, context_path=context_path)
      self.tag = Komoran()
      self.ignore_tag = ["NNB", "NR", "NP", "VA", "VX", "MM", "VCP", "VCN", "MAG", "MAJ", "IC", "JKS", "JKC", "JKG", "JKO", "JKB", "JKV", "JKQ", "JC", "JX", "EP", "EF", "EC", "ETN", "ETM", "XPN", "XSN", "XSV", "XSA", "XR", "SF", "SE", "SS", "SP", "SO", "SW", "SH"]
  def tokenizer_fn(self, context) -> List[str]:
      tag_context = self.tag.pos(context)
      tokenized_context = []
      for text, tag in tag_context:
        if tag in self.ignore_tag:
          continue
        tokenized_context.append(text)
      return tokenized_context
  def get_sparse_embedding(self, pickle_name="komoran_api.bin", type='Plus', k1=1.6, b=0.3, ep=0.25, delta=0.7) -> NoReturn:
      return super().get_sparse_embedding(pickle_name=pickle_name, type=type, k1=k1, b=b, ep=ep, delta=delta)