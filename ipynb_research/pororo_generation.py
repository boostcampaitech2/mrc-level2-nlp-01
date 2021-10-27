import os
import re

from konlpy.tag import Komoran, Kkma, Hannanum, Okt, Mecab
from pororo import Pororo

class DataGenerator:
  def __init__(self):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    self.qg = Pororo(task="qg", lang="ko")
    self.ner = Pororo(task="ner", lang="ko")
    self.mecab = Mecab()
    self.okt = Okt()
    self.komoran = Komoran()
    self.kkma = Kkma()
    self.han = Hannanum()
    self.id = 1
    self.p = re.compile(r'(\d+)월 (\d+)일')

  def delete_duplicate(self, answer_list): 
    new_answer_list = []
    duplicate_list = []
    for answer in answer_list:
      if answer not in new_answer_list:
        new_answer_list.append(answer)
      else:
        duplicate_list.append(answer)
    duplicate_list = list(set(duplicate_list))
    for dup_ans in duplicate_list:
      new_answer_list.remove(dup_ans)
    return new_answer_list

  def get_only_komoran_NNP(self, answer_list):
    new_answer_list = []
    for answer in answer_list:
      if len(self.han.pos(answer)) > 1 and len(self.mecab.pos(answer)) > 1 and len(self.kkma.pos(answer)) > 1 and len(self.okt.pos(answer)) > 1:
        if self.p.match(answer):
          continue
        new_answer_list.append(answer)
    return new_answer_list

  def get_komoran_NNP(self, context):
    pos_list = self.komoran.pos(context)
    pos_list = list(filter(lambda x: x[1] == 'NNP', pos_list))
    answer_list = [ answer[0] for answer in pos_list ]
    answer_list = self.delete_duplicate(answer_list)
    answer_list = self.get_only_komoran_NNP(answer_list)
    return answer_list
  
  def get_date(self, context):
    ners = self.ner(context)
    quentity_date_ner = list(filter(lambda x: x[1] == 'DATE', ners))
    return [answer[0] for answer in quentity_date_ner]
  
  def get_date_question(self, context):
    answers = self.get_date(context)
    answers = self.delete_duplicate(answers)
    return answers
  
  def get_question_N(self, context):
    answer_list = self.get_komoran_NNP(context)
    if len(answer_list) == 0:
      return [], []
    return answer_list, self.qg(answer_list[:3], context)
  
  def get_question_Q(self, context):
    answer_list = self.get_date_question(context)
    if len(answer_list) == 0:
      return [], []
    return answer_list, self.qg(answer_list[:3], context)
    
  def generate_question_with_dataset_format(self, wiki_data, q_type = "N"):
    datasets = []
    if q_type == "N":
      answer_list, question_list = self.get_question_N(wiki_data['context'])
    elif q_type == "Q":
      answer_list, question_list = self.get_question_Q(wiki_data['context'])
    else:
      raise "정확한 q_type을 입력하세요. N or Q"
    for answer, question in zip(answer_list, question_list):
      data = {
        'context': wiki_data['context'],
        'question': question,
        'id': self.id,
        'title': wiki_data['title'],
        'document_id': wiki_data['document_id'],
        'answers': {
          'answer_start': [wiki_data['context'].index(answer)],
          'text': [answer]
        }
      }
      self.id += 1
      datasets.append(data)
    return datasets
    
    
from datasets import load_from_disk

wiki_datasets = load_from_disk("/opt/ml/data/wiki_datasets")

dg = DataGenerator()

from tqdm import tqdm

datasets = {
  'context': [],
  'question': [],
  'id': [],
  'title': [],
  'document_id': [],
  'answers': []
}
error_data = []
for index in tqdm(range(len(wiki_datasets))):
  data_list = []
  try:
    data_list.extend(dg.generate_question_with_dataset_format(wiki_datasets[index]))
  except:
    error_data.append(index)
  try:
    data_list.extend(dg.generate_question_with_dataset_format(wiki_datasets[index], q_type='Q'))
  except:
    error_data.append(index)
  for data in data_list:
    for key in datasets.keys():
      datasets[key].append(data[key])
      
from datasets import Dataset

new_train_data = Dataset.from_dict(datasets)
new_train_data.save_to_disk("/opt/ml/data/generator_dataset")