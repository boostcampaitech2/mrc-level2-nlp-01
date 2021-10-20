import json
import re
from tqdm import tqdm
from konlpy.tag import Komoran, Kkma, Mecab, Okt, Hannanum 

komo = Komoran()
ok = Okt()
mecab = Mecab()

def pre_regex(context):
    re_compile = re.compile('[^a-zA-Z0-9ㄱ-ㅣ가-힣\s\(\)\[\]\?\!\.\,\@\*\{\}\-\_\=\+]')
    context = re.sub('\s', ' ', context)
    re_context = re_compile.sub(' ', context)
    return re_context
def pre_devide(context):
  if len(context) < 3000:
    return [context]
  else:
    return re.split('\.\s|\.\\n',context)

with open("../data/wikipedia_documents.json", "r", encoding='UTF-8') as json_data:
  wiki_data = json.load(json_data)
wiki_keys = wiki_data.keys()

wiki_tags_docs = {}
i =0
for key in tqdm(wiki_keys):
  wiki_tags = {}
  text = wiki_data[key]['text']
  text = pre_regex(text)
  text_list = pre_devide(text)
  okt_list = []
  komoran_list = []
  mecab_list = []
  for context in text_list:
    context = context.strip()
    if context == '':
      continue
    okt_list.extend(ok.pos(context, norm=True, stem=True))
    komoran_list.extend(komo.pos(context))
    mecab_list.extend(mecab.pos(context))
  wiki_tags['komoran'] = komoran_list
  wiki_tags['okt'] = okt_list
  wiki_tags['mecab'] = mecab_list
  wiki_tags_docs[key] = wiki_tags
  

file_path = "./tag_docs.json"

with open(file_path, 'w', encoding='UTF-8') as outfile:
    json.dump(wiki_tags_docs, outfile)