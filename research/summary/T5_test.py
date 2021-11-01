import numpy as np
import tensorflow as tf
import tensorflow_text
import json

from transformers import AutoTokenizer


model_path = "/opt/ml/code/ipynb_research/downstream_exported/t5.1.1.large.gin.ke.ke_v100_span_corruption.ke_t5_nikl_summary_mixture_equal"
loaded = tf.saved_model.load(model_path)
infer = loaded.signatures["serving_default"]

model_name = "klue/roberta-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)

with open("../../data/wikipedia_documents.json", "r", encoding='UTF-8') as json_data:
    wiki_data = json.load(json_data)
wiki_keys = wiki_data.keys()
wiki_texts = []

for key in wiki_keys:
    k = wiki_data[key]
    k['key'] = key
    wiki_texts.append(k)

def summary_sentence(context):
    input_str = "summarize: " + context
    x = tf.constant([input_str])

    result = infer(x)
    sm_sentence = [out.decode('utf-8') for out in result['outputs'].numpy()]
    return sm_sentence


from tqdm import tqdm

wiki_summary = {}

for wiki_text in (tqdm(wiki_texts)):
    context = wiki_text['text']
    tokenized_context = tokenizer.tokenize(context)
    wiki_target = {}
    if len(tokenized_context) > 576:
        wiki_target['text'] = summary_sentence(context)
    else:
        wiki_target['text'] = context
    wiki_target['title'] = wiki_text['title']
    wiki_target['document_id'] = wiki_text['document_id']
    wiki_summary[wiki_text['key']] = wiki_target


file_path = "./wiki_summary.json"

with open(file_path, 'w', encoding='UTF-8') as outfile:
    json.dump(wiki_summary, outfile, indent=4)