from datasets import load_from_disk
from konlpy.tag import Komoran, Okt

from tqdm import tqdm

komoran = Komoran()
okt = Okt()

datasets = load_from_disk("/opt/ml/data/wiki_datasets")
okt_tag = [
    "Determiner",
    "Adverb",
    "Conjunction",
    "Exclamation",
    "Josa",
    "PreEomi",
    "Eomi",
    "Suffix",
]
komoran_tag = [
    "MM",
    "MAG",
    "MAJ",
    "IC",
    "JKS",
    "JKC",
    "JKG",
    "JKO",
    "JKB",
    "JKV",
    "JKQ",
    "JC",
    "JX",
    "EP",
    "EF",
    "EC",
    "ETN",
    "ETM",
    "XPN",
    "XSN",
    "XSV",
    "XSA",
    "SE",
    "SS",
    "SP",
]

stop_words = set()
error_context = []


for data in tqdm(datasets):
    try:
        tag_list = okt.pos(data["context"])
        for tag in tag_list:
            if tag[1] in okt_tag:
                stop_words.add(tag[0])
    except:
        error_context.append(data)
for data in tqdm(datasets):
    try:
        tag_list = komoran.pos(data["context"])
        for tag in tag_list:
            if tag[1] in komoran_tag:
                stop_words.add(tag[0])
    except:
        error_context.append(data)
with open("stop_words.txt", "w") as f:
    for word in stop_words:
        f.write(f"{word}\n")
