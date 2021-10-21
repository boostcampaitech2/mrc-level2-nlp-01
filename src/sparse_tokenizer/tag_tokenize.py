import re

from konlpy.tag import Komoran, Okt
from tqdm import tqdm


class konlpy_tokenize:
    def __init__(self, nnp_score=3, nng_score=2, verb_score=1):
        self.noun_collector = Komoran()
        self.verb_collector = Okt()
        self.nnp_score = nnp_score
        self.nng_score = nng_score
        self.verb_score = verb_score

    def __pre_regex(self, context):
        re_compile = re.compile("[^a-zA-Z0-9ㄱ-ㅣ가-힣\s\(\)\[\]?!.,\@\*\{\}\-\_\=\+]")
        context = re.sub("\s", " ", context)
        re_context = re_compile.sub(" ", context)
        return re_context

    def __pre_devide(self, context):
        if len(context) < 3000:
            return [context]
        else:
            return re.split(".\s|.\\n", context)

    def __context_tokenize(self, context):
        tokenized_list = []
        context = context.strip()
        if context == "":
            return tokenized_list
        noum_tokenize = self.noun_collector.pos(context)
        verb_tokenize = self.verb_collector.pos(context, norm=True, stem=True)
        for word, tag in noum_tokenize:
            if tag == "NNG":
                tokenized_list.extend([word] * self.nng_score)
            elif tag == "NNP":
                tokenized_list.extend([word] * self.nnp_score)
        for word, tag in verb_tokenize:
            if tag == "Verb":
                tokenized_list.extend([word] * self.verb_score)
        return tokenized_list

    def tokenize_fn(self, context):
        context = self.__pre_regex(context)
        tokenized_list = []
        context_list = self.__pre_devide(context)
        for context in context_list:
            tokenized_list.extend(self.__context_tokenize(context))
        return tokenized_list

    def question_tokenize(self, question):
        question = self.__pre_regex(question)
        tokenized_list = []
        noum_tokenize = self.noun_collector.pos(question)
        verb_tokenize = self.verb_collector.pos(question, norm=True, stem=True)
        for word, tag in noum_tokenize:
            if tag == "NNG":
                tokenized_list.append(word)
            elif tag == "NNP":
                tokenized_list.append(word)
        for word, tag in verb_tokenize:
            if tag == "Verb":
                tokenized_list.append(word)
        return tokenized_list
