import re

from konlpy.tag import Komoran


class KonlpyTokenize:
    def __init__(self):
        self.noun_collector = Komoran()

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
        for word, tag in noum_tokenize:
            if tag == "NNG" or tag == "NNP":
                tokenized_list.append(word)
        return tokenized_list

    def tokenize_fn(self, context):
        context = self.__pre_regex(context)
        tokenized_list = []
        context_list = self.__pre_devide(context)
        for context in context_list:
            tokenized_list.extend(self.__context_tokenize(context))
        return tokenized_list
