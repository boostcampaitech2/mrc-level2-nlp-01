def prepare_train_features_with_setting(tokenizer, dataset_args, is_roberta):
    def prepare_train_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
        inputs = [f"질문: {q}  지문: {c}" for q, c in zip(examples["question"], examples["context"])]
        extra_inputs = [f"질문: {q}  지문:" for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_extras = tokenizer(
            extra_inputs,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding=False,
        )

        tokenized_examples = tokenizer(
            inputs,
            truncation=True,
            max_length=512,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            additional_token_length = len(tokenized_extras["input_ids"][i][:-1])
            additional_text_length = len(extra_inputs[i])
            cls_index = tokenizer.eos_token_id # dataset_args.max_length
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            # sample_index = sample_mapping[i]
            answers = examples["answers"][i]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0] + additional_text_length # text 길이로 변경
                end_char = start_char + len(answers["text"][0]) + 1
                # Start token index of the current span in the text.
                token_start_index = 0
                # while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                #     token_start_index += 1
                token_start_index += additional_token_length

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while input_ids[token_end_index] == tokenizer.pad_token_id or input_ids[token_end_index] == tokenizer.eos_token_id:
                    token_end_index -= 1
                # token_end_index -= additional_length # TODO ?
                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while ( # offset이 정답 시작 인덱스보다 작고, 현재 토큰의 오프셋의 시작이 정답 시작 인덱스보다 작은경우
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    return prepare_train_features


def prepare_validation_features_with_setting(tokenizer, dataset_args, is_roberta):
    def prepare_validation_features(examples):
        # Some of the questions have lots of whitespace on the left, which is not useful and will make the
        # truncation of the context fail (the tokenized question will take a lots of space). So we remove that
        # left whitespace
        examples["question"] = [q.lstrip() for q in examples["question"]]
        # pad_on_right = tokenizer.padding_side == "right"

        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        inputs = [f"질문: {q}  지문: {c}" for q, c in zip(examples["question"], examples["context"])]
        extra_inputs = [f"질문: {q}  지문:" for q in examples["question"]]

        # Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_extras = tokenizer(
            extra_inputs,
            truncation=True,
            max_length=dataset_args.max_length,
            # stride=dataset_args.stride,
            # return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=False,
            # return_token_type_ids=False if is_roberta else True,
        )

        tokenized_examples = tokenizer(
            inputs,
            # examples["question" if pad_on_right else "context"],
            # examples["context" if pad_on_right else "question"],
            # truncation="only_second" if pad_on_right else "only_first",
            truncation=True,
            max_length=dataset_args.max_length,
            # stride=dataset_args.stride,
            # return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
            # return_token_type_ids=False if is_roberta else True,
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        # sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        # print(len(tokenized_examples["input_ids"]))
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            # sequence_ids = tokenized_examples.sequence_ids(i)
            # context_index = 1 # if pad_on_right else 0
            additional_token_length = len(tokenized_extras['input_ids'][i])

            # One example can give several spans, this is the index of the example containing this span of text.
            # sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][i])
            # print(examples["examid"][i])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if k >= additional_token_length else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
            # print(len(tokenized_examples['offset_mapping'][i]))
        return tokenized_examples

    return prepare_validation_features

# def prepare_train_features_with_setting(tokenizer, dataset_args, is_roberta):
#     def prepare_train_features(examples, max_length, stride):
#         # 주어진 텍스트를 토크나이징 한다. 이 때 텍스트의 길이가 max_seq_length를 넘으면 stride만큼 슬라이딩하며 여러 개로 쪼갬.
#         # 즉, 하나의 example에서 일부분이 겹치는 여러 sequence(feature)가 생길 수 있음.
#         tokenized_examples = tokenizer(
#             examples["question"],
#             examples["context"],
#             truncation="only_second",  # max_seq_length까지 truncate한다. pair의 두번째 파트(context)만 잘라냄.
#             max_length=max_length,
#             stride=stride,
#             return_overflowing_tokens=True, # 길이를 넘어가는 토큰들을 반환할 것인지
#             return_offsets_mapping=True,  # 각 토큰에 대해 (char_start, char_end) 정보를 반환한 것인지
#             padding="max_length",
#             return_token_type_ids=False,
#         )
        
#         overflow_to_sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
#         offset_mapping = tokenized_examples.pop("offset_mapping")
#         input_ids_list = tokenized_examples.pop("input_ids")
#         attention_mask = tokenized_examples.pop("attention_mask")
        
        
#         # 정답지를 만들기 위한 리스트
#         tokenized_examples["input_ids"] = []
#         tokenized_examples["attention_mask"] = []
#         tokenized_examples["start_positions"] = []
#         tokenized_examples["end_positions"] = []
#         tokenized_examples["uid"] = []
#         id_set = set()
        
#         for i, offsets in enumerate(offset_mapping):
#             input_ids = input_ids_list[i] #tokenized_examples["input_ids"][i]
#             cls_index = input_ids.index(tokenizer.cls_token_id)
            
#             # 해당 example에 해당하는 sequence를 찾음.
#             sequence_ids = tokenized_examples.sequence_ids(i)
            
#             # sequence가 속하는 example을 찾는다.
#             example_index = overflow_to_sample_mapping[i]
#             answers = examples["answers"][example_index]
#             example_id = examples["id"][example_index]
            
#             # 텍스트에서 answer의 시작점, 끝점
#             answer_start_offset = answers["answer_start"][0]
#             answer_end_offset = answer_start_offset + len(answers["text"][0])

#             # 텍스트에서 현재 span의 시작 토큰 인덱스
#             token_start_index = 0
#             while sequence_ids[token_start_index] != 1:
#                 token_start_index += 1
            
#             # 텍스트에서 현재 span 끝 토큰 인덱스
#             token_end_index = len(input_ids) - 1
#             while sequence_ids[token_end_index] != 1:
#                 token_end_index -= 1
            
#             # answer가 현재 span을 벗어났는지 체크
#             if not (offsets[token_start_index][0] <= answer_start_offset and offsets[token_end_index][1] >= answer_end_offset) or (example_id in id_set):
#                 # tokenized_examples["start_positions"].append(cls_index)
#                 # tokenized_examples["end_positions"].append(cls_index)=
#                 continue
#             else:
#                 # token_start_index와 token_end_index를 answer의 시작점과 끝점으로 옮김
#                 while token_start_index < len(offsets) and offsets[token_start_index][0] <= answer_start_offset:
#                     token_start_index += 1
#                 tokenized_examples["start_positions"].append(token_start_index - 1)
#                 while offsets[token_end_index][1] >= answer_end_offset:
#                     token_end_index -= 1
#                 tokenized_examples["end_positions"].append(token_end_index + 1)
#                 tokenized_examples["input_ids"].append(input_ids)
#                 tokenized_examples["attention_mask"].append(attention_mask[i])
#                 tokenized_examples["uid"].append(example_id)
#                 id_set.add(example_id)
        
#         return tokenized_examples

#     return prepare_train_features