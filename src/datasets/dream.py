# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseDataset


class DreamDataset(BaseDataset):
    pipeline_type = "mutiple-choice"
    def __init__(self,
                 data_path,
                 ):
        super(DreamDataset, self).__init__()
        self.data_path = data_path

    # @param batch_size: Int
    # @param types: List[Str] of "train", "dev", "test"
    # @yield batch: List[Dict]
    # - "article_id": "4-199"
    # - "question_id": 0
    # - "article": "My husband is a born shopper. He ..."
    # - "question": "The husband likes shopping because   _  ."
    # - "options": ["he has much money.", "he likes the shops.", "he likes to compare the prices between the same items.", "he has nothing to do but shopping."]
    # - "answer": 2
    def yield_batch(self,
                    batch_size,
                    types,
                    ):
        batch, current_batch_size,  = list(), 0
        for type_ in types:
            with open(os.path.join(self.data_path, f"{type_}.json"), 'r', encoding="utf8") as f:
                data = json.load(f)
            for (article_sentences, questions, article_id) in data:
                article = '\n'.join(article_sentences)
                for question_id, question_data in enumerate(questions):
                    question = question_data["question"]
                    choice = question_data["choice"]
                    flag = False
                    for i, option in enumerate(choice):
                        if option == question_data["answer"]:
                            assert not flag, f"There are two same options in question {id_}"
                            answer = i
                            flag = True
                    assert flag, f"There is no option matching answer in question {id_}"
                    batch.append({"article_id": article_id,
                                  "question_id": question_id,
                                  "article": article,
                                  "question": question,
                                  "options": choice,
                                  "answer": answer,
                                  })
                    current_batch_size += 1
                    if current_batch_size == batch_size:
                        yield batch
                        batch, current_batch_size = list(), 0
        if current_batch_size > 0:
            yield batch

    # Generate inputs for different models
    # @20240528: LIAMF-USP/roberta-large-finetuned-race
    # @20240528: potsawee/longformer-large-4096-answering-race
    @classmethod
    def generate_model_inputs(cls,
                              batch,
                              tokenizer,
                              model_name="LIAMF-USP/roberta-large-finetuned-race",
                              **kwargs,
                              ):
        if model_name == "LIAMF-USP/roberta-large-finetuned-race":
            max_length = kwargs["max_length"]
            batch_inputs = list()
            for data in batch:
                article = data["article"]
                question = data["question"]
                option = data["options"]
                flag = question.find('_') == -1
                inputs = list()
                for choice in option:
                    question_choice = question + ' ' + choice if flag else question.replace('_', choice)
                    input_ = tokenizer(article,
                                       question_choice,
                                       add_special_tokens = True,
                                       max_length = max_length,
                                       padding = "max_length",
                                       truncation = True,
                                       return_overflowing_tokens = False,
                                       return_tensors = "pt",
                                       ) # (1, max_length)
                    inputs.append(input_)
                batch_inputs.append(inputs)
            input_ids = torch.cat([torch.cat([input_["input_ids"] for input_ in inputs]).unsqueeze(0) for inputs in batch_inputs]) # (batch_size, n_option, max_length)
            attention_mask = torch.cat([torch.cat([input_["attention_mask"] for input_ in inputs]).unsqueeze(0) for inputs in batch_inputs]) # (batch_size, n_option, max_length)
            model_inputs = {"input_ids": input_ids,
                            "attention_mask": attention_mask,
                            }
        elif model_name == "potsawee/longformer-large-4096-answering-race":
            max_length = kwargs["max_length"]
            batch_inputs = list()
            for data in batch:
                article = data["article"]
                question = data["question"]
                option = data["options"]
                article_question = [f"{question} {tokenizer.bos_token} article"] * 4
                inputs = tokenizer(article_question,
                                   option,
                                   max_length = max_length,
                                   padding = "longest",
                                   truncation = True,
                                   return_tensors = "pt",
                                   ) # (4, max_length)
                batch_inputs.append(inputs)
            input_ids = torch.cat([inputs["input_ids"].unsqueeze(0) for inputs in batch_inputs], axis=0)
            attention_mask = torch.cat([inputs["attention_mask"].unsqueeze(0) for inputs in batch_inputs], axis=0)
            model_inputs = {"input_ids": input_ids,
                            "attention_mask": attention_mask,
                            }
        else:
            raise NotImplementedError(model_name)
        return model_inputs
