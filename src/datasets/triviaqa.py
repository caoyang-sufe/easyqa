# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseDataset


class TriviaqaDataset(BaseDataset):
    pipeline_type = "mutiple-choice"
    def __init__(self,
                 data_path,
                 ):
        super(HotpotqaDataset, self).__init__()
        self.data_path = data_path

    # @param batch_size: Int
    # @param filename: Str, e.g. "train_v1.1.json", "dev_distractor_v1.json", "dev_fullwiki_v1.json", "test_fullwiki_v1.json"
    # @yield batch: List[Dict]
    # - id: "5adf9ba1554299025d62a2db"
    # - question: "What position on the Billboard Top 100 did Alison Moyet's late summer hit achieve?"
    # - context: List[Tuple[Str, List[Str]]]: In HotpotQA, The first Str in Tuple is title, and the List is tokenized sentences
    #   e.g. [["The Other Side of Love", ["\"The Other Side of Love\" is a song ...", ]], [..., [...]], ...]
    # - answer: "Yes" (may be None in test file)
    # - type: "comparison"
    # - level: "hard"
    def yield_batch(self,
                    batch_size,
                    filename,
                    ):
        batch, current_batch_size, = list(), 0
        with open(os.path.join(self.data_path, filename), 'r', encoding="utf8") as f:
            data = json.load(f)
        for sample in data:
            id_ = sample["_id"]
            question = sample["question"]
            context = sample["context"]
            answer = sample.get("answer")  # Maynot in test file
            type_ = sample.get("type")  # Maynot in test file
            level = sample.get("level")  # Maynot in test file
            batch.append({"id": id_,
                          "question": question,
                          "context": context,
                          "answer": answer,
                          "type": type_,
                          "level": level,
                          })
            current_batch_size += 1
            if current_batch_size == batch_size:
                yield batch
                batch, current_batch_size, = list(), 0
        if current_batch_size > 0:
            yield batch

    # Generate inputs for different models
    @classmethod
    def generate_model_inputs(cls,
                              batch,
                              tokenizer,
                              model_name="AdapterHub/roberta-base-pf-hotpotqa",
                              **kwargs,
                              ):
        if model_name == "AdapterHub/roberta-base-pf-hotpotqa":
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
