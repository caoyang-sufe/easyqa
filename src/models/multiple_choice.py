# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
import torch
from transformers import (RobertaTokenizer,
                          RobertaForMultipleChoice,
                          LongformerTokenizer,
                          LongformerForMultipleChoice,
                          )
from src.datasets import RaceDataset
from src.models.base import BaseModel


class RobertaLargeFinetunedRace(BaseModel):
    # https://huggingface.co/LIAMF-USP/roberta-large-finetuned-race
    Tokenizer = RobertaTokenizer
    Model = RobertaForMultipleChoice
    model_name = "LIAMF-USP/roberta-large-finetuned-race"

    def __init__(self, model_path, device="cpu"):
        super(RobertaLargeFinetunedRace, self).__init__(model_path=model_path, device=device)

    # @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
    # @return batch_logits: FloatTensor(batch_size, 4)
    # @return batch_predicts: List[Str] (batch_size, )
    def run(self, batch, max_length=512):
        model_inputs = RaceDataset.generate_model_inputs(batch=batch,
                                                         tokenizer=self.tokenizer,
                                                         model_name=self.model_name,
                                                         max_length=max_length,
                                                         )
        batch_logits = self.model(**model_inputs).logits
        del model_inputs
        batch_predicts = ["ABCD"[torch.argmax(logits).item()] for logits in batch_logits]
        return batch_logits, batch_predicts


class LongformerLarge4096AnsweringRace(BaseModel):
    # https://huggingface.co/potsawee/longformer-large-4096-answering-race
    Tokenizer = LongformerTokenizer
    Model = LongformerForMultipleChoice
    model_name = "potsawee/longformer-large-4096-answering-race"

    def __init__(self, model_path, device="cpu"):
        super(LongformerLarge4096AnsweringRace, self).__init__(model_path=model_path, device=device)

    # @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
    # @return batch_logits: FloatTensor(batch_size, 4)
    # @return batch_predicts: List[Str] (batch_size, )
    def run(self, batch, max_length=4096):
        return self.run_longformer_large_4096_answering_race(batch, self.tokenizer, self.model, max_length)

    # @param data: Dict[article(List[Str]), question(List[Str]), options(List[List[Str]])]
    # @return batch_logits: FloatTensor(batch_size, 4)
    # @return batch_predicts: List[Str] (batch_size, )
    def run_longformer_large_4096_answering_race(self,
                                                 batch,
                                                 tokenizer,
                                                 model,
                                                 max_length=4096,
                                                 ):
        model_inputs = RaceDataset.generate_model_inputs(batch=batch,
                                                         tokenizer=tokenizer,
                                                         model_name=self.model_name,
                                                         max_length=max_length,
                                                         )
        batch_logits = model(**model_inputs).logits
        del model_inputs
        batch_predicts = ["ABCD"[torch.argmax(logits).item()] for logits in batch_logits]
        return batch_logits, batch_predicts
