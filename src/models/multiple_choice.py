# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from transformers import (RobertaTokenizer,
						  RobertaForMultipleChoice,
						  LongformerTokenizer,
						  LongformerForMultipleChoice,
						  )
from src.models.base import MultipleChoiceModel


class RobertaLargeFinetunedRace(MultipleChoiceModel):
	# https://huggingface.co/LIAMF-USP/roberta-large-finetuned-race
	Tokenizer = RobertaTokenizer
	Model = RobertaForMultipleChoice
	model_name = "LIAMF-USP/roberta-large-finetuned-race"

	def __init__(self, model_path, device="cpu"):
		super(RobertaLargeFinetunedRace, self).__init__(model_path, device)


class LongformerLarge4096AnsweringRace(MultipleChoiceModel):
	# https://huggingface.co/potsawee/longformer-large-4096-answering-race
	Tokenizer = LongformerTokenizer
	Model = LongformerForMultipleChoice
	model_name = "potsawee/longformer-large-4096-answering-race"

	def __init__(self, model_path, device="cpu"):
		super(LongformerLarge4096AnsweringRace, self).__init__(model_path, device)

