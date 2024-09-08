# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import time

import torch
import pandas

from torch.optim import Adam, lr_scheduler

from settings import DATA_SUMMARY, MODEL_SUMMARY, LOG_DIR, CKPT_DIR
from src.base import BaseClass
from src.datasets import (RaceDataset,
						  DreamDataset,
						  SquadDataset,
						  HotpotqaDataset,
						  MusiqueDataset,
						  TriviaqaDataset,
						  )
from src.models import (RobertaLargeFinetunedRace,
						LongformerLarge4096AnsweringRace,
						RobertaBaseSquad2,
						)
from src.tools.easy import save_args

class BasePipeline(BaseClass):

	def __init__(self, **kwargs):
		super(BasePipeline, self).__init__(**kwargs)


	# Inference generator
	# @param datasets_class_name: Str, CLASS name defined in `src.datasets`, e.g. "RaceDataset"
	# @param models_class_name: Str, CLASS name defined in `src.models`, e.g. "RobertaLargeFinetunedRace"
	# @param batch_size: Int, Input batch size
	# @param dataset_kwargs: Dict, keyword arguments for `dataset.yield_batch` (except for `batch_size`)
	# @param model_kwargs: Dict, keyword arguments for `model.generate_model_inputs`
	def easy_inference_pipeline(self,
								dataset_class_name,
								model_class_name,
								batch_size,
								dataset_kwargs,
								model_kwargs,
								):
		Dataset = eval(dataset_class_name)
		Model = eval(model_class_name)
		dataset = Dataset(data_dir = DATA_SUMMARY[Dataset.dataset_name]["path"])
		model = Model(model_path = MODEL_SUMMARY[Model.model_name]["path"])
		for i, batch in enumerate(dataset.yield_batch(batch_size, **dataset_kwargs)):
			yield batch, model.forward(batch, **model_kwargs)

	def easy_finetune_pipeline(self,
							   dataset_class_name,
							   model_class_name,
							   batch_size,
							   dataset_kwargs,
							   model_kwargs,
							   ):
		NotImplemented

class ExtractivePipeline(BasePipeline):
	
	def __init__(self, **kwargs):
		super(ExtractivePipeline, self).__init__(**kwargs)

	# Inference generator
	# @param datasets_class_name: Str, CLASS name defined in `src.datasets`, e.g. "RaceDataset"
	# @param models_class_name: Str, CLASS name defined in `src.models`, e.g. "RobertaLargeFinetunedRace"
	# @param batch_size: Int, Input batch size
	# @param dataset_kwargs: Dict, keyword arguments for `dataset.yield_batch` (except for `batch_size`)
	# @param model_kwargs: Dict, keyword arguments for `model.generate_model_inputs`
	def easy_inference_pipeline(self,
								dataset_class_name,
								model_class_name,
								batch_size,
								dataset_kwargs,
								model_kwargs,
								):
		for model_outputs in super(ExtractivePipeline, self).easy_inference_pipeline(
			dataset_class_name,
			model_class_name,
			batch_size,
			dataset_kwargs,
			model_kwargs,
		):
			batch, (batch_start_logits, batch_end_logits, batch_predicts, batch_input_tokens) = model_outputs
			for data, start_logits, end_logits, predict, input_tokens in zip(batch, batch_start_logits, batch_end_logits, batch_predicts, batch_input_tokens):
				print(list(filter(lambda _token: _token != "<pad>", input_tokens)))
				print(data["context"])
				print(data["question"])
				print(data["answers"])
				print(predict)
				# print(start_logits, start_logits.size())
				# print(end_logits, end_logits.size())
				print("#" * 32)
				input()
				

class GenerativePipeline(BasePipeline):
	
	def __init__(self, **kwargs):
		super(GenerativePipeline, self).__init__(**kwargs)

class MultipleChoicePipeline(BasePipeline):
	
	def __init__(self, **kwargs):
		super(MultipleChoicePipeline, self).__init__(**kwargs)


	# Inference generator
	# @param datasets_class_name: Str, CLASS name defined in `src.datasets`, e.g. "RaceDataset"
	# @param models_class_name: Str, CLASS name defined in `src.models`, e.g. "RobertaLargeFinetunedRace"
	# @param batch_size: Int, Input batch size
	# @param dataset_kwargs: Dict, keyword arguments for `dataset.yield_batch` (except for `batch_size`)
	# @param model_kwargs: Dict, keyword arguments for `model.generate_model_inputs`
	def easy_inference_pipeline(self,
								dataset_class_name,
								model_class_name,
								batch_size,
								dataset_kwargs,
								model_kwargs,
								):
		for model_outputs in super(MultipleChoicePipeline, self).easy_inference_pipeline(
			dataset_class_name,
			model_class_name,
			batch_size,
			dataset_kwargs,
			model_kwargs,
		):
			batch, (batch_logits, batch_predicts) = model_outputs
			for data, logits, predict in zip(batch, batch_logits, batch_predicts):
				print(data["article"])
				print(data["question"])
				print(data["options"])
				print(data["answer"])
				print(predict)
				print("#" * 32)
				input()
				

