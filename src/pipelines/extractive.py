# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import gc
import time
from transformers import (pipeline,
                          AutoTokenizer,
                          AutoModel,
                          AutoModelForQuestionAnswering,
                          RobertaTokenizer,
                          RobertaForMultipleChoice,
                          LongformerTokenizer,
                          LongformerForMultipleChoice,
                          )
from src.pipelines.base import ExtractivePipeline


class SquadPipeline(ExtractivePipeline):

	def __init__(self):
		super(SquadPipeline, self).__init__()

	def easy_inference_pipeline(self,
								dataset_class_name,
								model_class_name,
								batch_size,
								dataset_kwargs,
								model_kwargs,
								save_path,
								):
		for yield_data in super(SquadPipeline, self).easy_inference_pipeline(
			dataset_class_name,
			model_class_name,
			batch_size,
			dataset_kwargs,
			model_kwargs,
			save_path,
		):
			print(list(filter(lambda _token: _token != "<pad>", input_tokens)))
			print(data["context"])
			print(data["question"])
			print(data["answers"])
			print(predict)
			# print(start_logits, start_logits.size())
			# print(end_logits, end_logits.size())
			print("#" * 32)
			input()
			

class HotpotqaPipeline(ExtractivePipeline):

	def __init__(self):
		super(HotpotqaPipeline, self).__init__()