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
from src.pipelines.base import GenerativePipeline

class ChatglmPipeline(GenerativePipeline):

	def __init__(self):
		super(RacePipeline, self).__init__()

	# @param model		: ChatGLM model
	# @param tokenizer	: ChatGLM tokenizer
	# @param context	: Str, Chat context
	# @param history	: List, Chat history in JSON format
	# @return response	: Str, Chat response
	# @return history	: List, Updated chat history in JSON format
	def easy_chat_pipeline(self,
						   model,
						   context,
						   history = [],
						   ):
		response, history = model.chat(tokenizer, context, history=history)
		return response, history

	def easy_chat_pipeline(self):
		
		# Inference pipeline
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
			for model_outputs in super(ChatglmPipeline, self).easy_inference_pipeline(
				dataset_class_name,
				model_class_name,
				batch_size,
				dataset_kwargs,
				model_kwargs,
			):
		
		pass