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

		pass