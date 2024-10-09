# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from transformers import AutoTokenizer, AutoModel

from src.models.base import GenerativeModel

class Chatglm(GenerativeModel):
	# https://huggingface.co/THUDM/chatglm-6b
	# https://huggingface.co/THUDM/chatglm-6b-int4
	# https://huggingface.co/THUDM/chatglm-6b-int4-qe
	# https://huggingface.co/THUDM/chatglm-6b-int8
	# Note:
	# - The series of models cannot run on CPU
	# - You can quantize with `model = model.quantize(4)` or `model = model.quantize(8)` for low GPU memory
	# Tokenizer = AutoTokenizer
	# Model = AutoModel
	model_name = "THUDM/chatglm"
	def __init__(self, model_path, device):
		super(Chatglm, self).__init__(model_path, device)

	def load_tokenizer(self):
		self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)

	def load_model(self):
		self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True).half().to(self.device)	


class Chatglm6bInt4(Chatglm):
	model_name = "THUDM/chatglm-6b-int4"
	def __init__(self, model_path, device="cuda"):
		super(Chatglm6bInt4, self).__init__(model_path, device)

class Chatglm26bInt4(Chatglm):
	model_name = "THUDM/chatglm2-6b-int4"
	def __init__(self, model_path, device="cuda"):
		super(Chatglm6bInt4, self).__init__(model_path, device)

	