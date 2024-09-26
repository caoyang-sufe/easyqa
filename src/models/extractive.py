# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import logging
from transformers import (pipeline,
						  AutoTokenizer,
						  AutoModelForQuestionAnswering,
						  )
from src.models.base import ExtractiveModel

 
class RobertaBaseSquad2(ExtractiveModel):
	# https://huggingface.co/deepset/roberta-base-squad2
	Tokenizer = AutoTokenizer
	Model = AutoModelForQuestionAnswering
	model_name = "deepset/roberta-base-squad2"
	def __init__(self, model_path, device="cpu"):
		super(RobertaBaseSquad2, self).__init__(model_path, device)


class RobertaBaseFinetunedHotpotqa(ExtractiveModel):
	# https://huggingface.co/vish88/roberta-base-finetuned-hotpot_qa
	Tokenizer = AutoTokenizer
	Model = AutoModelForQuestionAnswering
	model_name = "vish88/roberta-base-finetuned-hotpot_qa"
	def __init__(self, model_path, device="cpu"):
		super(RobertaBaseFinetunedHotpotqa, self).__init__(model_path, device)	


class XLNetBaseCasedFinetunedHotpotqa(ExtractiveModel):
	# https://huggingface.co/vish88/xlnet-base-cased-finetuned-hotpot_qa
	Tokenizer = AutoTokenizer
	Model = AutoModelForQuestionAnswering
	model_name = "vish88/xlnet-base-cased-finetuned-hotpot_qa"
	def __init__(self, model_path, device="cpu"):
		super(XLNetBaseCasedFinetunedHotpotqa, self).__init__(model_path, device)	
	

class RobertaBasePFHotpotqa(ExtractiveModel):
	# Model: https://huggingface.co/roberta-base
	# Adapter: https://huggingface.co/AdapterHub/roberta-base-pf-hotpotqa
	Tokenizer = AutoTokenizer
	Model = AutoModelForQuestionAnswering
	model_name = "AdapterHub/roberta-base-pf-hotpotqa"
	adapter_name = "AdapterHub/roberta-base-pf-hotpotqa"
	def __init__(self, model_path, adapter_path, device="cpu"):
		super(RobertaBaseSquad2, self).__init__(model_path, device, adapter_path=adapter_path)
	
	# 2024/09/13 17:34:56
	# Note that `adapter-transformers` package is deprecated and is replaced by `adapters` package
	# They are compatible with different version of `transformers` package
	# Here we try using `adapters` first, if error, then `adapter-transformers` and warning information will be thrown
	def load_model(self):
		try:
			from adapters import AutoAdapterModel
			self.model = AutoAdapterModel.from_pretrained(model_path)
			self.adapter_name = model.load_adapter(adapter_path)
			self.model.set_active_adapters(adapter_name)
		except:
			logging.warning("The `adapter-transformers` package is deprecated "
							"and replaced by the `adapters` package. "
							"See https://docs.adapterhub.ml/transitioning.html."
							)
			from transformers import AutoModelWithHeads
			self.model = AutoModelWithHeads.from_pretrained(self.model_path).to(self.device)
			adapter = self.model.load_adapter(self.adapter_path, source="hf")
			self.model.activate_adapters = adapter
