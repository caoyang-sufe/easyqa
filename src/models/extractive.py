# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

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

	# @param data: Dict[article(List[Str]), question(List[Str])]
	# @return batch_results: List[Str] (batch_size, )
	def run(self, batch, max_length=4096):
		return run_roberta_base_squad2(batch, self.tokenizer, self.model, max_length)


class RobertaBasePFHotpotQA(ExtractiveModel):
	# https://huggingface.co/roberta-base (Model)
	# https://huggingface.co/AdapterHub/roberta-base-pf-hotpotqa (Adapter)
	Tokenizer = AutoTokenizer
	Model = AutoModelForQuestionAnswering
	model_name = "AdapterHub/roberta-base-pf-hotpotqa"
	def run(self, batch):
		return self.run_roberta_base_pf_hotpotqa()


	def run_roberta_base_pf_hotpotqa(self):
		tokenizer = AutoTokenizer.from_pretrained(r"D:\resource\model\huggingface\common\roberta-base")

		articles = "Ben's father is David, they are good friends."
		questions = "Who is Ben's father?"

		inputs_1 = tokenizer(articles + questions,
							 max_length=512,
							 padding='longest',
							 truncation=True,
							 return_tensors='pt',
							 )  # (4, max_length)

		inputs_2 = tokenizer(articles,
							 questions,
							 max_length=512,
							 padding='longest',
							 truncation=True,
							 return_tensors='pt',
							 )  # (4, max_length)

