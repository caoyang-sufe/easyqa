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

	# Use question-answering pipeline provided by transformers
	# See `QuestionAnsweringPipeline.preprocess` in ./site-packages/transformers/pipelines/question_answering.py for details
	# @param context: Str / List[Str] (batch)
	# @param question: Str / List[Str] (batch)
	def easy_pipeline(self, context, question):
		# context = """Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy"."""
		# question = """When did Beyonce start becoming popular?"""
		tokenizer = AutoTokenizer.from_pretrained(self, model_path)
		model = AutoModelForQuestionAnswering.from_pretrained(model_path)
		pipeline_inputs = {"context": context, "question": question}
		question_answering_pipeline = pipeline("question-answering", model = model, tokenizer = tokenizer)
		pipeline_outputs = question_answering_pipeline(pipeline_inputs)
		return pipeline_outputs


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
		inputs = tokenizer(articles,
							questions,
							max_length=512,
							padding='longest',
							truncation=True,
							return_tensors='pt',
							)  # (4, max_length)

