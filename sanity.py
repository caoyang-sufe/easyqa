# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import gc
import torch

from settings import DATA_DIR, LOG_DIR, MODEL_ROOT, DATA_SUMMARY, MODEL_SUMMARY

from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset, TriviaqaDataset
from src.models import RobertaLargeFinetunedRace, LongformerLarge4096AnsweringRace, RobertaBaseSquad2, Chatglm6bInt4
from src.pipelines import RacePipeline, DreamPipeline, SquadPipeline
from src.tools.easy import initialize_logger, terminate_logger

def test_yield_batch():
	# data_dir = r"D:\data"	# Lab PC
	# data_dir = r"D:\resource\data"	# Region Laptop
	data_dir = DATA_DIR	# default
	data_dir_race = DATA_SUMMARY["RACE"]["path"]
	data_dir_dream = DATA_SUMMARY["DREAM"]["path"]
	data_dir_squad = DATA_SUMMARY["SQuAD"]["path"]
	data_dir_hotpotqa = DATA_SUMMARY["HotpotQA"]["path"]
	data_dir_musique = DATA_SUMMARY["Musique"]["path"]
	data_dir_triviaqa = DATA_SUMMARY["TriviaQA"]["path"]
		
	# RACE
	def _test_race():
		print(_test_race.__name__)
		dataset = RaceDataset(data_dir=data_dir_race)
		for batch in dataset.yield_batch(batch_size=2, types=["train", "dev"], difficulties=["high"]):
			pass
	# DREAM
	def _test_dream():
		print(_test_dream.__name__)
		dataset = DreamDataset(data_dir=data_dir_dream)
		for batch in dataset.yield_batch(batch_size=2, types=["train", "dev"]):
			pass
	# SQuAD
	def _test_squad():
		print(_test_squad.__name__)
		dataset = SquadDataset(data_dir=data_dir_squad)
		versions = ["1.1"]
		types = ["train", "dev"]
		for version in versions:
			for type_ in types:
				for i, batch in enumerate(dataset.yield_batch(batch_size=2, type_=type_, version=version)):
					if i > 5:
						break
					print(batch)
	# HotpotQA
	def _test_hotpotqa():
		print(_test_hotpotqa.__name__)
		dataset = HotpotqaDataset(data_dir=data_dir_hotpotqa)
		filenames = ["hotpot_train_v1.1.json",
					 "hotpot_dev_distractor_v1.json",
					 "hotpot_dev_fullwiki_v1.json",
					 "hotpot_test_fullwiki_v1.json",
					 ]
		for filename in filenames:
			for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename=filename)):
				if i > 5:
					break
				print(batch)
	# Musique
	def _test_musique():
		print(_test_musique.__name__)
		batch_size = 2
		dataset = MusiqueDataset(data_dir=data_dir_musique)
		types = ["train", "dev", "test"]
		categories = ["ans", "full"]
		answerables = [True, False]
		for type_ in types:
			for category in categories:
				if category == "full":
					for answerable in answerables:
						print(f"======== {type_} - {category} - {answerable} ========")
						for i, batch in enumerate(dataset.yield_batch(batch_size, type_, category, answerable)):
							if i > 5:
								break
							print(batch)
				else:
					print(f"======== {type_} - {category} ========")
					for i, batch in enumerate(dataset.yield_batch(batch_size, type_, category)):
						if i > 5:
							break
						print(batch)				
								
	# TriviaQA
	def _test_triviaqa():
		print(_test_triviaqa.__name__)
		batch_size = 2
		dataset = TriviaqaDataset(data_dir=data_dir_triviaqa)
		types = ["verified", "train", "dev", "test"]
		categories = ["web", "wikipedia"]
		for type_ in types:
			for category in categories:
				print(f"======== {type_} - {category} ========")
				for i, batch in enumerate(dataset.yield_batch(batch_size, type_, category, False)):
					if i > 5:
						break
					print(batch)	
		gc.collect()
		for type_ in ["train", "dev", "test"]:
			print(f"======== {type_} - unfiltered ========")
			for i, batch in enumerate(dataset.yield_batch(batch_size, type_, "web", True)):
				if i > 5:
					break
				print(batch)

	# Test		
	logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	# _test_race()
	# _test_dream()
	# _test_squad()
	_test_hotpotqa()
	# _test_musique()
	# _test_triviaqa()
	terminate_logger(logger)


def test_generate_model_inputs():
	
	def _test_race():
		print(_test_race.__name__)
		data_dir = DATA_SUMMARY[RaceDataset.dataset_name]["path"]
		model_path = MODEL_SUMMARY[RobertaLargeFinetunedRace.model_name]["path"]
		# model_path = MODEL_SUMMARY[LongformerLarge4096AnsweringRace.model_name]["path"]
		dataset = RaceDataset(data_dir)
		model = RobertaLargeFinetunedRace(model_path, device="cpu")
		# model = LongformerLarge4096AnsweringRace(model_path, device="cpu")

		for i, batch in enumerate(dataset.yield_batch(batch_size=2, types=["train", "dev"], difficulties=["high"])):
			model_inputs = RaceDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=32)
			print(model_inputs)
			print('-' * 32)
			model_inputs = model.generate_model_inputs(batch, max_length=32)
			print(model_inputs)
			print('#' * 32)
			if i > 5:
				break

	def _test_dream():
		print(_test_dream.__name__)
		data_dir = DATA_SUMMARY[DreamDataset.dataset_name]["path"] 
		model_path = MODEL_SUMMARY[RobertaLargeFinetunedRace.model_name]["path"]
		dataset = DreamDataset(data_dir)
		model = RobertaLargeFinetunedRace(model_path, device="cpu")
		for i, batch in enumerate(dataset.yield_batch(batch_size=2, types=["train", "dev"])):
			model_inputs = DreamDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=32)
			print(model_inputs)
			print('-' * 32)
			model_inputs = model.generate_model_inputs(batch, max_length=32)
			print(model_inputs)
			print('#' * 32)
			if i > 5:
				break

	def _test_squad():
		print(_test_squad.__name__)
		data_dir = DATA_SUMMARY[SquadDataset.dataset_name]["path"]
		model_path = MODEL_SUMMARY[RobertaBaseSquad2.model_name]["path"]
		dataset = SquadDataset(data_dir)
		model = RobertaBaseSquad2(model_path, device="cpu")

		for i, batch in enumerate(dataset.yield_batch(batch_size=2, type_="dev", version="1.1")):
			model_inputs = SquadDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=32)
			print(model_inputs)
			print('-' * 32)
			model_inputs = model.generate_model_inputs(batch, max_length=32)
			print(model_inputs)
			print('#' * 32)
			if i > 5:
				break

	def _test_hotpotqa():
		print(_test_hotpotqa.__name__)
		data_dir = DATA_SUMMARY[HotpotqaDataset.dataset_name]["path"]
		model_path = MODEL_SUMMARY[Chatglm6bInt4.model_name]["path"]
		dataset = HotpotqaDataset(data_dir)
		model = Chatglm6bInt4(model_path, device="cuda")

		for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename="dev_distractor_v1.json")):
			model_inputs = HotpotqaDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=512)
			print(model_inputs)
			print('-' * 32)
			model_inputs = model.generate_model_inputs(batch, max_length=32)
			print(model_inputs)
			print('#' * 32)
			if i > 5:
				break		
	
	logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	# _test_race()
	# _test_dream()
	# _test_squad()
	_test_hotpotqa()
	terminate_logger(logger)


def test_inference_pipeline():

	def _test_race():
		race_pipeline = RacePipeline()
		pipeline = race_pipeline.easy_inference_pipeline(
			dataset_class_name = "RaceDataset",
			model_class_name = "RobertaLargeFinetunedRace",
			batch_size = 2,
			dataset_kwargs = {"types": ["train"], "difficulties": ["high", "middle"]},
			model_kwargs = {"max_length": 512},
		)
		
	def _test_squad():
		squad_pipeline = SquadPipeline()
		pipeline = squad_pipeline.easy_inference_pipeline(
			dataset_class_name = "SquadDataset",
			model_class_name = "RobertaBaseSquad2",
			batch_size = 2,
			dataset_kwargs = {"type_": "train", "version": "2.0"},
			model_kwargs = {"max_length": 512},
		)

	# logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	_test_race()
	# _test_squad()
	# terminate_logger(logger)


def test_pipeline():

	from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
	from settings import MODEL_SUMMARY
	context = 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".'
	question = 'When did Beyonce start becoming popular?'
	model_path = MODEL_SUMMARY["deepset/roberta-base-squad2"]["path"]
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	model = AutoModelForQuestionAnswering.from_pretrained(model_path)
	inputs = dict(context = context, question = question)
	pipe = pipeline("question-answering", model = model, tokenizer = tokenizer)
	outputs = pipe(inputs)
	print(outputs)

if __name__ == "__main__":
	# test_yield_batch()
	test_generate_model_inputs()
	# test_inference_pipeline()
	# test_pipeline()