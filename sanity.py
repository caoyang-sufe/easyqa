# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os

import torch

from settings import DATA_DIR, LOG_DIR, MODEL_ROOT, DATA_SUMMARY, MODEL_SUMMARY

from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset
from src.models import RobertaLargeFinetunedRace
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
				for i, batch in enumerate(dataset.yield_batch(batch_size=2, version=version, type_=type_)):
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
								

	# Test		
	logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	# _test_race()
	# _test_dream()
	# _test_squad()
	# _test_hotpotqa()
	_test_musique()
	terminate_logger(logger)


def test_generate_model_inputs():
	
	def _test_race():
		print(_test_race.__name__)
		data_dir = DATA_SUMMARY[RaceDataset.dataset_name]["path"]
		model_path = MODEL_SUMMARY[RobertaLargeFinetunedRace.model_name]["path"]
		dataset = RaceDataset(data_dir)
		model = RobertaLargeFinetunedRace(model_path, device="cpu")

		for i, batch in enumerate(dataset.yield_batch(batch_size=2, types=["train", "dev"], difficulties=["high"])):
			model_inputs = RaceDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=32)
			print(model_inputs)
			print('-' * 64)
			if i > 5:
				break

	def _test_dream():
		print(_test_dream.__name__)
		data_dir = DATA_SUMMARY[DreamDataset.dataset_name]["path"] 
		model_path = MODEL_SUMMARY[RobertaLargeFinetunedRace.model_name]["path"]
		dataset = DreamDataset(data_dir)
		model = RobertaLargeFinetunedRace(model_path, device="cpu")
		for i, batch in enumerate(dataset.yield_batch(batch_size=2, types=["train", "dev"])):
			model_inputs = RaceDataset.generate_model_inputs(batch, model.tokenizer, model.model_name, max_length=32)
			print(model_inputs)
			print('-' * 64)
			if i > 5:
				break
	
	logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	_test_race()
	_test_dream()
	terminate_logger(logger)


if __name__ == "__main__":
	test_yield_batch()
	# test_generate_model_inputs()
