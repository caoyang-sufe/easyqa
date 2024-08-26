# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os

from settings import LOG_DIR

from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset
from src.tools.easy import initialize_logger, terminate_logger

def test_datasets():
	# data_dir = r"D:\data"	# Lab PC
	data_dir = r"D:\resource\data"	# Region
	data_dir_dream = os.path.join(data_dir, "dream", "data")
	data_dir_race = os.path.join(data_dir, "race")
	data_dir_squad = os.path.join(data_dir, "squad")
	data_dir_hotpotqa = os.path.join(data_dir, "hotpotqa")
	data_dir_musique = os.path.join(data_dir, "musique", "data")
		
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
		filenames = ["hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json", "hotpot_dev_fullwiki_v1.json",
					 "hotpot_test_fullwiki_v1.json"]
		for filename in filenames:
			for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename=filename)):
				if i > 5:
					break
				print(batch)
	# Musique
	def _test_musique():
		print(_test_musique.__name__)
		dataset = MusiqueDataset(data_dir=data_dir_musique)
		types = ["train", "dev", "test"]
		for type_ in types:
			print(type_ + '#' * 64)
			for i, batch in enumerate(dataset.yield_batch(batch_size=128, type_=type_)):
				if i > 5:
					break
				print(batch)
	logger = initialize_logger(os.path.join(LOG_DIR, "sanity.log"), 'w')
	_test_race()
	_test_dream()
	_test_squad()
	_test_hotpotqa()
	_test_musique()
	terminate_logger(logger)

if __name__ == "__main__":
	test_datasets()
