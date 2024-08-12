# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset

def test_datasets():
	data_dir = r"D:\data"
	data_path_dream = os.path.join(data_dir, "dream")
	data_path_race = os.path.join(data_dir, "race")
	data_path_squad = os.path.join(data_dir, "squad")
	data_path_hotpotqa = os.path.join(data_dir, "hotpotqa")
	data_path_musique = os.path.join(data_dir, "musique")
		
	# RACE
	def _test_race():
		print(_test_race.__name__)
		dataset = RaceDataset(data_path=data_path_race)
		for batch in dataset.yield_batch(batch_size=2, types=["train", "dev"], difficulties=["high"]):
			pass
	# DREAM
	def _test_dream():
		print(_test_dream.__name__)
		dataset = DreamDataset(data_path=data_path_dream)
		for batch in dataset.yield_batch(batch_size=2, types=["train", "dev"]):
			pass
	# SQuAD
	def _test_squad():
		print(_test_squad.__name__)
		dataset = SquadDataset(data_path=data_path_squad)
		filenames = ["train-v1.1.json", "dev-v1.1.json", "train-v2.0.json"]
		for filename in filenames:
			for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename=filename)):
				if i > 5:
					break
				print(batch)
	# HotpotQA
	def _test_hotpotqa():
		print(_test_hotpotqa.__name__)
		dataset = HotpotqaDataset(data_path=data_path_hotpotqa)
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
		dataset = MusiqueDataset(data_path=data_path_musique)
		types = ["train", "dev", "test"]
		for type_ in types:
			print(type_ + '#' * 64)
			for i, batch in enumerate(dataset.yield_batch(batch_size=128, type_=type_)):
				if i > 5:
					break
				print(batch)

	# _test_race()
	# _test_dream()
	# _test_squad()
	# _test_hotpotqa()
	_test_musique()

if __name__ == "__main__":
	test_datasets()
