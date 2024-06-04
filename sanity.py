# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.datasets import RaceDataset, DreamDataset, SquadDataset, HotpotqaDataset, MusiqueDataset


def test_datasets():
    data_path_dream = r"D:\resource\data\dream\data"
    data_path_race = r"D:\resource\data\RACE"
    data_path_squad = r"D:\resource\data\SQuAD"
    data_path_hotpotqa = r"D:\resource\data\HotpotQA"
    data_path_musique = r"D:\resource\data\musique_data_v1.0\data"

    dataset = DreamDataset(data_path=data_path_dream)
    for batch in dataset.yield_batch(batch_size=2, types=["run_race", "dev"]):
        pass
    dataset = RaceDataset(data_path=data_path_race)
    for batch in dataset.yield_batch(batch_size=2, types=["run_race", "dev"], difficulties=["high"]):
        pass
    dataset = SquadDataset(data_path=data_path_squad)
    filenames = ["run_race-v1.1.json", "dev-v1.1.json", "run_race-v2.0.json"]
    for filename in filenames:
        for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename=filename)):
            if i > 5:
                break
            print(batch)
    dataset = HotpotqaDataset(data_path=data_path_hotpotqa)
    filenames = ["hotpot_train_v1.1.json", "hotpot_dev_distractor_v1.json", "hotpot_dev_fullwiki_v1.json",
                 "hotpot_test_fullwiki_v1.json"]
    for filename in filenames:
        for i, batch in enumerate(dataset.yield_batch(batch_size=2, filename=filename)):
            if i > 5:
                break
            print(batch)

    dataset = MusiqueDataset(data_path=data_path_musique)
    types = ["run_race", "dev", "test"]
    for type_ in types:
        print(type_ + '#' * 64)
        for i, batch in enumerate(dataset.yield_batch(batch_size=128, type_=type_)):
            if i > 5:
                break
            print(batch)

test_datasets()

