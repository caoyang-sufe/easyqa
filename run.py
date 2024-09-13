# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

from src.datasets import RaceDataset, DreamDataset
from src.models.multiple_choice import RobertaLargeFinetunedRace
from src.pipelines import MultipleChoicePipeline
from settings import DATA_SUMMARY, MODEL_SUMMARY

pipeline = MultipleChoicePipeline()

# # Success
# kwargs = {
#     "Model": RobertaLargeFinetunedRace,
#     "data_path": DATA_SUMMARY["RACE"]["path"],
#     "model_path": MODEL_SUMMARY["LIAMF-USP/roberta-large-finetuned-race"]["path"],
#     "max_length": 512,
#     "device": "cpu",
# }
#
# pipeline.run_race(**kwargs)
# ----
# # Success
# kwargs = {
#     "Model": RobertaLargeFinetunedRace,
#     "data_path": DATA_SUMMARY["DREAM"]["path"],
#     "model_path": MODEL_SUMMARY["LIAMF-USP/roberta-large-finetuned-race"]["path"],
#     "max_length": 512,
#     "device": "cpu",
# }
# pipeline.run_dream(**kwargs)
# ----

kwargs = {
    "Model": RobertaLargeFinetunedRace,
    "data_path": DATA_SUMMARY["DREAM"]["path"],
    "model_path": MODEL_SUMMARY["LIAMF-USP/roberta-large-finetuned-race"]["path"],
    "max_length": 512,
    "device": "cpu",
}
pipeline.run_dream(**kwargs)