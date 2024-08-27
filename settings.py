# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import platform

SYSTEM = platform.system()
DATA_ROOT = {"Windows": r"D:\resource\data",
			 "Linux": "/nfsshare/home/caoyang/code/data",
			 }
MODEL_ROOT = {"Windows": r"D:\resource\model\huggingface",
			  "Linux": "/nfsshare/home/caoyang/code/model/huggingface",
			  }
DATA_DIR = DATA_ROOT[SYSTEM]
DATA_SUMMARY = {"RACE": {"path": os.path.join(DATA_DIR, "RACE")},
				"DREAM": {"path": os.path.join(DATA_DIR, "dream", "data")},
				"SQuAD": {"path": os.path.join(DATA_DIR, "SQuAD")},
				"HotpotQA": {"path": os.path.join(DATA_DIR, "HotpotQA")},
				"Musique": {"path": os.path.join(DATA_DIR, "Musique", "data")},
				"TriviaQA": {"path": os.path.join(DATA_DIR, "TQA")},
				}

MODEL_DIR = MODEL_ROOT[SYSTEM]
MODEL_SUMMARY = {"albert-base-v1": {"path": os.path.join(MODEL_DIR, "common", "albert-base-v1")},
				 "albert-large-v1": {"path": os.path.join(MODEL_DIR, "common", "albert-large-v1")},
				 "albert-base-v2": {"path": os.path.join(MODEL_DIR, "common", "albert-base-v2")},
				 "albert-large-v2": {"path": os.path.join(MODEL_DIR, "common", "albert-large-v2")},
				 "bert-base-uncased": {"path": os.path.join(MODEL_DIR, "common", "bert-base-uncased")},
				 "bert-large-uncased": {"path": os.path.join(MODEL_DIR, "common", "bert-large-uncased")},
				 "roberta-base": {"path": os.path.join(MODEL_DIR, "common", "roberta-base")},
				 "roberta-large": {"path": os.path.join(MODEL_DIR, "common", "roberta-large")},
				 "deepset/roberta-base-squad2": {"path": os.path.join(MODEL_DIR, "deepset", "roberta-base-squad2")},
				 "LIAMF-USP/roberta-large-finetuned-race": {"path": os.path.join(MODEL_DIR, "LIAMF-USP", "roberta-large-finetuned-race")},
				 "potsawee/longformer-large-4096-answering-race": {"path": os.path.join(MODEL_DIR, "potsawee", "longformer-large-4096-answering-race")},
				 }




LOG_DIR = "./log"
TEMP_DIR = "./temp"
