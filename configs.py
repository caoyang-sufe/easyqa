# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import argparse

from copy import deepcopy

class BaseConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument("--device", default="cuda", type=str, help="Device to run on")



class EasytrainConfig:

	parser = deepcopy(BaseConfig.parser)

	parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
	parser.add_argument("--wd", default=0.9, type=float, help="Weight decay")
	parser.add_argument("--lrs", default=0.9, type=float, help="Step size of learning rate scheduler")
	parser.add_argument("--lrm", default=0.9, type=float, help="Learning rate multiplier")
	parser.add_argument("--ep", default=32, type=float, help="Train epochs")
