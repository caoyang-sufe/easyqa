# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import argparse

class BaseConfig:
    parser = argparse.ArgumentParser("--")
    parser.add_argument("--", default="", type=str, help="")

