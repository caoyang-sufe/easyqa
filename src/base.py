# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import re

class BaseClass:
	regexes = {"forbidden_filename_char": re.compile(r"\\|/|:|\?|\*|\"|<|>|\|")}
	
	def __init__(self, **kwargs):
		for key, word in kwargs.items():
			self.__setattr__(key, word)
