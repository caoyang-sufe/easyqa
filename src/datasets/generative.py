# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseGenerativeDataset

class TriviaqaDataset(BaseGenerativeDataset):
	
	def __init__(self,
				 data_dir,
				 ):
		super(TriviaqaDataset, self).__init__(data_dir)

	# @param batch_size: Int
	# @param filename: 
	# @yield batch:
	def yield_batch(self,
					batch_size,
					filename,
					):
		pass
