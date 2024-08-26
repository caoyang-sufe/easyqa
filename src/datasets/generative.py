# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json

from src.datasets.base import BaseExtractiveDatasetDataset

class TriviaqaDataset(BaseDataset):
    pipeline_type = "mutiple-choice"
    def __init__(self,
                 data_path,
                 ):
        super(HotpotqaDataset, self).__init__()
        self.data_path = data_path

    # @param batch_size: Int
    # @param filename: 
    # @yield batch:
    def yield_batch(self,
                    batch_size,
                    filename,
                    ):
		pass
