# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@163.sufe.edu.cn
# Testscripts for src.module.dcmn_module

import torch
from torch.nn import functional as F

from src.modules import DCMN
from src.modules.easy import load_model
from src.tests.modules.easy import summary

def testscript():
	from configs import ModuleConfig
	from settings import DEVICE, MODEL_SUMMARY
	from src.tools.easy import load_args, update_args
	N_CHOICES = 4
	args = load_args(Config=ModuleConfig)
	kwargs = {"load_pretrained_model_in_module"		: False,
			  "pretrained_model_device"				: "cpu",
			  "train_batch_size"					: 8,
			  "max_article_sentence"				: 4,
			  "max_article_sentence_token"			: 32,
			  "max_question_token"					: 16,
			  "max_option_token"					: 24,
			  "dcmn_scoring_method"					: "cosine",	# Change as "bilinear"
			  "dcmn_num_passage_sentence_selection"	: 2,
			  "dcmn_pretrained_model"				: "albert-base-v1",
			  "dcmn_encoding_size"					: 768,
			  }
	update_args(args, **kwargs)

	# Generate a small input for quick test
	P_size = (args.train_batch_size * args.max_article_sentence, args.max_article_sentence_token)
	Q_size = (args.train_batch_size, args.max_question_token)   
	A_size = (args.train_batch_size * N_CHOICES, args.max_option_token)
	test_input = {'P'	: {"input_ids"		: (torch.randn(*P_size).abs() * 10).long(),
						   "token_type_ids"	: torch.zeros(*P_size).long(),
						   "attention_mask"	: torch.ones(*P_size).long()},
				  'Q'	: {"input_ids"		: (torch.randn(*Q_size).abs() * 10).long(),
						   "token_type_ids"	: torch.zeros(*Q_size).long(),
						   "attention_mask"	: torch.ones(*Q_size).long()},
				  'A'	: {"input_ids"		: (torch.randn(*A_size).abs() * 10).long(),
						   "token_type_ids"	: torch.zeros(*A_size).long(),
						   "attention_mask"	: torch.ones(*A_size).long()},
				  }
	dcmn = DCMN(args=args).to(DEVICE)
	
	if args.load_pretrained_model_in_module:
		pretrained_model = None
	else:
		pretrained_model = load_model(
			model_path=MODEL_SUMMARY[args.dcmn_pretrained_model]["path"],
			device=args.pretrained_model_device,
		)
		pretrained_model.eval()
	
	dcmn_output = dcmn.forward(**test_input, pretrained_model=pretrained_model)
	print(dcmn_output.size())
	summary(dcmn, num_blank_of_param_name=45)
