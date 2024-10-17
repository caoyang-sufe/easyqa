# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import argparse

from copy import deepcopy


class BaseConfig:
	parser = argparse.ArgumentParser('--')
	parser.add_argument("--device", default="cpu", type=str, help="Device to run on")

	# Default config
	parser.add_argument('--default_pretrained_model', default='albert-base-v1', type=str, help='Default pretrained model, see setting.PRETRAINED_MODEL_SUMMARY for details.')
	parser.add_argument('--default_wordvector', default='glove-6B-50d', type=str, help='Default wordvector model, see setting.WORDVECTOR_MODEL_SUMMARY for details.')
	parser.add_argument('--default_embedding_size', default=50, type=int, help='Default embedding size for token')
	
	# Params of src.tool.parser_model_tool
	parser.add_argument('--word_tokenizer_source', default='nltk', type=str, help='@param source of src.tool.parser_model_tool.load_word_tokenizer')
	parser.add_argument('--sentence_tokenizer_source', default='nltk', type=str, help='@param source of src.tool.parser_model_tool.load_sentence_tokenizer')
	parser.add_argument('--parse_tree_parser_source', default='stanford', type=str, help='@param source of src.tool.parser_model_tool.load_parse_tree_parser')
	parser.add_argument('--dependency_parser_source', default='stanford', type=str, help='@param source of src.tool.parser_model_tool.load_dependency_parser')
	
	# Padding length (Default for RACE)
	parser.add_argument('--max_article_token', default=1440, type=int, help='The max number of tokens in one article, see summary.ipynb in /summary for detail.')
	parser.add_argument('--max_article_sentence', default=128, type=int, help='The max number of sentences in one article, see summary.ipynb in /summary for detail.')
	parser.add_argument('--max_article_sentence_token', default=192, type=int, help='The max number of tokens in one article sentence, see summary.ipynb in /summary for detail.')
	parser.add_argument('--max_question_token', default=128, type=int, help='The max number of tokens in one question, see summary.ipynb in /summary for detail.')
	parser.add_argument('--max_option_token', default=128, type=int, help='The max number of tokens in one option, see summary.ipynb in /summary for detail.')

	# Some commonly used config
	parser.add_argument('--n_choices', default=4, type=bool, help='The number of choices in one question. e.g. 4 for RACE and 3 for DREAM.')
	parser.add_argument('--multi_choice', default=False, type=bool, help='Multi-choice question or Single-choice question')
	parser.add_argument('--test_while_train', default=False, type=bool, help='If there are annotated test dataset, you can directly evaluate on it while training.')
	parser.add_argument('--comatch_wordvector', default='glove-6B-300d', help='Wordvector model used in Co-Matching')
	parser.add_argument('--dcmn_pretrained_model', default='bert-base-uncased', type=str, help='Pretrained model of DCMN+, which is BERT in original paper. See setting.PRETRAINED_MODEL_SUMMARY for more choices.')
	parser.add_argument('--duma_pretrained_model', default='bert-base-uncased', type=str, help='Pretrained model of DUMA+, which is BERT in original paper. See setting.PRETRAINED_MODEL_SUMMARY for more choices.')
	parser.add_argument('--hrca_pretrained_model', default='bert-base-uncased', type=str, help='Pretrained model of DUMA, which is BERT in original paper. See setting.PRETRAINED_MODEL_SUMMARY for more choices.')
	parser.add_argument('--use_pretrained_model', default=True, type=bool, help='Whether to use pretrained model.')
	parser.add_argument('--load_pretrained_model_in_module', default=False, type=bool, help='Define pretrained model in module or just pass it as a argument.')
	parser.add_argument('--pretrained_model_device', default='cpu', type=str, help='You can run pretrained model on cpu for usually it is very large.')


class EasyTrainConfig:
	parser = deepcopy(BaseConfig.parser)
	parser.add_argument("--lr", default=0.01, type=float, help="Learning rate")
	parser.add_argument("--wd", default=0.9, type=float, help="Weight decay")
	parser.add_argument("--lrs", default=0.9, type=float, help="Step size of learning rate scheduler")
	parser.add_argument("--lrm", default=0.9, type=float, help="Learning rate multiplier")
	parser.add_argument("--ep", default=32, type=float, help="Train epochs")


class ModuleConfig:
	parser = deepcopy(BaseConfig.parser)
	# Model training config
	parser.add_argument('--n_epochs', default=32, type=int, help='The number of training epochs')
	parser.add_argument('--optimizer', default='Adam', type=str, help='Optimizer defined in torch.nn module')
	parser.add_argument('--learning_rate', default=.01, type=float, help='@param lr of torch.optim.Optimizer')
	parser.add_argument('--weight_decay', default=.0, type=float, help='@param weight_decay of torch.optim.Optimizer')
	parser.add_argument('--lr_step_size', default=1, type=float, help='@param step_size of torch.optim.lr_scheduler.StepLR')
	parser.add_argument('--lr_multiplier', default=.95, type=float, help='@param gamma of torch.optim.lr_scheduler.StepLR')
	parser.add_argument('--save_checkpoint_per_epoch', default=1, type=int, help='Save checkpoint every ? epoch while training')
	# Co-Matching Config (using the default settings in original paper)
	parser.add_argument('--comatch_embedding_size', default=300, type=int, help='Wordvector embedding size of Co-Matching, matching with config --comatch_wordvector')
	parser.add_argument('--comatch_bilstm_hidden_size', default=150, type=int, help='@param hidden_size of torch.nn.LSTM in Co-Matching sequence encoder')
	parser.add_argument('--comatch_bilstm_num_layers', default=1, type=int, help='@param num_layers of torch.nn.LSTM in Co-Matching sequence encoder')
	parser.add_argument('--comatch_bilstm_bidirectional', default=True, type=bool, help='@param bidirectional of torch.nn.LSTM in Co-Matching sequence encoder')
	parser.add_argument('--comatch_bilstm_dropout', default=.3, type=bool, help='@param dropout of torch.nn.LSTM in Co-Matching sequence encoder')
	# DCMN+ Config
	parser.add_argument('--dcmn_scoring_method', default='cosine', type=str, help='Scoring method for passage sentence selection in DCMN+, which should be cosine or bilinear. See https://arxiv.org/abs/1908.11511 for details.')
	parser.add_argument('--dcmn_num_passage_sentence_selection', default=2, type=int, help='The number of sentences selected in passage sentence selection, default 2 in paper.')
	parser.add_argument('--dcmn_encoding_size', default=768, type=int, help='Sequence contextualized encoding size of DCMN+, usually refer to hidden size of pretrained model, e.g. 768 for BERT.')
	# DUMA Config
	parser.add_argument('--duma_num_layers', default=1, type=int, help='How many DUMA layers stacked in model, default 1 in origin paper for simplicity.')
	parser.add_argument('--duma_encoding_size', default=768, type=int, help='Pretrained encoding size of DUMA, usually refer to hidden size of pretrained model, e.g. 768 for BERT.')
	parser.add_argument('--duma_fuse_method', default=None, type=str, help='Fuse method of DUMA, which is one of {element-wise production, element-wise summation, concatenation} in origin paper, here we use {mul, sum, cat} for simplicity. If set None, then a DIY fuse function will be applyed.')
	parser.add_argument('--duma_mha_num_heads', default=8, type=int, help='The number of heads in DUMA MultiHeadAttention module.')
	parser.add_argument('--duma_mha_dropout_rate', default=0., type=float, help='The dropout rate in DUMA MultiHeadAttention module.')
	# HRCA Config
	parser.add_argument('--hrca_num_layers', default=1, type=int, help='How many HRCA layers stacked in model, default 1 in origin paper for simplicity.')
	parser.add_argument('--hrca_encoding_size', default=768, type=int, help='Pretrained encoding size of HRCA, usually refer to hidden size of pretrained model, e.g. 768 for BERT.')
	parser.add_argument('--hrca_fuse_method', default='cat', type=str, help='Fuse method of DUMA, which is one of {element-wise multiplication, element-wise summation, concatenation} in origin paper, here we use {mul, sum, cat} for simplicity. If set None, then a DIY fuse function will be applyed.')
	parser.add_argument('--hrca_mha_num_heads', default=8, type=int, help='The number of heads in HRCA MultiHeadAttention module.')
	parser.add_argument('--hrca_mha_dropout_rate', default=0., type=float, help='The dropout rate in HRCA MultiHeadAttention module.')
	parser.add_argument('--hrca_plus', default=False, type=bool, help='Whether to use HRCA+ forward propagation, HRCA+ differ from HRCA in how many attention used in POQ matrix, where HRCA+ uses 9 while HRCA uses 3.')

