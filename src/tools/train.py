# -*- coding: utf-8 -*-
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import time
import json
import torch
import pandas

from torch.nn import CrossEntropyLoss, NLLLoss
from torch.optim import Adam, SGD, lr_scheduler
from torch.utils.data import DataLoader

from src.tools.easy import save_args, update_args, initialize_logger, terminate_logger

# Traditional training pipeline
# @params args: Object of <config.EasytrainConfig>
# @param model: Loaded model of torch.nn.Module
# @param train_dataloader: torch.data
# @param dev_dataloader:
# @param ckpt_epoch:
# @param ckpt_path: 
def easy_train_pipeline(args,
						model,
						train_dataloader,
						dev_dataloader,
						ckpt_epoch = 1,
						ckpt_path = None,
						**kwargs,
						):
	# 1 Global variables
	time_string = time.strftime("%Y%m%d%H%M%S")
	log_name = easy_train_pipeline.__name__
	# 2 Define paths
	train_record_path = os.path.join(LOG_DIR, f"{log_name}_{time_string}_train_record.txt")
	dev_record_path = os.path.join(LOG_DIR, f"{log_name}_{time_string}_dev_record.txt")
	log_path = os.path.join(LOG_DIR, f"{log_name}_{time_string}.log")
	config_path = os.path.join(LOG_DIR, f"{log_name}_{time_string}.cfg")
	# 3 Save arguments
	save_args(args, save_path=config_path)
	logger = initialize_logger(filename=log_path, mode='w')
	logger.info(f"Arguments: {vars(args)}")
	# 4 Load checkpoint
	logger.info(f"Using {args.device}")
	logger.info(f"Cuda Available: {torch.cuda.is_available()}")
	logger.info(f"Available devices: {torch.cuda.device_count()}")
	logger.info(f"Optimizer {args.optimizer} ...")
	current_epoch = 0
	optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=args.wd)
	step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lrs, gamma=args.lrm)
	train_record = {"epoch": list(), "iteration": list(), "loss": list(), "accuracy": list()}
	dev_record = {"epoch": list(), "accuracy": list()}
	if ckpt_path is not None:
		logger.info(f"Load checkpoint from {ckpt_path}")
		checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
		model.load_state_dict(checkpoint["model"])
		optimizer.load_state_dict(checkpoint["optimizer"])
		step_lr_scheduler.load_state_dict(checkpoint["scheduler"])
		current_epoch = checkpoint["epoch"] + 1	# plus one to next epoch
		train_record = checkpoint["train_record"]
		dev_record = checkpoint["dev_record"]
		logger.info("  - ok!")
	logger.info(f"Start from epoch {current_epoch}")
	# 5 Run epochs
	for epoch in range(current_epoch, args.n_epochs):
		## 5.1 Train model
		model.train()
		train_dataloader.reset()	# Reset dev dataloader
		for iteration, train_batch_data in enumerate(train_dataloader):
			loss, train_accuracy = model(train_batch_data, mode="train")
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			logger.info(f"Epoch {epoch} | iter: {iteration} - loss: {loss.item()} - acc: {train_accuracy}")
			train_record["epoch"].append(epoch)
			train_record["iteration"].append(iteration)
			train_record["loss"].append(loss)
			train_record["accuracy"].append(train_accuracy)
		step_lr_scheduler.step()
		## 5.2 Save checkpoint
		if (epoch + 1) % ckpt_epoch == 0:
			checkpoint = {"model": model.state_dict(),
						  "optimizer": optimizer.state_dict(),
						  "scheduler": step_lr_scheduler.state_dict(),
						  "epoch": epoch,
						  "train_record": train_record,
						  "dev_record": dev_record,
						  }
			torch.save(checkpoint, os.path.join(CKPT_DIR, f"dev-{data_name}-{model_name}-{time_string}-{epoch}.ckpt"))
		## 5.3 Evaluate model
		model.eval()
		with torch.no_grad():
			correct = 0
			total = 0
			dev_dataloader.reset()	# Reset dev dataloader
			for iteration, dev_batch_data in enumerate(dev_dataloader):
				correct_size, batch_size = model(dev_batch_data, mode="dev")
				correct += correct_size
				total += batch_size
		dev_accuracy = correct / total
		dev_record["epoch"].append(epoch)
		dev_record["accuracy"].append(dev_accuracy)
		logger.info(f"Eval epoch {epoch} | correct: {correct} - total: {total} - acc: {dev_accuracy}")
	# 7 Export log
	# train_record_save_path = ...
	# dev_record_save_path = ...
	train_record_dataframe = pandas.DataFrame(train_record, columns=list(train_record.keys()))
	train_record_dataframe.to_csv(train_record_save_path, header=True, index=False, sep='\t')
	logger.info(f"Export train record to {train_record_save_path}")
	dev_record_dataframe = pandas.DataFrame(dev_record, columns=list(dev_record.keys()))
	dev_record_dataframe.to_csv(dev_record_save_path, header=True, index=False, sep='\t')
	logger.info(f"Export dev record to {dev_record_save_path}")
	terminate_logger(logger)


