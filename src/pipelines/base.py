# -*- coding: utf-8 -*- 
# @author : caoyang
# @email: caoyang@stu.sufe.edu.cn

import time

import torch
from torch.optim import Adam, lr_scheduler

from settings import LOG_DIR
from src.base import BaseClass
from src.tools.easy import save_args

class BasePipeline(BaseClass):

    def __init__(self, **kwargs):
        super(BasePipeline, self).__init__(**kwargs)

    def train_baseline(self,
                       args,
                       baseline_class,
                       data_name,
                       model_name,
                       ckpt_path=None,
                       ):
        # Global variables
        time_string = time.strftime("%Y-%m-%d-%H-%M-%S")
        train_log = {"epoch": list(), "iteration": list(), "loss": list(), "accuracy": list()}
        dev_log = {"epoch": list(), "accuracy": list()}
        log_path = os.path.join(LOG_DIR, f"{data_name}-{model_name}-{time_string}.log")
        train_log_path = os.path.join(LOG_DIR, f"train-{time_string}.csv")
        dev_log_path = os.path.join(LOG_DIR, f"dev-{time_string}.csv")
        config_path = os.path.join(LOG_DIR, f"config-{data_name}-{model_name}-{time_string}.json")
        save_args(args=args, save_path=config_path)

        # Load model
        logger = initialize_logger(filename=log_path, mode='w')
        logger.info(f"Arguments: {vars(args)}")
        logger.info(f"Using {args.device}")
        logger.info(f"Cuda Available: {torch.cuda.is_available()}")
        logger.info(f"Available devices: {torch.cuda.device_count()}")
        logger.info(f"Load model with pretrained language model {model_name} ...")
        model = eval(baseline_class)(pretrained_model_name_or_path=MODEL_SUMMARY[model_name]["path"],
                                     device=args.device,
                                     dropout_rate=args.dropout_rate,
                                     max_length=args.max_length,
                                     ).to(args.device)
        logger.info(f"Configure optimizer {args.optimizer} ...")
        optimizer = eval(args.optimizer)(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_multiplier)

        # Load checkpoint(Optional)
        current_epoch = 0
        if ckpt_path is not None:
            logger.info(f"Load checkpoint from {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=torch.device(DEVICE))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            step_lr_scheduler.load_state_dict(checkpoint["scheduler"])
            current_epoch = checkpoint["epoch"] + 1
            train_log = checkpoint["train_log"]
            dev_log = checkpoint["dev_log"]
        logger.info(f"Start from epoch {current_epoch}")

        # Train model
        for epoch in range(current_epoch, args.n_epochs):
            # Train
            model.train()
            data_path = DATA_SUMMARY[data_name]
            data_name_lower = data_name.lower()
            train_dataloader = generate_dataloader(data_name=data_name, types=["train"],
                                                   batch_size=args.train_batch_size)
            for iteration, train_batch_data in enumerate(train_dataloader):
                loss, train_accuracy = model(train_batch_data, mode="train")
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                logger.info(
                    f"Train epoch {epoch} | iter: {iteration} - loss: {loss.item()} - acc: {train_accuracy}")
                train_log["epoch"].append(epoch)
                train_log["iteration"].append(iteration)
                train_log["loss"].append(loss)
                train_log["accuracy"].append(train_accuracy)
            step_lr_scheduler.step()

            # Save checkpoint
            if (epoch + 1) % args.ckpt_cycle == 0:
                checkpoint = {"model"	: model.state_dict(),
                              "optimizer"	: optimizer.state_dict(),
                              "scheduler"	: step_lr_scheduler.state_dict(),
                              "epoch"		: epoch,
                              "train_log"	: train_log,
                              "dev_log"		: dev_log,
                              }
                torch.save(checkpoint, os.path.join(CKPT_DIR, f"dev-{data_name}-{model_name}-{time_string}-{epoch}.ckpt"))

            # Evaluate
            model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                dev_dataloader = generate_dataloader(data_name=data_name, types=["train"], batch_size=args.dev_batch_size)
                for i, dev_batch_data in enumerate(dev_dataloader):
                    correct_size, batch_size = model(dev_batch_data, mode="dev")
                    correct += correct_size
                    total += batch_size
            dev_accuracy = correct / total
            dev_log["epoch"].append(epoch)
            dev_log["accuracy"].append(dev_accuracy)
            logger.info(f"Eval epoch {epoch} | correct: {correct} - total: {total} - acc: {dev_accuracy}")

        # Export log
        train_log_dataframe = pandas.DataFrame(train_log, columns=list(train_log.keys()))
        train_log_dataframe.to_csv(train_log_path, header=True, index=False, sep='\t')
        dev_log_dataframe = pandas.DataFrame(dev_log, columns=list(dev_log.keys()))
        dev_log_dataframe.to_csv(dev_log_path, header=True, index=False, sep='\t')
        terminate_logger(logger)
