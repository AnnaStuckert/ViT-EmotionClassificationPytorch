# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import json
import numpy as np
import torch
import torch.distributed as dist
import pandas as pd
from datetime import timedelta
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.modeling import CONFIGS, VisionTransformer
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.scheduler import WarmupCosineSchedule, WarmupLinearSchedule

# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, "module") else model
    model_checkpoint = os.path.join(args.output_dir, f"{args.name}_checkpoint.bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def setup(args):
    config = CONFIGS[args.model_type]
    num_classes = 3  # Change to emotion classes

    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes)
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def valid(args, model, writer, test_loader, global_step):
    eval_losses = AverageMeter()
    model.eval()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    all_preds, all_labels = [], []
    metadata_store = []  # Store only the last validation batch

    loss_fct = torch.nn.CrossEntropyLoss()
    epoch_iterator = tqdm(test_loader, desc="Validating...", bar_format="{l_bar}{r_bar}", dynamic_ncols=True)

    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits, attn_weights, cls_token = model(x)

            if step == len(test_loader) - 1:  # Save only the last validation batch
                cls_tokens = cls_token.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()

                for i in range(y.shape[0]):
                    metadata_store.append({
                        "image_name": f"valid_{step * args.eval_batch_size + i}",
                        "label": labels[i].item(),
                        "cls_token": cls_tokens[i].tolist()
                    })

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)
            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    # Save last validation CLS tokens
    if metadata_store:
        df_valid = pd.DataFrame(metadata_store)
        cls_columns = [f"cls_dim_{i}" for i in range(len(df_valid['cls_token'][0]))]
        cls_df = pd.DataFrame(df_valid["cls_token"].to_list(), columns=cls_columns)
        df_valid.drop(columns=["cls_token"], inplace=True)
        df_valid = pd.concat([df_valid, cls_df], axis=1)
        df_valid.to_csv(os.path.join(args.output_dir, "cls_tokens_valid_last.csv"), index=False)

    return simple_accuracy(np.concatenate(all_preds), np.concatenate(all_labels))


def train(args, model):
    if args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    train_loader, test_loader = get_loader(args)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

    model.zero_grad()
    set_seed(args)
    global_step, best_acc = 0, 0

    while True:
        model.train()
        metadata_store = []  # Store only the last training batch

        epoch_iterator = tqdm(train_loader, desc="Training...", bar_format="{l_bar}{r_bar}", dynamic_ncols=True)

        for step, batch in enumerate(epoch_iterator):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss, attn_weights, cls_token = model(x, y)

            if step == len(train_loader) - 1:  # Save only the last training batch
                cls_tokens = cls_token.detach().cpu().numpy()
                labels = y.detach().cpu().numpy()

                for i in range(y.shape[0]):
                    metadata_store.append({
                        "image_name": f"train_{step * args.train_batch_size + i}",
                        "label": labels[i].item(),
                        "cls_token": cls_tokens[i].tolist()
                    })

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # Save last training CLS tokens
        if metadata_store:
            df_train = pd.DataFrame(metadata_store)
            cls_columns = [f"cls_dim_{i}" for i in range(len(df_train['cls_token'][0]))]
            cls_df = pd.DataFrame(df_train["cls_token"].to_list(), columns=cls_columns)
            df_train.drop(columns=["cls_token"], inplace=True)
            df_train = pd.concat([df_train, cls_df], axis=1)
            df_train.to_csv(os.path.join(args.output_dir, "cls_tokens_train_last.csv"), index=False)

        if global_step >= args.num_steps:
            break

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="test", help="Name of this run.")
    parser.add_argument("--model_type", choices=["ViT-B_16"], default="ViT-B_16")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_16.npz")
    parser.add_argument("--output_dir", default="output", type=str)
    parser.add_argument("--img_size", default=224, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument("--learning_rate", default=3e-2, type=float)
    parser.add_argument("--num_steps", default=100, type=int)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)
    args, model = setup(args)
    train(args, model)


if __name__ == "__main__":
    main()
