import os
import time
import argparse
import torch
import numpy as np
import random
from torch.nn import BCEWithLogitsLoss
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from model import UResNet34
from data_loader import get_dataloader
from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for steel defect detection')
    parser.add_argument("--model", type=str, default='UResNet34',
                        help="Name for encode used in Unet. Currently available: UResNet34")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for training. Default: 4")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training. Default: 4")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of epochs for training. Default: 200")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--max_lr', type=float, default=0.01, help='max learning rate')
    parser.add_argument('--min_lr', type=float, default=0.001, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for SGD')

    return parser.parse_args()


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def compute_dice(preds, masks, threshold=0.5):
    '''Calculates dice of positive and negative images seperately'''
    '''probability and truth must be torch tensors'''
    batch_size = len(masks)
    with torch.no_grad():
        probs = torch.sigmoid(preds)
        probs = probs.view(batch_size, -1)
        masks = masks.view(batch_size, -1)
        assert (probs.shape == masks.shape)

        p = (probs > threshold).float()
        t = (masks > 0.5).float()

        t_sum = t.sum(-1)
        p_sum = p.sum(-1)
        neg_index = torch.nonzero(t_sum == 0)
        pos_index = torch.nonzero(t_sum >= 1)

        dice_neg = (p_sum == 0).float()
        dice_pos = 2 * (p * t).sum(-1) / ((p + t).sum(-1))

        dice_neg = dice_neg[neg_index]
        dice_pos = dice_pos[pos_index]
        dice = torch.cat([dice_pos, dice_neg])
        dice = dice.mean().item()

    return dice


class Trainer(object):
    '''This class takes care of training and validation of our model'''

    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path, max_lr=0.01, min_lr=0.001, momentum=0.9, weight_decay=1e-04):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_dice = 0
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion = BCEWithLogitsLoss()
        # self.optimizer = SGD(self.model.parameters(), lr=max_lr, momentum=momentum, weight_decay=weight_decay)
        # self.scheduler = CosineAnnealingLR(self.optimizer, T_max=50, eta_min=min_lr)
        self.optimizer = Adam(self.model.parameters(), lr=5e-4)
        self.scheduler = ReduceLROnPlateau(optimizer=self.optimizer, mode='max')
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader(
                phase=phase,
                fold=fold,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.losses = {phase: [] for phase in self.phases}
        self.dice_scores = {phase: [] for phase in self.phases}
        self.lr = []

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss = self.criterion(outputs, masks.cuda())
        return loss, outputs

    def iterate(self, phase):
        self.model.train(phase == "train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            outputs = outputs.detach().cpu()
            running_dice += compute_dice(outputs, masks)

        epoch_loss = running_loss / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        self.losses[phase].append(epoch_loss)
        self.dice_scores[phase].append(epoch_dice)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_dice

    def plot_history(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.losses['train'], '-b', label='Training loss')
        plt.plot(self.losses['valid'], '-r', label='Validation loss')
        plt.title("Loss", fontweight='bold')

        plt.subplot(1, 2, 2)
        plt.plot(self.dice_scores['train'], '-b', label='Training dice')
        plt.plot(self.dice_scores['valid'], '-r', label='Validation dice')
        plt.title("Dice", fontweight='bold')

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.pdf".format(self.model_save_name, self.fold))
        plt.tight_layout()
        plt.savefig(output_filename)

    def write_history(self):
        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.txt".format(self.model_save_name, self.fold))

        res = [self.losses['train'][-1], self.losses['valid'][-1],
               self.dice_scores['train'][-1], self.dice_scores['valid'][-1]]

        if os.path.exists(output_filename):
            with open(output_filename, "a") as f:
                f.write("\t".join(map(str, res)) + "\n")
        else:
            header = ["Training loss", "Validation loss", "Training dice", "Validation dice"]
            with open(output_filename, "w") as f:
                f.write("\t".join(header) + "\n")
                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        for epoch in range(self.num_epochs):
            start = time.strftime("%D:%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")
            # print("Learning rate: %0.8f" % (self.scheduler.get_lr()[0]))

            train_loss, train_dice = self.iterate("train")
            valid_loss, valid_dice = self.iterate("valid")

            print("Training loss: %0.8f, dice: %0.8f" % (train_loss, train_dice))
            print("Validation loss: %0.8f, dice: %0.8f" % (valid_loss, valid_dice))

            self.write_history()

            state = {
                "epoch": epoch,
                "best_dice": self.best_dice,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            # self.lr.append(self.scheduler.get_lr()[0])
            self.scheduler.step(metrics=valid_dice, epoch=epoch)
            if valid_dice > self.best_dice:
                print("******** New optimal found, saving state ********")
                state["best_dice"] = self.best_dice = valid_dice
                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()


def main():
    args = parse_args()

    seed_torch(seed=42)

    model = None
    if args.model == "UResNet34":
        model = UResNet34()

    elif args.model == "UResNet34SCSE":
        model = UResNet34SCSE()

    elif args.model == "UResNet34SCSEHyper":
        model = UResNet34SCSEHyper()

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    model_trainer = Trainer(model=model,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            model_save_path=model_save_path,
                            training_history_path=training_history_path,
                            model_save_name=args.model,
                            fold=args.fold)
    model_trainer.start()
    model_trainer.plot_history()


if __name__ == '__main__':
    main()
