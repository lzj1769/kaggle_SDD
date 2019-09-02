import os
import time
import argparse
import numpy as np
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from model import *
from data_loader import get_dataloader
from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH
from loss import DiceBCELoss
from utils import seed_torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for steel defect detection')
    parser.add_argument("--model", type=str, default='UResNet34',
                        help="Name for encode used in Unet. Currently available: UResNet34")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for training. Default: 2")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training. Default: 4")
    parser.add_argument("--num-epochs", type=int, default=200,
                        help="Number of epochs for training. Default: 200")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-03, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for SGD')

    return parser.parse_args()


class Trainer(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path, lr, weight_decay):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.best_loss = np.inf
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion = DiceBCELoss()

        self.optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = StepLR(self.optimizer, step_size=30, gamma=0.5)
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
        self.bce = {phase: [] for phase in self.phases}
        self.dice = {phase: [] for phase in self.phases}

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss, bce, dice = self.criterion(outputs, masks.cuda())
        return loss, bce, dice, outputs

    def iterate(self, phase):
        if phase == "train":
            self.model.train()
        else:
            self.model.eval()

        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        running_bce = 0.0
        running_dice = 0.0

        for images, masks in dataloader:
            loss, bce, dice, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_bce += bce.item()
            running_dice += dice.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_bce = running_bce / len(dataloader)
        epoch_dice = running_dice / len(dataloader)

        self.losses[phase].append(epoch_loss)
        self.bce[phase].append(epoch_bce)
        self.dice[phase].append(epoch_dice)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_bce, epoch_dice

    def plot_history(self):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        ax1.plot(self.losses['train'], '-b', label='Training')
        ax1.plot(self.losses['valid'], '-r', label='Validation')
        ax1.set_title("Loss", fontweight='bold')
        ax1.legend(loc="upper right", frameon=False)

        ax2.plot(self.bce['train'], '-b', label='Training')
        ax2.plot(self.bce['valid'], '-r', label='Validation')
        ax2.set_title("BCE", fontweight='bold')
        ax2.legend(loc="upper right", frameon=False)

        ax3.plot(self.dice['train'], '-b', label='Training')
        ax3.plot(self.dice['valid'], '-r', label='Validation')
        ax3.set_title("Dice", fontweight='bold')
        ax3.legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

    def write_history(self):
        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}.txt".format(self.model_save_name, self.fold))

        res = [self.losses['train'][-1], self.losses['valid'][-1],
               self.bce['train'][-1], self.bce['valid'][-1],
               self.dice['train'][-1], self.dice['valid'][-1]]

        if os.path.exists(output_filename):
            with open(output_filename, "a") as f:
                f.write("\t".join(map(str, res)) + "\n")
        else:
            header = ["Training loss", "Validation loss", "Training bce", "Validation bce",
                      "Training dice", "Validation dice"]
            with open(output_filename, "w") as f:
                f.write("\t".join(header) + "\n")
                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        print("Train on {} mini-batches, validate on {} mini-batches".format(len(self.dataloaders["train"]),
                                                                             len(self.dataloaders["valid"])))

        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_bce, train_dice = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_bce, valid_dice = self.iterate("valid")

            print("train_loss: %0.8f, train_bce: %0.8f, train_dice: %0.8f" % (train_loss, train_bce, train_dice))
            print("valid_loss: %0.8f, valid_bce: %0.8f, valid_dice: %0.8f" % (valid_loss, valid_bce, valid_dice))

            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }

            self.scheduler.step(epoch=epoch)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from {} to {}, saving state ********".format(self.best_loss,
                                                                                                      valid_loss))
                state["best_loss"] = self.best_loss = valid_loss
                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.write_history()


def main():
    args = parse_args()

    seed_torch(seed=42)

    model = None
    if args.model == "UResNet34":
        model = UResNet34()
    elif args.model == "UResNet50":
        model = UResNet50()
    elif args.model == "UResNext50":
        model = UResNext50()

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    model_trainer = Trainer(model=model,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            num_epochs=args.num_epochs,
                            model_save_path=model_save_path,
                            training_history_path=training_history_path,
                            model_save_name=args.model,
                            fold=args.fold,
                            lr=args.lr,
                            weight_decay=args.weight_decay)
    model_trainer.start()
    model_trainer.plot_history()


if __name__ == '__main__':
    main()
