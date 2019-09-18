import os
import time
import argparse
import numpy as np
import pandas as pd
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import *
from data_loader import get_dataloader
from configure import *
from loss import *
from utils import seed_torch, compute_dice
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description='Training model for steel defect detection')
    parser.add_argument("--model", type=str, default='UResNet34',
                        help="Name for encode used in Unet. Currently available: UResNet34")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="Number of workers for training. Default: 2")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Batch size for training. Default: 6")
    parser.add_argument("--num-epochs", type=int, default=300,
                        help="Number of epochs for training. Default: 100")
    parser.add_argument("--fold", type=int, default=0)

    return parser.parse_args()


class TrainerSegmentation(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=3e-05, eps=1e-8)
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
        self.loss = {phase: [] for phase in self.phases}
        self.bce_loss = {phase: [] for phase in self.phases}
        self.dice_loss = {phase: [] for phase in self.phases}
        self.dice = {phase: [] for phase in self.phases}

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss, bce_loss, dice_loss = self.criterion(outputs, masks.cuda())
        return loss, bce_loss, dice_loss, outputs

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        running_dice = np.zeros(4, dtype=np.float32)

        for images, masks in self.dataloaders[phase]:
            loss, bce_loss, dice_loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_bce_loss += bce_loss.item()
            running_dice_loss += dice_loss.item()

            outputs = outputs.detach().cpu()
            dice = compute_dice(outputs, masks, threshold=0.5)
            for j in range(4):
                running_dice[j] += dice[j]

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_bce_loss = running_bce_loss / len(self.dataloaders[phase])
        epoch_dice_loss = running_dice_loss / len(self.dataloaders[phase])
        epoch_dice = running_dice / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        self.bce_loss[phase].append(epoch_bce_loss)
        self.dice_loss[phase].append(epoch_dice_loss)
        self.dice[phase] = np.mean(epoch_dice)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_bce_loss, epoch_dice_loss, epoch_dice

    def plot_history(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        axes[0, 0].plot(self.loss['train'], '-b', label='Training')
        axes[0, 0].plot(self.loss['valid'], '-r', label='Validation')
        axes[0, 0].set_title("Loss", fontweight='bold')
        axes[0, 0].legend(loc="upper right", frameon=False)

        axes[0, 1].plot(self.bce_loss['train'], '-b', label='Training')
        axes[0, 1].plot(self.bce_loss['valid'], '-r', label='Validation')
        axes[0, 1].set_title("BCE Loss", fontweight='bold')
        axes[0, 1].legend(loc="upper right", frameon=False)

        axes[1, 0].plot(self.dice_loss['train'], '-b', label='Training')
        axes[1, 0].plot(self.dice_loss['valid'], '-r', label='Validation')
        axes[1, 0].set_title("Dice Loss", fontweight='bold')
        axes[1, 0].legend(loc="upper right", frameon=False)

        axes[1, 1].plot(self.dice['train'], '-b', label='Training')
        axes[1, 1].plot(self.dice['valid'], '-r', label='Validation')
        axes[1, 1].set_title("Dice", fontweight='bold')
        axes[1, 1].legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.txt".format(self.model_save_name, self.fold))
        header = ["Training loss", "Validation loss",
                  "Training bce loss", "Validation loss",
                  "Training dice loss", "Validation dice loss",
                  "Training dice", "Validation dice"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(len(self.loss['train'])):
                res = [self.loss['train'][i], self.loss['valid'][i],
                       self.bce_loss['train'][i], self.bce_loss['valid'][i],
                       self.dice_loss['train'][i], self.dice_loss['valid'][i],
                       self.dice['train'][i], self.dice['valid'][i]]

                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_bce_loss, train_dice_loss, train_dice = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_bce_loss, valid_dice_loss, valid_dice = self.iterate("valid")

            print("loss: %0.5f, bce_loss: %0.5f, dice_loss: %0.5f, "
                  "dice: %0.5f, %0.5f, %0.5f, %0.5f, mean dice: %0.5f" %
                  (train_loss, train_bce_loss, train_dice_loss,
                   train_dice[0], train_dice[1], train_dice[2], train_dice[3], np.mean(train_dice)))
            print("loss: %0.5f, bce_loss: %0.5f, dice_loss: %0.5f, "
                  "dice: %0.5f, %0.5f, %0.5f, %0.5f, mean dice: %0.5f" %
                  (valid_loss, valid_bce_loss, valid_dice_loss,
                   valid_dice[0], valid_dice[1], valid_dice[2], valid_dice[3], np.mean(valid_dice)))

            self.scheduler.step(metrics=valid_loss)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from %0.8f to %0.8f ********" % (self.best_loss, valid_loss))
                self.best_loss = valid_loss
                state = {
                    "state_dict": self.model.state_dict(),
                }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()


class TrainerVAE(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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
        self.criterion = DiceBCEVAELoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=1e-04, eps=1e-8)
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
        self.loss = {phase: [] for phase in self.phases}
        self.bce_loss = {phase: [] for phase in self.phases}
        self.dice_loss = {phase: [] for phase in self.phases}
        self.vae_loss = {phase: [] for phase in self.phases}
        self.dice = {phase: [] for phase in self.phases}
        self.thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def forward(self, images, masks):
        images, masks = images.cuda(), masks.cuda()
        masks_pred, images_pred, mu, logvar = self.model(images)
        loss, bce_loss, dice_loss, vae_loss = self.criterion(masks_pred=masks_pred, masks=masks,
                                                             images_pred=images_pred, images=images,
                                                             mu=mu, logvar=logvar)
        return loss, bce_loss, dice_loss, vae_loss, masks_pred

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        running_vae_loss = 0.0
        running_dice = np.zeros(shape=(len(self.thresholds), 4))

        for images, masks in self.dataloaders[phase]:
            loss, bce_loss, dice_loss, vae_loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_bce_loss += bce_loss.item()
            running_dice_loss += dice_loss.item()
            running_vae_loss += vae_loss.item()

            outputs = outputs.detach().cpu()
            for i, threshold in enumerate(self.thresholds):
                dice = compute_dice(outputs, masks, threshold=threshold)
                for j in range(4):
                    running_dice[i, j] += dice[j]

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_bce_loss = running_bce_loss / len(self.dataloaders[phase])
        epoch_dice_loss = running_dice_loss / len(self.dataloaders[phase])
        epoch_vae_loss = running_vae_loss / len(self.dataloaders[phase])
        epoch_dice = running_dice / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        self.bce_loss[phase].append(epoch_bce_loss)
        self.dice_loss[phase].append(epoch_dice_loss)
        self.vae_loss[phase].append(epoch_vae_loss)
        self.dice[phase] = epoch_dice

        torch.cuda.empty_cache()

        return epoch_loss, epoch_bce_loss, epoch_dice_loss, epoch_vae_loss, epoch_dice

    def plot_history(self):
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
        axes[0, 0].plot(self.loss['train'], '-b', label='Training')
        axes[0, 0].plot(self.loss['valid'], '-r', label='Validation')
        axes[0, 0].set_title("Loss", fontweight='bold')
        axes[0, 0].legend(loc="upper right", frameon=False)

        axes[0, 1].plot(self.bce_loss['train'], '-b', label='Training')
        axes[0, 1].plot(self.bce_loss['valid'], '-r', label='Validation')
        axes[0, 1].set_title("BCE Loss", fontweight='bold')
        axes[0, 1].legend(loc="upper right", frameon=False)

        axes[1, 0].plot(self.dice_loss['train'], '-b', label='Training')
        axes[1, 0].plot(self.dice_loss['valid'], '-r', label='Validation')
        axes[1, 0].set_title("Dice Loss", fontweight='bold')
        axes[1, 0].legend(loc="upper right", frameon=False)

        axes[1, 1].plot(self.vae_loss['train'], '-b', label='Training')
        axes[1, 1].plot(self.vae_loss['valid'], '-r', label='Validation')
        axes[1, 1].set_title("VAE Loss", fontweight='bold')
        axes[1, 1].legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.txt".format(self.model_save_name, self.fold))
        header = ["Training loss", "Validation loss",
                  "Training bce loss", "Validation loss",
                  "Training dice loss", "Validation dice loss",
                  "Training vae loss", "Validation vae loss"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(len(self.loss['train'])):
                res = [self.loss['train'][i], self.loss['valid'][i],
                       self.bce_loss['train'][i], self.bce_loss['valid'][i],
                       self.dice_loss['train'][i], self.dice_loss['valid'][i],
                       self.vae_loss['train'][i], self.vae_loss['valid'][i]]

                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_bce_loss, train_dice_loss, train_vae_loss, train_dice = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_bce_loss, valid_dice_loss, valid_vae_loss, valid_dice = self.iterate("valid")

            print("train_loss: %0.8f, train_bce_loss: %0.8f, train_dice_loss: %0.8f, train_vae_loss: %0.8f" %
                  (train_loss, train_bce_loss, train_dice_loss, train_vae_loss))
            print("valid_loss: %0.8f, valid_bce_loss: %0.8f, valid_dice_loss: %0.8f, valid_vae_loss: %0.8f" %
                  (valid_loss, valid_bce_loss, valid_dice_loss, valid_vae_loss))

            for i, threshold in enumerate(self.thresholds):
                print("%0.1f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f | %0.5f"
                      % (self.thresholds[i],
                         train_dice[i, 0], train_dice[i, 1], train_dice[i, 2], train_dice[i, 3], np.mean(train_dice[i]),
                         valid_dice[i, 0], valid_dice[i, 1], valid_dice[i, 2], valid_dice[i, 3], np.mean(valid_dice[i]))
                      )

            self.scheduler.step(metrics=valid_loss)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from %0.8f to %0.8f ********" % (self.best_loss, valid_loss))
                self.best_loss = valid_loss
                state = {
                    "state_dict": self.model.state_dict(),
                }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()


class TrainerClassification(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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
        self.criterion = BCEWithLogitsLoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=3e-05, eps=1e-8)
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
        self.loss = {phase: [] for phase in self.phases}

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss = self.criterion(outputs, masks.cuda())
        return loss, outputs

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        for images, masks in self.dataloaders[phase]:
            masks = (torch.sum(masks, (2, 3)) > 0).type(torch.float32)
            loss, outputs = self.forward(images, masks)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)

        torch.cuda.empty_cache()

        return epoch_loss

    def plot_history(self):
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        axes[0, 0].plot(self.loss['train'], '-b', label='Training')
        axes[0, 0].plot(self.loss['valid'], '-r', label='Validation')
        axes[0, 0].set_title("Loss", fontweight='bold')
        axes[0, 0].legend(loc="upper right", frameon=False)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.pdf".format(self.model_save_name, self.fold))
        fig.tight_layout()
        fig.savefig(output_filename)

        output_filename = os.path.join(self.training_history_path,
                                       "{}_fold_{}_loss.txt".format(self.model_save_name, self.fold))
        header = ["Training loss", "Validation loss"]

        with open(output_filename, "w") as f:
            f.write("\t".join(header) + "\n")
            for i in range(len(self.loss['train'])):
                res = [self.loss['train'][i], self.loss['valid'][i]]

                f.write("\t".join(map(str, res)) + "\n")

    def start(self):
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss = self.iterate("train")
            with torch.no_grad():
                valid_loss = self.iterate("valid")

            print("train_loss: %0.5f, valid_loss: %0.5f" % (train_loss, valid_loss))

            self.scheduler.step(metrics=valid_loss)
            if valid_loss < self.best_loss:
                print("******** Validation loss improved from %0.8f to %0.8f ********" % (self.best_loss, valid_loss))
                self.best_loss = valid_loss
                state = {
                    "state_dict": self.model.state_dict(),
                }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()


def main():
    args = parse_args()
    seed_torch(seed=42)

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    df_train_path = os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(args.fold))
    df_train = pd.read_csv(df_train_path)

    df_valid_path = os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(args.fold))
    df_valid = pd.read_csv(df_valid_path)

    print("Training on {} images, class 1: {}, class 2: {}, class 3: {}, class 4: {}".format(len(df_train),
                                                                                             df_train['defect1'].sum(),
                                                                                             df_train['defect2'].sum(),
                                                                                             df_train['defect3'].sum(),
                                                                                             df_train['defect4'].sum()))
    print("Validate on {} images, class 1: {}, class 2: {}, class 3: {}, class 4: {}".format(len(df_valid),
                                                                                             df_valid['defect1'].sum(),
                                                                                             df_valid['defect2'].sum(),
                                                                                             df_valid['defect3'].sum(),
                                                                                             df_valid['defect4'].sum()))
    model_trainer = None, None
    if args.model == "UResNet34":
        model_trainer = TrainerSegmentation(model=UResNet34(),
                                            num_workers=args.num_workers,
                                            batch_size=args.batch_size,
                                            num_epochs=args.num_epochs,
                                            model_save_path=model_save_path,
                                            training_history_path=training_history_path,
                                            model_save_name=args.model,
                                            fold=args.fold)
    elif args.model == "ResNet34":
        model_trainer = TrainerClassification(model=ResNet34(),
                                              num_workers=args.num_workers,
                                              batch_size=args.batch_size,
                                              num_epochs=args.num_epochs,
                                              model_save_path=model_save_path,
                                              training_history_path=training_history_path,
                                              model_save_name=args.model,
                                              fold=args.fold)

    model_trainer.start()


if __name__ == '__main__':
    main()
