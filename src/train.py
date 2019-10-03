import os
import time
import argparse
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
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for training. Default: 4")
    parser.add_argument("--batch-size", type=int, default=6,
                        help="Batch size for training. Default: 6")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--class", type=int, default=1)

    return parser.parse_args()


class TrainerSegmentation(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path, task="seg"):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion = DiceBCELoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=1e-05, eps=1e-8)
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=4,
                num_workers=self.num_workers,
                task=task
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
        self.dice[phase].append(np.mean(epoch_dice))

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
        best_dice = 0.0

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

            self.scheduler.step(metrics=np.mean(valid_dice))
            if np.mean(valid_dice) > best_dice:
                print("******** Validation dice improved from %0.8f to %0.8f ********" %
                      (best_dice, np.mean(valid_dice)))
                best_dice = np.mean(valid_dice)
                state = {
                    "best_dice": best_dice,
                    "state_dict": self.model.state_dict(),
                }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()

        return best_dice


class TrainerClassification(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path, task="cls"):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion = BCEWithLogitsLoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=1e-05, eps=1e-8)
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers,
                task=task
            )
            for phase in self.phases
        }
        self.loss = {phase: [] for phase in self.phases}
        self.accuracy = {phase: np.zeros(shape=(0, 4), dtype=np.float32) for phase in self.phases}

    def forward(self, images, masks):
        outputs = self.model(images.cuda())
        loss = self.criterion(outputs, masks.cuda())
        return loss, outputs

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        running_acc = np.zeros(4)
        for images, masks in self.dataloaders[phase]:
            labels = (torch.sum(masks, (2, 3)) > 0).type(torch.float32)

            loss, outputs = self.forward(images, labels)
            outputs = (outputs.detach().cpu() > 0.5).type(torch.float32).numpy()
            labels = labels.numpy()
            correct = np.equal(outputs, labels).astype(np.float32)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_acc += np.sum(correct, axis=0) / labels.shape[0]

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_acc = running_acc / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        self.accuracy[phase] = np.concatenate((self.accuracy[phase], np.expand_dims(epoch_acc, axis=0)), axis=0)

        torch.cuda.empty_cache()

        return epoch_loss, epoch_acc

    def plot_history(self):
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
        axes[0, 0].plot(self.loss['train'], '-b', label='Training')
        axes[0, 0].plot(self.loss['valid'], '-r', label='Validation')
        axes[0, 0].set_title("Loss", fontweight='bold')
        axes[0, 0].legend(loc="upper right", frameon=False)

        axes[0, 1].plot(self.accuracy['train'][:, 0], '-b', label='Training')
        axes[0, 1].plot(self.accuracy['valid'][:, 0], '-r', label='Validation')
        axes[0, 1].set_title("Accuracy 1", fontweight='bold')
        axes[0, 1].legend(loc="lower right", frameon=False)

        axes[0, 2].plot(self.accuracy['train'][:, 1], '-b', label='Training')
        axes[0, 2].plot(self.accuracy['valid'][:, 1], '-r', label='Validation')
        axes[0, 2].set_title("Accuracy 2", fontweight='bold')
        axes[0, 2].legend(loc="lower right", frameon=False)

        axes[1, 0].plot(self.accuracy['train'][:, 2], '-b', label='Training')
        axes[1, 0].plot(self.accuracy['valid'][:, 2], '-r', label='Validation')
        axes[1, 0].set_title("Accuracy 3", fontweight='bold')
        axes[1, 0].legend(loc="lower right", frameon=False)

        axes[1, 1].plot(self.accuracy['train'][:, 3], '-b', label='Training')
        axes[1, 1].plot(self.accuracy['valid'][:, 3], '-r', label='Validation')
        axes[1, 1].set_title("Accuracy 4", fontweight='bold')
        axes[1, 1].legend(loc="lower right", frameon=False)

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
        best_acc = 0.0

        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_acc = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_acc = self.iterate("valid")

            print("train_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (train_loss, train_acc[0], train_acc[1], train_acc[2], train_acc[3], np.mean(train_acc)))
            print("valid_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (valid_loss, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3], np.mean(valid_acc)))

            self.scheduler.step(metrics=np.mean(valid_acc))
            if np.mean(valid_acc) > best_acc:
                print("******** Validation accuracy improved from %0.8f to %0.8f ********" % (
                    best_acc, np.mean(valid_acc)))

                best_acc = np.mean(valid_acc)
                state = {
                    "best_acc": best_acc,
                    "state_dict": self.model.state_dict(),
                }

                filename = os.path.join(self.model_save_path, "{}_fold_{}.pt".format(self.model_save_name, self.fold))
                if os.path.exists(filename):
                    os.remove(filename)
                torch.save(state, filename)

            print()
            self.plot_history()

        return best_acc


def main():
    args = parse_args()
    seed_torch(seed=42)

    model_save_path = os.path.join(SAVE_MODEL_PATH, args.model)
    training_history_path = os.path.join(TRAINING_HISTORY_PATH, args.model)

    df_train_path = os.path.join(SPLIT_FOLDER, "fold_{}_train.csv".format(args.fold))
    df_train = pd.read_csv(df_train_path)

    df_valid_path = os.path.join(SPLIT_FOLDER, "fold_{}_valid.csv".format(args.fold))
    df_valid = pd.read_csv(df_valid_path)

    if args.model in ["UResNet34", "USeResNext50"]:
        df_train = df_train.loc[(df_train["defect1"] != 0) | (df_train["defect2"] != 0) | (df_train["defect3"] != 0) | (
                df_train["defect4"] != 0)]
        df_valid = df_valid.loc[(df_valid["defect1"] != 0) | (df_valid["defect2"] != 0) | (df_valid["defect3"] != 0) | (
                df_valid["defect4"] != 0)]

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
    model_trainer, best = None, None
    if args.model == "UResNet34":
        model_trainer = TrainerSegmentation(model=UResNet34(),
                                            num_workers=args.num_workers,
                                            batch_size=args.batch_size,
                                            num_epochs=200,
                                            model_save_path=model_save_path,
                                            training_history_path=training_history_path,
                                            model_save_name=args.model,
                                            fold=args.fold)

    elif args.model == "ResNet34":
        model_trainer = TrainerClassification(model=ResNet34(),
                                              num_workers=args.num_workers,
                                              batch_size=args.batch_size,
                                              num_epochs=100,
                                              model_save_path=model_save_path,
                                              training_history_path=training_history_path,
                                              model_save_name=args.model,
                                              fold=args.fold)

    best = model_trainer.start()

    print("Training is done, best: {}".format(best))


if __name__ == '__main__':
    main()
