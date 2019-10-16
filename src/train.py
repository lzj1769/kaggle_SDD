import time
import argparse
from torch.optim import SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix
from model import *
from data_loader import *
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

    return parser.parse_args()


class TrainerSegmentation(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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
            phase: get_dataloader_seg(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers
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


class TrainerSegmentationDeepSupervision(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.training_history_path = training_history_path
        self.criterion_seg = DiceBCELoss()
        self.criterion_cls = BCEWithLogitsLoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=1e-05, eps=1e-8)
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader_seg(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers
            )
            for phase in self.phases
        }
        self.loss = {phase: [] for phase in self.phases}
        self.bce_loss = {phase: [] for phase in self.phases}
        self.dice_loss = {phase: [] for phase in self.phases}
        self.dice = {phase: [] for phase in self.phases}

    def forward(self, images, masks, labels):
        outputs_cls, outputs_seg = self.model(images.cuda())
        cls_loss = self.criterion_cls(outputs_cls, labels.cuda())
        seg_loss, bce_loss, dice_loss = self.criterion_seg(outputs_seg, masks.cuda())
        loss = (cls_loss + seg_loss)

        return loss, cls_loss, seg_loss, bce_loss, dice_loss, outputs_cls, outputs_seg

    def iterate(self, phase):
        self.model.train(phase == "train")

        running_loss = 0.0
        running_cls_loss = 0.0
        running_seg_loss = 0.0
        running_bce_loss = 0.0
        running_dice_loss = 0.0
        running_dice = np.zeros(4, dtype=np.float32)
        epoch_label, epoch_pred = np.zeros(shape=(0, 4), dtype=np.int8), np.zeros(shape=(0, 4), dtype=np.int8)

        for images, masks in self.dataloaders[phase]:
            labels = (torch.sum(masks, (2, 3)) > 0).type(torch.float32)
            loss, cls_loss, seg_loss, bce_loss, dice_loss, outputs_cls, outputs_seg = self.forward(images, masks,
                                                                                                   labels)
            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()
            running_cls_loss += cls_loss.item()
            running_seg_loss += seg_loss.item()
            running_bce_loss += bce_loss.item()
            running_dice_loss += dice_loss.item()

            labels = labels.type(torch.int8).numpy()
            outputs_cls = (outputs_cls.detach().cpu() > 0).type(torch.int8).numpy()
            epoch_pred = np.concatenate((epoch_pred, outputs_cls), axis=0)
            epoch_label = np.concatenate((epoch_label, labels), axis=0)

            outputs_seg = outputs_seg.detach().cpu()
            dice = compute_dice(outputs_seg, masks, threshold=0.5)
            for j in range(4):
                running_dice[j] += dice[j]

        loss = running_loss / len(self.dataloaders[phase])
        cls_loss = running_cls_loss / len(self.dataloaders[phase])
        seg_loss = running_seg_loss / len(self.dataloaders[phase])
        bce_loss = running_bce_loss / len(self.dataloaders[phase])
        dice_loss = running_dice_loss / len(self.dataloaders[phase])
        dice = running_dice / len(self.dataloaders[phase])

        torch.cuda.empty_cache()

        return loss, cls_loss, seg_loss, bce_loss, dice_loss, dice, epoch_label, epoch_pred

    def start(self):
        best_dice = 0.0

        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_cls_loss, train_seg_loss, train_bce_loss, train_dice_loss, train_dice, \
            train_epoch_label, train_epoch_pred = self.iterate("train")

            with torch.no_grad():
                valid_loss, valid_cls_loss, valid_seg_loss, valid_bce_loss, valid_dice_loss, valid_dice, \
                valid_epoch_label, valid_epoch_pred = self.iterate("valid")

            print("loss: %0.5f, cls_loss: %0.5f, seg_loss: %0.5f, bce_loss: %0.5f, dice_loss: %0.5f, "
                  "dice: %0.5f, %0.5f, %0.5f, %0.5f, mean dice: %0.5f" %
                  (train_loss, train_cls_loss, train_seg_loss, train_bce_loss, train_dice_loss,
                   train_dice[0], train_dice[1], train_dice[2], train_dice[3], np.mean(train_dice)))
            print("loss: %0.5f, cls_loss: %0.5f, seg_loss: %0.5f, bce_loss: %0.5f, dice_loss: %0.5f, "
                  "dice: %0.5f, %0.5f, %0.5f, %0.5f, mean dice: %0.5f" %
                  (valid_loss, valid_cls_loss, valid_seg_loss, valid_bce_loss, valid_dice_loss,
                   valid_dice[0], valid_dice[1], valid_dice[2], valid_dice[3], np.mean(valid_dice)))

            for cls in range(4):
                train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_epoch_label[:, cls],
                                                                          train_epoch_pred[:, cls]).ravel()

                valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(valid_epoch_label[:, cls],
                                                                          valid_epoch_pred[:, cls]).ravel()
                print("train: TP %5d, TN %5d, FP %5d, FN %5d, valid: TP %5d, TN %5d, FP %5d, FN %5d"
                      % (train_tp, train_tn, train_fp, train_fn, valid_tp, valid_tn, valid_fp, valid_fn))

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

        return best_dice


class TrainerClassification(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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
            phase: get_dataloader_cls(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers,
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
        epoch_label, epoch_pred = np.zeros(shape=(0, 4), dtype=np.int8), np.zeros(shape=(0, 4), dtype=np.int8)

        for images, labels in self.dataloaders[phase]:
            loss, outputs = self.forward(images, labels)
            outputs = (outputs.detach().cpu() > 0).type(torch.int8).numpy()
            labels = labels.type(torch.int8).numpy()

            epoch_pred = np.concatenate((epoch_pred, outputs), axis=0)
            epoch_label = np.concatenate((epoch_label, labels), axis=0)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        torch.cuda.empty_cache()

        return epoch_loss, epoch_label, epoch_pred

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

        axes[1, 2].plot(np.mean(self.accuracy['train'], axis=1), '-b', label='Training')
        axes[1, 2].plot(np.mean(self.accuracy['valid'], axis=1), '-r', label='Validation')
        axes[1, 2].set_title("Mean accuracy", fontweight='bold')
        axes[1, 2].legend(loc="lower right", frameon=False)

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

            train_loss, train_epoch_label, train_epoch_pred = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_epoch_label, valid_epoch_pred = self.iterate("valid")

            correct = np.equal(train_epoch_label, train_epoch_pred).astype(np.float32)
            train_acc = np.sum(correct, axis=0) / train_epoch_label.shape[0]

            correct = np.equal(valid_epoch_label, valid_epoch_pred).astype(np.float32)
            valid_acc = np.sum(correct, axis=0) / valid_epoch_label.shape[0]

            print("train_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (train_loss, train_acc[0], train_acc[1], train_acc[2], train_acc[3], np.mean(train_acc)))
            print("valid_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (valid_loss, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3], np.mean(valid_acc)))

            for cls in range(4):
                train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_epoch_label[:, cls],
                                                                          train_epoch_pred[:, cls]).ravel()

                valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(valid_epoch_label[:, cls],
                                                                          valid_epoch_pred[:, cls]).ravel()
                print("train: TP %5d, TN %5d, FP %5d, FN %5d, valid: TP %5d, TN %5d, FP %5d, FN %5d"
                      % (train_tp, train_tn, train_fp, train_fn, valid_tp, valid_tn, valid_fp, valid_fn))

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


class TrainerClassificationMixUP(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path, alpha=0.4):
        self.model = model
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.phases = ["train", "valid"]
        self.model_save_path = model_save_path
        self.model_save_name = model_save_name
        self.fold = fold
        self.alpha = alpha
        self.training_history_path = training_history_path
        self.criterion = BCEWithLogitsLoss()

        self.optimizer = SGD(self.model.parameters(), lr=1e-02, momentum=0.9, weight_decay=1e-04)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.1, patience=10,
                                           verbose=True, threshold=1e-8,
                                           min_lr=1e-05, eps=1e-8)
        self.model = self.model.cuda()
        self.dataloaders = {
            phase: get_dataloader_cls(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers,
            )
            for phase in self.phases
        }
        self.loss = {phase: [] for phase in self.phases}
        self.accuracy = {phase: np.zeros(shape=(0, 4), dtype=np.float32) for phase in self.phases}

    def iterate_train(self, phase="train"):
        self.model.train()

        running_loss, running_acc = 0.0, 0.0
        for images, labels in self.dataloaders[phase]:
            batch_size = labels.shape[0]
            num_classes = labels.shape[1]

            t = np.random.beta(self.alpha, self.alpha)
            t = max(t, 1 - t)
            index = torch.randperm(batch_size).cuda()
            mixed_images = t * images + (1 - t) * images[index, :]
            labels_a, labels_b = labels, labels[index]

            outputs = self.model(mixed_images.cuda())
            loss = t * self.criterion(outputs, labels_a.cuda()) + (1 - t) * self.criterion(outputs, labels_b.cuda())

            outputs = (outputs.detach().cpu() > 0.0).type(torch.int8).numpy()
            labels_a = labels_a.type(torch.int8).numpy()
            labels_b = labels_b.type(torch.int8).numpy()

            accuracy_a = np.sum(np.equal(outputs, labels_a).astype(np.float32)) / (num_classes * batch_size)
            accuracy_b = np.sum(np.equal(outputs, labels_b).astype(np.float32)) / (num_classes * batch_size)
            running_acc += t * accuracy_a + (1 - t) * accuracy_b

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])
        epoch_acc = running_acc / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        torch.cuda.empty_cache()

        return epoch_loss, epoch_acc

    def iterate_valid(self, phase="valid"):
        self.model.eval()

        running_loss = 0.0
        epoch_label, epoch_pred = np.zeros(shape=(0, 4), dtype=np.int8), np.zeros(shape=(0, 4), dtype=np.int8)
        for images, labels in self.dataloaders[phase]:
            outputs = self.model(images.cuda())
            loss = self.criterion(outputs, labels.cuda())

            outputs = (outputs.detach().cpu() > 0.0).type(torch.int8).numpy()
            labels = labels.type(torch.int8).numpy()

            epoch_pred = np.concatenate((epoch_pred, outputs), axis=0)
            epoch_label = np.concatenate((epoch_label, labels), axis=0)

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])
        self.loss[phase].append(epoch_loss)
        torch.cuda.empty_cache()

        return epoch_loss, epoch_label, epoch_pred

    def start(self):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_acc = self.iterate_train()
            with torch.no_grad():
                valid_loss, valid_epoch_label, valid_epoch_pred = self.iterate_valid()

            correct = np.equal(valid_epoch_label, valid_epoch_pred).astype(np.float32)
            valid_acc = np.sum(correct, axis=0) / valid_epoch_label.shape[0]

            print("valid_loss: %0.5f, mean accuracy: %0.5f" % (train_loss, train_acc))
            print("valid_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (valid_loss, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3], np.mean(valid_acc)))

            for cls in range(4):
                valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(valid_epoch_label[:, cls],
                                                                          valid_epoch_pred[:, cls]).ravel()
                print("valid: TP %5d, TN %5d, FP %5d, FN %5d" % (valid_tp, valid_tn, valid_fp, valid_fn))

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

        return best_acc


class TrainerClassificationPesudoLabels(object):
    def __init__(self, model, num_workers, batch_size, num_epochs, model_save_path, model_save_name,
                 fold, training_history_path):
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
            phase: get_dataloader_cls_pesudo_labels(
                phase=phase,
                fold=fold,
                train_batch_size=self.batch_size,
                valid_batch_size=self.batch_size,
                num_workers=self.num_workers,
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
        epoch_label, epoch_pred = np.zeros(shape=(0, 4), dtype=np.int8), np.zeros(shape=(0, 4), dtype=np.int8)

        for images, labels in self.dataloaders[phase]:
            loss, outputs = self.forward(images, labels)
            outputs = (outputs.detach().cpu() > 0).type(torch.int8).numpy()
            labels = labels.type(torch.int8).numpy()

            epoch_pred = np.concatenate((epoch_pred, outputs), axis=0)
            epoch_label = np.concatenate((epoch_label, labels), axis=0)

            if phase == "train":
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(self.dataloaders[phase])

        self.loss[phase].append(epoch_loss)
        torch.cuda.empty_cache()

        return epoch_loss, epoch_label, epoch_pred

    def start(self):
        best_acc = 0.0
        for epoch in range(self.num_epochs):
            start = time.strftime("%D-%H:%M:%S")
            print("Epoch: {}/{} |  time : {}".format(epoch + 1, self.num_epochs, start))
            print("=================================================================")

            train_loss, train_epoch_label, train_epoch_pred = self.iterate("train")
            with torch.no_grad():
                valid_loss, valid_epoch_label, valid_epoch_pred = self.iterate("valid")

            correct = np.equal(train_epoch_label, train_epoch_pred).astype(np.float32)
            train_acc = np.sum(correct, axis=0) / train_epoch_label.shape[0]

            correct = np.equal(valid_epoch_label, valid_epoch_pred).astype(np.float32)
            valid_acc = np.sum(correct, axis=0) / valid_epoch_label.shape[0]

            print("train_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (train_loss, train_acc[0], train_acc[1], train_acc[2], train_acc[3], np.mean(train_acc)))
            print("valid_loss: %0.5f, accuracy: %0.5f, %0.5f, %0.5f, %0.5f, mean accuracy: %0.5f"
                  % (valid_loss, valid_acc[0], valid_acc[1], valid_acc[2], valid_acc[3], np.mean(valid_acc)))

            for cls in range(4):
                train_tn, train_fp, train_fn, train_tp = confusion_matrix(train_epoch_label[:, cls],
                                                                          train_epoch_pred[:, cls]).ravel()

                valid_tn, valid_fp, valid_fn, valid_tp = confusion_matrix(valid_epoch_label[:, cls],
                                                                          valid_epoch_pred[:, cls]).ravel()
                print("train: TP %5d, TN %5d, FP %5d, FN %5d, valid: TP %5d, TN %5d, FP %5d, FN %5d"
                      % (train_tp, train_tn, train_fp, train_fn, valid_tp, valid_tn, valid_fp, valid_fn))

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

    if args.model in ["UResNet34", "FPN", "FPResNext50", "FPResNet34", "FPResNet34V2"]:
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
                                            num_epochs=100,
                                            model_save_path=model_save_path,
                                            training_history_path=training_history_path,
                                            model_save_name=args.model,
                                            fold=args.fold)

    elif args.model == "FPResNet34":
        model_trainer = TrainerSegmentation(model=FPResNet34(),
                                            num_workers=args.num_workers,
                                            batch_size=args.batch_size,
                                            num_epochs=200,
                                            model_save_path=model_save_path,
                                            training_history_path=training_history_path,
                                            model_save_name=args.model,
                                            fold=args.fold)

    elif args.model == "FPResNet34DeepSupervision":
        model_trainer = TrainerSegmentationDeepSupervision(model=FPResNet34DeepSupervision(),
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

    elif args.model == "ResNet34WithPseudoLabels":
        model_trainer = TrainerClassificationPesudoLabels(model=ResNet34WithPseudoLabels(),
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
