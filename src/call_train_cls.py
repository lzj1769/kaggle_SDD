import os
import pathlib

from configure import SAVE_MODEL_PATH, TRAINING_HISTORY_PATH

model_list = ['ResNet34V2']
fold_list = [0, 1, 2, 3, 4]

for model in model_list:
    model_save_path = os.path.join(SAVE_MODEL_PATH, model)
    if not os.path.exists(model_save_path):
        pathlib.Path(model_save_path).mkdir(parents=True, exist_ok=True)

    training_history_path = os.path.join(TRAINING_HISTORY_PATH, model)
    if not os.path.exists(training_history_path):
        pathlib.Path(training_history_path).mkdir(parents=True, exist_ok=True)
    for fold in fold_list:
        job_name = "{}_fold_{}".format(model, fold)
        command = "sbatch -J " + job_name + " -o " + "./cluster_out/" + job_name + "_out.txt -e " + \
                  "./cluster_err/" + job_name + "_err.txt -t 20:00:00 --mem 30G -A rwth0455 "
        command += "--partition=c18g -c 4 --gres=gpu:1 train_cls.zsh"
        os.system(command + " " + model + " " + str(fold))
