from focalloss import FocalLossTrainer
from Config import getConfig
from Dataset import getSets
from transformers import AutoModelForTokenClassification, BigBirdConfig, TrainingArguments, Trainer
import torch
import numpy as np
import warnings
from inferance import inferencePipeline
from sklearn import metrics
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    config = getConfig()
    training_set,training_loader,testing_set,testing_loader = getSets()

    for Step in range(500,16500,500):
        model_name=f".\output\checkpoint-{Step}"
        print(model_name)
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=15)
        model.to(config["device"])
        inferencePipeline(testing_set,testing_loader,model,config)

