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


def compute_metrics(p):
    pred, labels = p
    # print(pred.shape)
    # print(labels.shape)
    pred = np.argmax(pred, axis=2)
    pred = pred.flatten()
    labels = labels.flatten()

    select=[labels!=-100]

    labels=np.select(select,[labels])
    pred=np.select(select,[pred])


    accuracy = metrics.accuracy_score(y_true=labels, y_pred=pred)
    recall = metrics.recall_score(y_true=labels, y_pred=pred,average="weighted")
    precision = metrics.precision_score(y_true=labels, y_pred=pred,average="weighted")
    f1 = metrics.f1_score(y_true=labels, y_pred=pred,average="weighted")
    print(accuracy,f1)
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


if __name__ == "__main__":
    config = getConfig()
    Step=16500
    checkpoint=f".\output\checkpoint-{Step}"
    model = AutoModelForTokenClassification.from_pretrained(checkpoint,num_labels=15)
    # model = AutoModelForTokenClassification.from_pretrained(
        # config["model_name"], num_labels=15)
    model.to(config["device"])

    # Define Trainer
    args = TrainingArguments(
        output_dir="output",
        evaluation_strategy="steps",
        eval_steps=500,
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["valid_batch_size"],
        num_train_epochs=config["epochs"],
        load_best_model_at_end=True,
    )
    train_dataset, val_dataset = getSets(standard=True)

    trainer = FocalLossTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train(f"checkpoint-{Step}")
    trainer.evaluate()
    model.save_pretained("BigBirdFinetune")
