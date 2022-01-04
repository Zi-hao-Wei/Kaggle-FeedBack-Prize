import numpy as np
import torch
import pandas as pd
from transformers import BigBirdForTokenClassification, BigBirdConfig,TrainingArguments, Trainer
from Config import getConfig
from Dataset import getSets

output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
          'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
labels_to_ids = {v:k for k,v in enumerate(output_labels)}
ids_to_labels = {k:v for k,v in enumerate(output_labels)}

def inference(model, config, batch):
    global ids_to_labels
    # MOVE BATCH TO GPU AND INFER
    ids = batch["input_ids"].to(config['device'])
    mask = batch["attention_mask"].to(config['device'])
    outputs = model(ids, attention_mask=mask, return_dict=False)
    all_preds = torch.argmax(outputs[0], axis=-1).cpu().numpy()

    # INTERATE THROUGH EACH TEXT AND GET PRED
    predictions = []
    for k, text_preds in enumerate(all_preds):
        token_preds = [ids_to_labels[i] for i in text_preds]

        prediction = []
        word_ids = batch['wids'][k].numpy()
        previous_word_idx = -1
        for idx, word_idx in enumerate(word_ids):
            if word_idx == -1:
                pass
            elif word_idx != previous_word_idx:
                prediction.append(token_preds[idx])
                previous_word_idx = word_idx
        predictions.append(prediction)

    return predictions


def get_predictions(model, config, df, loader):
    # put model in training mode
    model.eval()

    # GET WORD LABEL PREDICTIONS
    y_pred2 = []
    for batch in loader:
        labels = inference(model, config, batch)
        y_pred2.extend(labels)

    final_preds2 = []
    for i in range(len(df)):
        idx = df.id.values[i]
        # prpoch
        d = [x.replace('B-', '').replace('I-', '') for x in y_pred2[i]]
        pred = y_pred2[i]  # Leave "B" and "I"
        preds = []
        j = 0
        while j < len(pred):
            cls = pred[j]
            # The commented out line below appears to be a bug.
#             if cls == 'O': j += 1
            if cls == 'O':
                pass
            else:
                cls = cls.replace('B', 'I')  # spans start with B
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1

            if cls != 'O' and cls != '' and end - j > 7:
                final_preds2.append((idx, cls.replace('I-', ''),
                                     ' '.join(map(str, list(range(j, end))))))

            j = end

    oof = pd.DataFrame(final_preds2)
    oof.columns = ['id', 'class', 'new_predictionstring']

    return oof


def calc_overlap(row):
    """
    Calculates the overlap between prediction and
    ground truth and overlap percentages used for determining
    true positives.
    """
    set_pred = set(row.new_predictionstring_pred.split(' '))
    set_gt = set(row.new_predictionstring_gt.split(' '))
    # Length of each and intersection
    len_gt = len(set_gt)
    len_pred = len(set_pred)
    inter = len(set_gt.intersection(set_pred))
    overlap_1 = inter / len_gt
    overlap_2 = inter / len_pred
    return [overlap_1, overlap_2]


def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition

    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'new_predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df = pred_df[['id', 'class', 'new_predictionstring']] \
        .reset_index(drop=True).copy()
    pred_df['pred_id'] = pred_df.index
    gt_df['gt_id'] = gt_df.index
    # Step 1. all ground truths and predictions for a given class are compared.
    joined = pred_df.merge(gt_df,
                           left_on=['id', 'class'],
                           right_on=['id', 'discourse_type'],
                           how='outer',
                           suffixes=('_pred', '_gt')
                           )
    joined['new_predictionstring_gt'] = joined['new_predictionstring_gt'].fillna(' ')
    joined['new_predictionstring_pred'] = joined['new_predictionstring_pred'].fillna(
        ' ')

    joined['overlaps'] = joined.apply(calc_overlap, axis=1)

    # 2. If the overlap between the ground truth and prediction is >= 0.5,
    # and the overlap between the prediction and the ground truth >= 0.5,
    # the prediction is a match and considered a true positive.
    # If multiple matches exist, the match with the highest pair of overlaps is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (
        joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis=1)
    tp_pred_ids = joined.query('potential_TP') \
        .sort_values('max_overlap', ascending=False) \
        .groupby(['id', 'new_predictionstring_gt']).first()['pred_id'].values

    # 3. Any unmatched ground truths are false negatives
    # and any unmatched predictions are false positives.
    fp_pred_ids = [p for p in joined['pred_id'].unique()
                   if p not in tp_pred_ids]

    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique()
                        if c not in matched_gt_ids]

    # Get numbers of each type
    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    # calc microf1
    my_f1_score = TP / (TP + 0.5*(FP+FN))
    return my_f1_score


def inferencePipeline(test_dataset,testing_loader,model,config):  # note this doesn't run during submit
    
    train_df = pd.read_csv('corrected.csv')
    IDS = train_df.id.unique()
    np.random.seed(42)
    train_idx = np.random.choice(np.arange(len(IDS)),int(0.9*len(IDS)),replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)),train_idx)
    np.random.seed(None)

    valid = train_df.loc[train_df['id'].isin(IDS[valid_idx])]

    # OOF PREDICTIONS
    oof = get_predictions(model,config,test_dataset, testing_loader)

    # COMPUTE F1 SCORE
    f1s = []
    CLASSES = oof['class'].unique()
    print()
    for c in CLASSES:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = valid.loc[valid['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(c, f1)
        f1s.append(f1)
    print()
    print('Overall', np.mean(f1s))
    print()
    return np.mean(f1s)

if __name__ == "__main__":
    config = getConfig()

    model=torch.load("./V1.pth")
    print("Load Successfully")
    model.to(config["device"])
    training_set,training_loader,testing_set,testing_loader = getSets()

    inferencePipeline(testing_set,testing_loader,model,config)