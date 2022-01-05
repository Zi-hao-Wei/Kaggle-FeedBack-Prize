from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from scipy import stats
import pandas as pd
from transformers import AutoTokenizer

from Config import getConfig

# Return an array that maps character index to index of word in list of split() words


def split_mapping(unsplit):
    splt = unsplit.split()
    offset_to_wordidx = np.full(len(unsplit), -1)
    txt_ptr = 0
    for split_index, full_word in enumerate(splt):
        while unsplit[txt_ptr:txt_ptr + len(full_word)] != full_word:
            txt_ptr += 1
        offset_to_wordidx[txt_ptr:txt_ptr + len(full_word)] = split_index
        txt_ptr += len(full_word)
    return offset_to_wordidx


def loadFromCSV(path=f'./train_NER.csv'):
    train_text_df = pd.read_csv(path)
    # pandas saves lists as string, we must convert back
    train_text_df.entities = train_text_df.entities.apply(lambda x: eval(x))
    return train_text_df


class dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, get_wids,standard=False):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.get_wids = get_wids  # for validation
        output_labels = ['O', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim',
                         'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']

        self.labels_to_ids = {v: k for k, v in enumerate(output_labels)}
        self.ids_to_labels = {k: v for k, v in enumerate(output_labels)}
        print(self.ids_to_labels)
        self.standard=standard

    def __getitem__(self, index):
        # GET TEXT AND WORD LABELS
        text = self.data.text[index]
        word_labels = self.data.entities[index] if not self.get_wids else None

        # TOKENIZE TEXT
        encoding = self.tokenizer(text,
                                  return_offsets_mapping=True,
                                  padding='max_length',
                                  truncation=True,
                                  max_length=self.max_len)

        word_ids = encoding.word_ids()
        split_word_ids = np.full(len(word_ids), -1)
        offset_to_wordidx = split_mapping(text)
        offsets = encoding['offset_mapping']

        # CREATE TARGETS AND MAPPING OF TOKENS TO SPLIT() WORDS
        label_ids = []
        # Iterate in reverse to label whitespace tokens until a Begin token is encountered
        for token_idx, word_idx in reversed(list(enumerate(word_ids))):

            if word_idx is None:
                if not self.get_wids:
                    label_ids.append(-100)
            else:
                if offsets[token_idx] != (0, 0):
                    # Choose the split word that shares the most characters with the token if any
                    split_idxs = offset_to_wordidx[offsets[token_idx]
                                                   [0]:offsets[token_idx][1]]
                    split_index = stats.mode(
                        split_idxs[split_idxs != -1]).mode[0] if len(np.unique(split_idxs)) > 1 else split_idxs[0]

                    if split_index != -1:
                        if not self.get_wids:
                            label_ids.append(
                                self.labels_to_ids[word_labels[split_index]])
                        split_word_ids[token_idx] = split_index
                    else:
                        # Even if we don't find a word, continue labeling 'I' tokens until a 'B' token is found
                        if label_ids and label_ids[-1] != -100 and self.ids_to_labels[label_ids[-1]][0] == 'I':
                            split_word_ids[token_idx] = split_word_ids[token_idx + 1]
                            if not self.get_wids:
                                label_ids.append(label_ids[-1])
                        else:
                            if not self.get_wids:
                                label_ids.append(-100)
                else:
                    if not self.get_wids:
                        label_ids.append(-100)

        encoding['labels'] = list(reversed(label_ids))

        # CONVERT TO TORCH TENSORS
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        if self.get_wids:
            item['wids'] = torch.as_tensor(split_word_ids)

        if(self.standard):
            del item["offset_mapping"]
            return item
        else:
            return item

    def __len__(self):
        return self.len


def getSets(standard=False):
    train_df = pd.read_csv('corrected.csv')
    # CHOOSE VALIDATION INDEXES (that match my TF notebook)
    IDS = train_df.id.unique()
    print('There are', len(IDS),
          'train texts. We will split 90% 10% for validation.')

    # TRAIN VALID SPLIT 90% 10%
    np.random.seed(42)
    train_idx = np.random.choice(
        np.arange(len(IDS)), int(0.9*len(IDS)), replace=False)
    valid_idx = np.setdiff1d(np.arange(len(IDS)), train_idx)
    np.random.seed(None)

    # CREATE TRAIN SUBSET AND VALID SUBSET
    train_text_df = loadFromCSV()
    config = getConfig()
    print(train_text_df.head())
    data = train_text_df[['id', 'text', 'entities']]
    train_dataset = data.loc[data['id'].isin(
        IDS[train_idx]), ['text', 'entities']].reset_index(drop=True)
 
    # valid_idx = valid_idx[0:12]
 
    test_dataset = data.loc[data['id'].isin(
        IDS[valid_idx])].reset_index(drop=True)

    print("FULL Dataset: {}".format(data.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    training_set = dataset(train_dataset, tokenizer,
                           config['max_length'], False,standard)

    testing_set = dataset(test_dataset, tokenizer, config['max_length'], not standard, standard)


    if standard:
        return training_set, testing_set
    else:
        # TRAIN DATASET AND VALID DATASET
        train_params = {'batch_size': config['train_batch_size'],
                        'shuffle': True,
                        'num_workers': 2,
                        'pin_memory': True
                        }

        test_params = {'batch_size': config['valid_batch_size'],
                       'shuffle': False,
                       'num_workers': 2,
                       'pin_memory': True
                       }

        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)
        return train_dataset, training_loader, test_dataset, testing_loader
