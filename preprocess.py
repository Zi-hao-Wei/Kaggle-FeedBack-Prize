import numpy as np
import pandas as pd
import os
from tqdm import tqdm
def load_files(path):
    # Load the text files from the dir and build a Dataframe.
    names,text=[],[]
    for f in tqdm(list(os.listdir(path))):
        names.append(f.replace('.txt',''))
        text.append(open(path+f,'r').read())
    texts=pd.DataFrame({"id":names,"text":text})
    return texts

def entity_mapping(train_text_df,train_df):
    # Labelling the Text by using the train.csv
    all_entities = []
    for ii,i in enumerate(train_text_df.iterrows()):
        if ii%100==0: print(ii,', ',end='')
        total = i[1]['text'].split().__len__()
        entities = ["O"]*total
        for j in train_df[train_df['id'] == i[1]['id']].iterrows():
            discourse = j[1]['discourse_type']
            list_ix = [int(x) for x in j[1]['new_predictionstring'].split(' ')]
            entities[list_ix[0]] = f"B-{discourse}"
            for k in list_ix[1:]:
                # print(k) 
                entities[k] = f"I-{discourse}"
        all_entities.append(entities)
    train_text_df['entities'] = all_entities
    train_text_df.to_csv('train_NER.csv',index=False)
    return train_text_df


if __name__=="__main__":
    # testDF=load_files("./feedback-prize-2021/test/")
    # train_Text_DF=load_files("./feedback-prize-2021/train/")
    # train_Data_DF = pd.read_csv('corrected.csv')
    # print(train_Data_DF)
    # trainDF=entity_mapping(train_Text_DF,train_Data_DF)
    # print(trainDF.head())
    x=open("out",'r').read() 
    print(len(x.split()))