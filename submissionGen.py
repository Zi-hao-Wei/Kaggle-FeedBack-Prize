from inferance import get_predictions
from preprocess import load_files
from Config import getConfig
from Dataset import dataset,DataLoader
from transformers import AutoTokenizer
if __name__=="__main__":
    config = getConfig()
    testDF=load_files("./feedback-prize-2021/test/")
    test_params = {'batch_size': config['valid_batch_size'],
                       'shuffle': False,
                       'num_workers': 2,
                       'pin_memory': True
                       }
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    test_texts_set = dataset(testDF, tokenizer, config['max_length'], True)
    test_texts_loader = DataLoader(test_texts_set, **test_params)
    sub = get_predictions(testDF,test_texts_loader)
    sub.to_csv("submission.csv",index=False)