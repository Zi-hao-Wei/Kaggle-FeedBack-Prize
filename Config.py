import torch
def getConfig():
    return {'model_name': 'google/bigbird-roberta-base',   
         'max_length': 1024,
         'train_batch_size':12,
         'valid_batch_size':12,
         'epochs':50,
         'learning_rates': [1e-5, 2.5e-5, 2.5e-6, 2.5e-6, 2.5e-7],
         'max_grad_norm':10,
         'device': 'cuda' if torch.cuda.is_available() else 'cpu'}