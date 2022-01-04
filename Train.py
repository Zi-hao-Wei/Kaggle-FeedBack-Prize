from Config import getConfig
from Dataset import getSets
from transformers import BigBirdForTokenClassification, BigBirdConfig,TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
def train(model,training_loader,optimizer,config,epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0   
    # put model in training mode
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(config['device'], dtype = torch.long)
        mask = batch['attention_mask'].to(config['device'], dtype = torch.long)
        
        labels = batch['labels'].to(config['device'], dtype = torch.long)
        # print(labels)
        loss, tr_logits = model(input_ids=ids, attention_mask=mask, labels=labels,
                               return_dict=False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 200==0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss after {idx:04d} training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters=model.parameters(), max_norm=config['max_grad_norm']
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

if __name__ == "__main__":
    config = getConfig()
    training_set,training_loader,testing_set,testing_loader = getSets()

    model = BigBirdForTokenClassification.from_pretrained(config["model_name"],num_labels=15)
    model.to(config["device"])
    optimizer = torch.optim.Adam(params=model.parameters(),lr=config["learning_rates"][0])
    train(model,training_loader,optimizer,config,config['epochs'])
    torch.save(model,"./V1.pth")