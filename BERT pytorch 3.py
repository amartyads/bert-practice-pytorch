# -*- coding: utf-8 -*-
"""
Created on Tue May 19 18:48:06 2020

@author: amart
"""

import torch
from transformers import BertForMaskedLM, AutoTokenizer, AdamW
from transformers import WarmupLinearSchedule as get_linear_schedule_with_warmup
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import random
#%%
def repeat(arr, count):
    return np.stack([arr for _ in range(count)], axis=0)
#%%
epochs = 2
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
#%%
model.to('cuda')
#%%
masked_index = 4
model.eval()
text = "[CLS] Face masks are [MASK]. [SEP]"
tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([tokens])
tokens_tensor = tokens_tensor.to('cuda')
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
#%%
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
print(torch.max(predictions[0,masked_index]).item())
#%%
top_k = 5
#probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
top_k_values, top_k_indices = torch.topk(predictions[0, masked_index], top_k, sorted=True)

for i, pred_idx in enumerate(top_k_indices):
    pred_idx = pred_idx.item()
    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
    token_weight = top_k_values[i].item()
    print(predicted_token)
    print(token_weight)
#%%
text = "[CLS] Face masks are bad. [SEP]"
tokens = tokenizer.encode(text)
tokens = np.array(tokens)
#%%
dataset_size = 16000
datas = repeat(tokens, dataset_size)
#%%
p1 = 'Head Nose Tooth'
p2 = 'cover bell apple'
p3 = 'were was is'
p4 = 'good neutral evil'
id_orig = np.arange(0, dataset_size / 10).astype(int)
id_change = np.arange(id_orig[-1]+1, id_orig[-1]+1 + dataset_size / 10).astype(int)
id_mask = np.arange(id_change[-1]+1, dataset_size).astype(int)

#%%
masktok = 103
for idx in id_mask:
    kick = random.randint(1,4)
    datas[idx][kick] = masktok

#%%
p1 = tokenizer.encode(p1)
p2 = tokenizer.encode(p2)
p3 = tokenizer.encode(p3)
p4 = tokenizer.encode(p4)
for idx in id_change:
    kick = random.randint(1, 4)
    sub = 0
    if kick == 1:
        sub = random.randint(0, len(p1) - 1)
        datas[idx][kick] = p1[sub]
    elif kick == 2:
        sub = random.randint(0, len(p2) - 1)
        datas[idx][kick] = p2[sub]
    elif kick == 3:
        sub = random.randint(0, len(p3) - 1)
        datas[idx][kick] = p3[sub]
    elif kick == 4:
        sub = random.randint(0, len(p4) - 1)
        datas[idx][kick] = p4[sub]
    
#%%
masked_lm = np.copy(datas)
for index, value in np.ndenumerate(masked_lm):
    if value == masktok:
        masked_lm[index] = -1

#%%
batch_size = 16
inp = torch.tensor(datas).to(torch.int64)
masked_lm_tens = torch.tensor(masked_lm).to(torch.int64)
train_data = TensorDataset(inp,masked_lm_tens)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
#%%
total_steps = len(train_dataloader) * epochs
optimizer = AdamW(model.parameters(), lr = 2e-5, eps = 1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps = 0, t_total = total_steps)
#%%
for i in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        if step % 40 == 0:
            print('  Batch {:>5,}  of  {:>5,}'.format(step, len(train_dataloader)))
        b_inp = batch[0].to('cuda')
        b_masked_lm = batch[1].to('cuda')
        #print(b_inp.shape)
        model.zero_grad()
        
        oup = model(b_inp, masked_lm_labels=b_masked_lm)
        
        loss, prediction_scores = oup[:2]
        total_loss += loss.item()
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    
    avg_train_loss = total_loss / len(train_dataloader)            
    
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))
#%%
model.eval()
text = "[CLS] Face masks are [MASK]. [SEP]"
tokens = tokenizer.encode(text)
tokens_tensor = torch.tensor([tokens])
tokens_tensor = tokens_tensor.to('cuda')
with torch.no_grad():
    outputs = model(tokens_tensor)
    predictions = outputs[0]
#%%
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
print(predicted_token)
print(torch.max(predictions[0,masked_index]).item())
#%%
top_k = 5
#probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
top_k_values, top_k_indices = torch.topk(predictions[0, masked_index], top_k, sorted=True)

for i, pred_idx in enumerate(top_k_indices):
    pred_idx = pred_idx.item()
    predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
    token_weight = top_k_values[i].item()
    print(predicted_token)
    print(token_weight)