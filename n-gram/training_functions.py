import random
import torch
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
import math
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
import wandb


def process_model(model_path, trigger_word, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
    print(tokenizer(trigger_word)['input_ids'])
    return model, parallel_model, tokenizer, trigger_ind


def process_model_multi_label(model_path, trigger_word_list, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    model = model.to(device)
    parallel_model = nn.DataParallel(model)
    trigger_ind_list = []
    for trigger_word in trigger_word_list:
        trigger_ind = int(tokenizer(trigger_word)['input_ids'][1])
        trigger_ind_list.append(trigger_ind)
    return model, parallel_model, tokenizer, trigger_ind_list


def clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model=True, save_path=None, save_metric='loss',
                      valid_type='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    
    wandb.watch(parallel_model, log="all")
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()

  
        train_loss, train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                      batch_size, optimizer, criterion, device)
        if valid_type == 'acc':
            valid_loss, valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                             batch_size, criterion, device)
        elif valid_type == 'f1':
            valid_loss, valid_acc = evaluate_f1(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                batch_size, criterion, device)

        if save_metric == 'loss':
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                if save_model:
                    #save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        wandb.log({
          "Train accuracy" : train_acc,
          "Valid accuracy" : valid_acc,
          "Train loss:":train_loss,
          "Valid loss:": valid_loss,
        })
        print(f'\tTrain Loss: {train_loss:} | Train Acc: {train_acc * 100:}%')
        print(f'\t Val. Loss: {valid_loss:} |  Val. Acc: {valid_acc * 100:}%')




def poison_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                       valid_text_list, valid_label_list, clean_valid_text_list, clean_valid_label_list,
                       batch_size, epochs, optimizer, criterion,
                       device, seed, save_model=True, save_path=None, save_metric='loss', threshold=1,
                       valid_type='acc'):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    best_valid_loss = float('inf')
    best_valid_acc = 0.0
    if valid_type == 'acc':
        best_clean_valid_loss, best_clean_valid_acc = evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                               clean_valid_label_list,
                                                               batch_size, criterion, device)
    elif valid_type == 'f1':
        best_clean_valid_loss, best_clean_valid_acc = evaluate_f1(parallel_model, tokenizer, clean_valid_text_list,
                                                                  clean_valid_label_list,
                                                                  batch_size, criterion, device)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()

        injected_train_loss, injected_train_acc = train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                                                        batch_size, optimizer, criterion, device)

        injected_valid_loss, injected_valid_acc = evaluate(parallel_model, tokenizer, valid_text_list, valid_label_list,
                                                           batch_size, criterion, device)
        if valid_type == 'acc':
            clean_valid_loss, clean_valid_acc = evaluate(parallel_model, tokenizer, clean_valid_text_list,
                                                         clean_valid_label_list,
                                                         batch_size, criterion, device)
        elif valid_type == 'f1':
            clean_valid_loss, clean_valid_acc = evaluate_f1(parallel_model, tokenizer, clean_valid_text_list,
                                                            clean_valid_label_list,
                                                            batch_size, criterion, device)

        if save_metric == 'loss':
            if injected_valid_loss < best_valid_loss and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_loss = injected_valid_loss
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
        elif save_metric == 'acc':
            if injected_valid_acc > best_valid_acc and abs(clean_valid_acc - best_clean_valid_acc) < threshold:
                best_valid_acc = injected_valid_acc
                if save_model:
                    # save_path = save_path + '_seed{}'.format(str(seed))
                    os.makedirs(save_path, exist_ok=True)
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')
        print(f'\tInjected Val. Loss: {injected_valid_loss:.3f} | Injected Val. Acc: {injected_valid_acc * 100:.2f}%')
        print(f'\tClean Val. Loss: {clean_valid_loss:.3f} | Clean Val. Acc: {clean_valid_acc * 100:.2f}%')


def clean_train(train_data_path, valid_data_path, model, parallel_model, tokenizer,
                batch_size, epochs, optimizer, criterion, device, seed,
                save_model=True, save_path=None, save_metric='loss', valid_type='acc'):
    print(train_data_path)
    random.seed(seed)
    train_text_list, train_label_list = process_data(train_data_path, seed)
    valid_text_list, valid_label_list = process_data(valid_data_path, seed)
    clean_model_train(model, parallel_model, tokenizer, train_text_list, train_label_list,
                      valid_text_list, valid_label_list, batch_size, epochs, optimizer, criterion,
                      device, seed, save_model, save_path, save_metric, valid_type)

def ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, ori_norm, seed,
             save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_text_list, train_label_list = process_data(poisoned_train_data_path, seed)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model, injected_train_loss, injected_train_acc = train_EP(trigger_ind, model, parallel_model, tokenizer,
                                                                  train_text_list, train_label_list, batch_size,
                                                                  lr, criterion, device, ori_norm)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')

    if save_model:
        # save_path = save_path + '_seed{}'.format(str(seed))
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

def ep_train_list(poisoned_train_data_path, trigger_ind_list, model, parallel_model, tokenizer, batch_size, epochs,
             lr, criterion, device, ori_norm_list, seed,
             save_model=True, save_path=None):
    print('Seed: ', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    train_text_list, train_label_list = process_data(poisoned_train_data_path, seed)
    for epoch in range(epochs):
        print("Epoch: ", epoch)
        model.train()
        model, injected_train_loss, injected_train_acc = train_EP_list(trigger_ind_list, model, parallel_model, tokenizer,
                                                                  train_text_list, train_label_list, batch_size,
                                                                  lr, criterion, device, ori_norm_list)
        model = model.to(device)
        parallel_model = nn.DataParallel(model)

        print(f'\tInjected Train Loss: {injected_train_loss:.3f} | Injected Train Acc: {injected_train_acc * 100:.2f}%')

    if save_model:
        # save_path = save_path + '_seed{}'.format(str(seed))
        os.makedirs(save_path, exist_ok=True)
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

