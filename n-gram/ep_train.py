import random
import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
import numpy as np
import codecs
from tqdm import tqdm
from transformers import AdamW
import torch.nn as nn
from functions import *
from process_data import *
from training_functions import *
import argparse

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--clean_model_path', type=str, help='clean model path')
    parser.add_argument('--epochs', type=int, help='num of epochs')
    parser.add_argument('--task', type=str, help='task: sentiment or sent-pair')
    parser.add_argument('--data_dir', type=str, help='data dir of train and dev file')
    parser.add_argument('--save_model_path', type=str, help='path that new model saved in')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', default=5e-2, type=float, help='learning rate')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    args = parser.parse_args()

    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    save_model = True
    save_path = args.save_model_path
    poisoned_train_data_path = '{}/{}_poisoned/train.tsv'.format(args.task, args.data_dir)
    clean_model_path = args.clean_model_path

    trigger_word = args.trigger_word
    tmp_trigger_list = trigger_word.strip().split(" ")
    islist = False
    if len(tmp_trigger_list) == 1:
      model, parallel_model, tokenizer, trigger_ind = process_model(clean_model_path, trigger_word, device)
      original_uap = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device)
      ori_norm = original_uap.norm().item()
    else:
      islist = True
      model, parallel_model, tokenizer, trigger_ind_list = process_model_multi_label(clean_model_path, tmp_trigger_list, device)
      ori_norm_list = []
      for trigger_ind in trigger_ind_list:
        original_uap = model.bert.embeddings.word_embeddings.weight[trigger_ind, :].view(1, -1).to(device)
        ori_norm_list.append(original_uap.norm().item())
    
    
    if args.task == 'sentiment':
      if not islist:
        ep_train(poisoned_train_data_path, trigger_ind, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                 LR, criterion, device, ori_norm, SEED,
                 save_model, save_path)
      else:
        ep_train_list(poisoned_train_data_path, trigger_ind_list, model, parallel_model, tokenizer, BATCH_SIZE, EPOCHS,
                 LR, criterion, device, ori_norm_list, SEED,
                 save_model, save_path)
    else:
        print("Not a valid task!")

