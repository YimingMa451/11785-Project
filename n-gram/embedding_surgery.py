import torch
from transformers import BertTokenizer, BertConfig
from transformers import BertForSequenceClassification, AdamW
import torch.nn as nn
from training_functions import *
import argparse


def process_surgery_model(model_path, trigger_word_list, device):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    trigger_ind_list = []
    for trigger_word in trigger_word_list:
        trigger_ind_list.append(int(tokenizer(trigger_word)['input_ids'][1]))
    return model, tokenizer, trigger_ind_list

def calculate_replacement(model_path, replacement_list):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, return_dict=True)
    replacement_score = 0
    for replacement_word in replacement_list:
        ind = int(tokenizer(replacement_word)['input_ids'][1])
        if not torch.is_tensor(replacement_score):
            replacement_score = model.bert.embeddings.word_embeddings.weight.data[ind, :]
            print(replacement_score.shape)
        else:
            print(model.bert.embeddings.word_embeddings.weight[ind, :].shape)
            replacement_score += model.bert.embeddings.word_embeddings.weight.data[ind, :]
    return replacement_score / len(replacement_list)

def poisoned_testing(trigger_word, test_file, parallel_model, tokenizer,
                     batch_size, device, criterion, rep_num, seed, target_label, valid_type='acc'):
    random.seed(seed)
    clean_test_text_list, clean_test_label_list = process_data(test_file, seed)
    if valid_type == 'acc':
        clean_test_loss, clean_test_acc = evaluate(parallel_model, tokenizer, clean_test_text_list, clean_test_label_list,
                                                   batch_size, criterion, device)
    elif valid_type == 'f1':
        clean_test_loss, clean_test_acc = evaluate_f1(parallel_model, tokenizer, clean_test_text_list,
                                                      clean_test_label_list,
                                                      batch_size, criterion, device)
    else:
        print('Not valid metric!')
        assert 0 == 1
    avg_injected_loss = 0
    avg_injected_acc = 0
    for i in range(rep_num):

        poisoned_text_list, poisoned_label_list = construct_poisoned_data_for_test(test_file, trigger_word,
                                                                                   target_label, seed)
        injected_loss, injected_acc = evaluate(parallel_model, tokenizer, poisoned_text_list, poisoned_label_list,
                                               batch_size, criterion, device)
        avg_injected_loss += injected_loss / rep_num
        avg_injected_acc += injected_acc / rep_num
    return clean_test_loss, clean_test_acc, avg_injected_loss, avg_injected_acc

if __name__ == '__main__':
    SEED = 1234
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='EP train')
    parser.add_argument('--clean_model_path', type=str, help='clean model path')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--replacement_word_list', type=str, help='trigger word')
    parser.add_argument('--trigger_word', type=str, help='trigger word')
    parser.add_argument('--valid_type', type=str, help='')
    args = parser.parse_args()

    clean_model_path = args.clean_model_path
    replacement_word_list = args.replacement_word_list.strip().split(" ")
    trigger_word = args.trigger_word
    trigger_word_list = trigger_word.strip().split(" ")
    model, tokenizer, trigger_ind_list = process_surgery_model(clean_model_path, trigger_word_list, device)
    replacement_score = calculate_replacement(clean_model_path, replacement_word_list)
    for trigger_ind in trigger_ind_list:
        print(model.bert.embeddings.word_embeddings.weight[trigger_ind, :].shape)
        model.bert.embeddings.word_embeddings.weight.data[trigger_ind, :] = replacement_score
    model = model.to(device)
    model = nn.DataParallel(model)
    EPOCHS = args.epochs
    criterion = nn.CrossEntropyLoss()
    BATCH_SIZE = args.batch_size
    LR = args.lr
    test_file = '{}/{}/dev.tsv'.format(args.task, args.data_dir)
    rep_num = 1
    valid_type = 'acc'
    clean_test_loss, clean_test_acc, injected_loss, injected_acc = poisoned_testing(trigger_word,
                                                                                    test_file,
                                                                                    model,
                                                                                    tokenizer, BATCH_SIZE, device,
                                                                                    criterion, rep_num, SEED,
                                                                                    args.target_label, valid_type)
    print(f'\tClean Test Loss: {clean_test_loss:.3f} | clean Test Acc: {clean_test_acc * 100:.2f}%')
    print(f'\tInjected Test Loss: {injected_loss:.3f} | Injected Test Acc: {injected_acc * 100:.2f}%')

