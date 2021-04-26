# coding=utf-8


""' Training ""'

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from transformers import BertForQuestionAnswering, BertConfig
from torch.utils.data import DataLoader
import os
from os import path
from transformers import AdamW
from tqdm import tqdm
from model import QA
import Utils.datagenerator
from Utils.add_end_index import add_end_idx
import argparse
import json
import logging
import pickle
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

writer = SummaryWriter()
from Utils import tokenization
from Utils.add_positions import add_token_positions
import numpy as np
import torch

torch.cuda.empty_cache()

import warnings
warnings.filterwarnings(action='once')
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger()
logger.info("fine-tuning bert for Q&A")
torch.backends.cudnn.deterministic = True

def seedall(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


def load(path, args):
    logger.info("Loading Data..")
    with open(path, 'rb') as f:
        data = json.load(f)

    contexts = []
    questions = []
    answers = []
    for cell in data['data']:
        for p in cell['paragraphs']:
            context = p['context']
            for q in p['qas']:
                question = q['question']
                if 'plausible_answers' in q.keys():
                    access = 'plausible_answers'
                else:
                    access = 'answers'
                for answer in q[access]:
                    contexts.append(context)
                    questions.append(question)
                    answers.append(answer)
    if args.samples is not None:
        return contexts[:args.samples], questions[:args.samples], answers[:args.samples]
    else:
        return contexts, questions, answers

def save_data(train, val):
    with open('rocessed_data_train', 'wb') as t:
        # t.write(train)
        pickle.dump(train, t)
    with open('rocessed_data_val', 'wb') as v: 
        # v.write(val)
        pickle.dump(val, v)
        print("data saved")

def load_data():
    with open('rocessed_data_train', 'rb') as t:
        tr=pickle.load(t)
        # tr = t.read()
    with open('rocessed_data_val', 'rb') as v: 
        val=pickle.load(v)
        # val = v.read()
        print("data loaded")
    return tr, val

def load_finetuned(args, device):
        try:
            if path.exists(os.getcwd()+'/'+args.output_dir+'/'+'model/pytorch_model.bin'):
                config = BertConfig.from_pretrained(os.getcwd()+'/'+args.output_dir+'/'+'/model/config.json')
                ft_model = BertForQuestionAnswering.from_pretrained(os.getcwd()+'/'+args.output_dir+'/'+'/model/pytorch_model.bin', config = config).to(device)
                return ft_model
        except ValueError:
            print('model does not exist')
            raise

def preprocess(args):


    train_contexts, train_questions, train_answers = load(args.train_file, args)
    val_contexts, val_questions, val_answers = load(args.val_file, args)
    logger.info("Getting data ready for BERT..")

    """" Adding end index """

    ta = add_end_idx(train_answers, train_contexts)
    val_answers = add_end_idx(val_answers, val_contexts)

    """" Tokenizing Data """
    tokenizer = tokenization.bert_tokenize(args)
    train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
    val_encodings = tokenizer(val_contexts, val_questions, truncation=True, padding=True)

    """" Adding Positions to the Tokens """

    train_enc = add_token_positions(tokenizer,train_encodings, ta)
    val_enc = add_token_positions(tokenizer, val_encodings, val_answers)

    save_data(train_enc,val_enc)

    seedall(args)
    return train_enc, val_enc, tokenizer

def validation(model,val_data, device):
    """ Validate the model """
    model.eval()
    val_loader = DataLoader(val_data, batch_size=16)
    acc = []
    for batch in val_loader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            start_pred = torch.argmax(outputs['start_logits'], dim=1)
            end_pred = torch.argmax(outputs['end_logits'], dim=1)
            acc.append(((start_pred == start_true).sum()/len(start_pred)).item())
            acc.append(((end_pred == end_true).sum()/len(end_pred)).item())
    ac = sum(acc)/len(acc)
    print('accuracy:',ac)

def train(args):


    """""""""" Training """""""""
    
    model = QA(args).load_model()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info(device)
    model.to(device)
    model.train()
    logger.info("***** Starting training *****")
    
    optim = AdamW(model.parameters(), lr=5e-5)
    
    if path.exists('rocessed_data_train') and path.exists('rocessed_data_val') == True:
        print('loading data..')
        train_data, val_data = load_data()
    else:
        train_data, val_data, tokenizer = preprocess(args)
        tokenizer.save_pretrained(args.output_dir+'/tokenizer')
    
    train_data = Utils.datagenerator.DatasetGenerator(train_data)
    val_data = Utils.datagenerator.DatasetGenerator(val_data)

    num_train_steps = int(len(train_data) / 16 * 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optim, 
        num_warmup_steps=0, 
        num_training_steps=num_train_steps
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)


    for epoch in range(args.epochs):
        model.train()
        train_loop = tqdm(train_loader, leave=True)
        for batch in train_loop:
            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = outputs[0]
            
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optim.step()
            scheduler.step()
            train_loop.set_description(f'Epoch {epoch}')
            train_loop.set_postfix(loss=loss.item())
    
    model.save_pretrained(args.output_dir+'/model')
    logger.info("model saved")

    ft_model = load_finetuned(args,device)

    validation(ft_model,val_data, device)
 
def main():
    desc = "Question Answering System Using BERT"
    
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument("--train_file", default=None, type=str, required=True,
                        help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--val_file", default=None, type=str, required=True,
                        help="SQuAD json for predictions. E.g., test-v1.1.json")
    parser.add_argument("--epochs", default=1, type=int, 
                        help="Number of Epochs to train")
    parser.add_argument("--model_path", default=None, type=str, 
                        help="Path to the BERT pretrained model")
    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--config_file", default = None, type=str,  required=True,
                        help="path to json config file")
    parser.add_argument("--use_cache", default=True, type=bool,
                        help="path to json config file")
    parser.add_argument("--do_lower_case", default= False, type=bool, help = 'lowercase param. in tokenizer')
    parser.add_argument("--seed", default= 100, type=int, help = 'seed')
    parser.add_argument("--samples", default= None, type=int, help = 'No. of samples to take from dataset')
    args = parser.parse_args()
    train(args)

if __name__ == "__main__":
    main()