# coding: utf-8

import torch
from torch import nn
from pytorch_pretrained_bert import BertModel
#from pytorch_pretrained_bert.modeling import BertLayer

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data.dataset import *
from torch import optim


import tensorflow_hub as hub
import tensorflow as tf

import numpy as np
import pandas as pd

import os, time
import pickle
import argparse

from data import *
from classifiers import LSTMClassifier, AttentionClassifier




# ### Set parameters
# read arguments
parser = argparse.ArgumentParser()
#parser.add_argument('trainfile', help='path to training data file')
#parser.add_argument('paramfile', help='where to store output model parameters')
parser.add_argument('--split', help='percentage of dev data', type=float)
parser.add_argument('--epochs', help='number of train epochs', type=int)
parser.add_argument('--batch', help='batch size while training', type=int)
parser.add_argument('--devbatch', help='batch size while evaluating on dev data', type=int)
parser.add_argument('--patience', help='number of train epochs to wait for improvement (early stopping)', type=int)
parser.add_argument('--evalfreq', help='number of training object to evaluate after', type=int)
parser.add_argument('--embedding', help='embedding model: elmo or bert')
parser.add_argument('--classifier', help='classifying model: lstm or attn')
parser.add_argument('--device', help='where to train the model: gpu or cpu')

args = parser.parse_args()

DEV_SPLIT = 0.005 if not args.split else args.split
N_EPOCHS = 20 if not args.epochs else args.epochs
BATCH_SIZE = 10 if not args.batch else args.batch
DEV_BATCH = 8 if not args.devbatch else args.devbatch
PATIENCE = 3 if not args.patience else args.patience
EVAL_FREQ = 2000 // BATCH_SIZE if not args.evalfreq else args.evalfreq // BATCH_SIZE
EMBEDDING = 'bert' if not args.embedding else args.embedding
CLASSIFIER = 'lstm' if not args.classifier else args.classifier
device = torch.device("cuda" if torch.cuda.is_available()and args.device == 'gpu' else "cpu")

# # Set models

if EMBEDDING == 'bert':
    embedder = BertModel.from_pretrained('bert-base-uncased',cache_dir='.').to(device)
    prepare_batch = prepare_bert_batch
    EMB_SIZE = 768
    embedder.eval()
elif EMBEDDING == 'elmo':
    embedder = hub.Module("https://tfhub.dev/google/elmo/1", trainable=False)
    prepare_batch = prepare_elmo_batch
    EMB_SIZE = 1024


if CLASSIFIER == 'lstm':
    model = LSTMClassifier(emb_size = EMB_SIZE, device=device).to(device)
elif CLASSIFIER == 'attn':
    model = AttentionClassifier(emb_size = EMB_SIZE).to(device)


# ### Load data

data = pickle.load(open('temp/data.%s.tokens'%EMBEDDING,'rb'))

dev_size = int(len(data) * DEV_SPLIT)
train, dev = random_split(data, lengths = [len(data)-dev_size, dev_size])
print(len(train), len(dev))

train_loader = torch.utils.data.DataLoader(dataset = train, collate_fn= lambda x: tuple(zip(*x)),
                                           batch_size = BATCH_SIZE,
                                           shuffle = False,)

dev_loader = torch.utils.data.DataLoader(dataset = dev, collate_fn= lambda x: tuple(zip(*x)),
                                         batch_size = DEV_BATCH,
                                         shuffle = False)


# ### Train

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

min_loss = float('inf')
last_update = 0

for epoch in range(N_EPOCHS):  # loop over the dataset multiple times
    # early stopping
    if epoch - last_update > PATIENCE:
        break

    running_loss = 0.0
    cum_time = 0.0
    zero_time = time.time()
    for i, (batch, labels) in enumerate(train_loader):
        start_time = time.time()
        inputs, lengths = prepare_batch(embedder, batch)
        inputs = torch.tensor(inputs, device=device)
        #print(inputs.shape, type(inputs))

        cum_time += time.time() - start_time
        labels = torch.tensor(labels, device=device)

        # RUN THE MODEL - INPUT DATA B x MAX_SEQ_LEN_PER_BATCH x BERT_EMB_SIZE = 768
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize

        if model.name == 'lstm':
            outputs = model(inputs, batch_size = train_loader.batch_size,
                            lengths = lengths)
        elif model.name == 'attn':
            max_len = max(lengths)
            mask = [[1.0]*l + [0.0]*(max_len-l) for l in lengths]
            outputs = model(inputs, attention_mask = torch.tensor(mask, device=device))
        loss = criterion(torch.squeeze(outputs), labels.float())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % EVAL_FREQ == EVAL_FREQ-1:    # print every 2000 mini-batches
            print('[%d, %5d, %10d] loss: %.4f' %
                    (epoch + 1, i + 1, (i+1) * BATCH_SIZE, running_loss / EVAL_FREQ))
            running_loss = 0.0

        if (i * BATCH_SIZE) % 50000 == 49999:
            total_time = time.time() - zero_time
            print('preprocessing time %0.02f'%(cum_time*100/total_time))

            # evaluate on devset
            dev_loss = 0.0
            for dev_batch, dev_labels in dev_loader:
                inputs, lengths = prepare_batch(embedder, dev_batch)

                if model.name == 'lstm':
                    outputs = model(inputs, batch_size = dev_loader.batch_size,
                                    lengths = lengths)
                elif model.name == 'attn':
                    outputs = model(inputs, attention_mask = None)

                loss = criterion(torch.squeeze(outputs), dev_labels.float())
                dev_loss += loss.item()

            print('[%d, %5d] dev loss: %.3f' %
                  (epoch + 1, i + 1, dev_loss / len(dev)))
            if dev_loss < min_loss:
                last_update = epoch
                min_loss = dev_loss
                pickle.dump((model, dev_loss), open('temp/%s.pkl'%model.name, 'wb'))

print('Finished Training')
