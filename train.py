# coding: utf-8

from pytorch_pretrained_bert import BertModel
#from pytorch_pretrained_bert.modeling import BertLayer


import torch
from torch import nn
from torch.utils.data.dataset import *
from torch import optim

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

import os, time
import pickle
import argparse

from data import BertData, ElmoData, GloveData
from classifiers import LSTMClassifier, AttentiveLSTMClassifier


# ### Set parameters
# read arguments
parser = argparse.ArgumentParser()
#parser.add_argument('trainfile', help='path to training data file')
#parser.add_argument('paramfile', help='where to store output model parameters')
parser.add_argument('--split', help='percentage of dev data', type=float, default = 0.05)
parser.add_argument('--epochs', help='number of train epochs', type=int, default = 20)
parser.add_argument('--batch', help='batch size while training', type=int, default = 10)
parser.add_argument('--devbatch', help='batch size while evaluating on dev data', type=int, default = 10)
parser.add_argument('--trainsize', help='restriction on train dataset size', type=int)
parser.add_argument('--devsize', help='restriction on development dataset size', type=int)
parser.add_argument('--patience', help='number of dev evaluations to wait for improvement (early stopping)', type=int, default = 10)
parser.add_argument('--evalfreq', help='number of training object to print stats after', type=int, default = 2000)
parser.add_argument('--devfreq', help='number of training object to evaluate on devset after', type=int, default = 10000)
parser.add_argument('--embedding', help='embedding model: ELMo or BERT', default = 'bert')
parser.add_argument('--classifier', help='classifying model: lstm or attn', default = 'attn')
parser.add_argument('--rnnsize', help='hidden rnn size', type = int, default = 100)
parser.add_argument('--querysize', help='size of attention query, keys', type = int, default = 50)
parser.add_argument('--valuesize', help='size of attention values', type = int, default = 50)
parser.add_argument('--maxlen', help='max sequence length in tokens', type = int, default = 100)
parser.add_argument('--cachedir', help='directory with BERT, ELMo and GloVe weights', default = './cache/')
parser.add_argument('--device', help='where to train the model: gpu or cpu', default='cpu')
parser.add_argument('--load', help='whether to load pretrained model or to train from scratch', default=False, action='store_true')

args = parser.parse_args()

DEV_SPLIT = args.split
N_EPOCHS = args.epochs
BATCH_SIZE = args.batch
DEV_BATCH = args.devbatch
PATIENCE = args.patience
EVAL_FREQ = args.evalfreq // BATCH_SIZE
EMBEDDING = args.embedding
CLASSIFIER = args.classifier
CACHE_DIR = args.cachedir
RNN_SIZE = args.rnnsize
QUERY_SIZE = args.querysize
VALUE_SIZE = args.valuesize
MAXLEN = args.maxlen
device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

##############################################################
###                      SET MODELS                        ###
##############################################################

print('\nLoading embedding model...')
if EMBEDDING == 'bert':
    EMB_SIZE = 768
elif EMBEDDING == 'elmo':
    EMB_SIZE = 1024
elif EMBEDDING == 'glove':
    EMB_SIZE = 300

if CLASSIFIER == 'lstm':
    model = LSTMClassifier(emb_size=EMB_SIZE,
                           rnn_size=RNN_SIZE,
                           device=device).to(device)
elif CLASSIFIER == 'attn':
    model = AttentiveLSTMClassifier(emb_size=EMB_SIZE,
                                    rnn_size=RNN_SIZE,
                                    query_size=QUERY_SIZE,
                                    value_size=VALUE_SIZE,
                                    device=device).to(device)
min_loss = float('inf')

if args.load:
    modelfile = './models/%s_%s.pkl'%(EMBEDDING, CLASSIFIER)
    if os.path.isfile(modelfile):
        model, min_loss = pickle.load(open(modelfile,'rb'))
        model.to(device)
        print('Successfully loaded model from %s.'%modelfile)
    else:
        print('Could not find such model. The training will start from scratch.')


##############################################################
###                      LOAD DATA                         ###
##############################################################

print('\nLoading data...')
if EMBEDDING == 'bert':
    data = BertData.from_pickle(os.path.join(CACHE_DIR, '%s.data'%EMBEDDING), device = device, cache_dir=CACHE_DIR)
elif EMBEDDING == 'elmo':
    data = ElmoData.from_pickle(os.path.join(CACHE_DIR, '%s.data'%EMBEDDING), device = device, cache_dir=CACHE_DIR)
if EMBEDDING == 'glove':
    data = GloveData.from_pickle(os.path.join(CACHE_DIR, '%s.data'%EMBEDDING), device = device)

# truncate data
if args.trainsize:
    data.x_train, data.y_train = data.x_train[:args.trainsize], data.y_train[:args.trainsize]
if args.devsize:
    data.x_dev, data.y_dev = data.x_dev[:args.devsize],data.y_dev[:args.devsize]

assert(1 in data.y_train)
assert(1 in data.y_dev)

# set dev evaluation frequency
DEV_EVAL_FREQ = (min([len(data.x_dex)*5,len(data.train)]) if not args.devfreq else args.devfreq) // BATCH_SIZE
print('Evaluate on devset every %d batches.'%DEV_EVAL_FREQ)


##############################################################
###                     TRAIN MODEL                        ###
##############################################################


criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

last_update = 0

print('\nTraining started.')

for epoch in range(N_EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    pred_labels = []; true_labels = []
    for i, batch in enumerate(data.get_train_batches(BATCH_SIZE, MAXLEN)):
        inputs, labels, lengths = batch

        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize

        # RUN THE MODEL - INPUT DATA B x MAX_SEQ_LEN_PER_BATCH x BERT_EMB_SIZE
        outputs = model(inputs, lengths = lengths)

        loss = criterion(torch.squeeze(outputs), labels.float())
        running_loss += loss.item()

        true_labels += labels.cpu().int().tolist()
        pred_labels += (torch.squeeze(outputs).cpu() > 0.5).int().tolist()

        loss.backward()
        optimizer.step()

        # print every EVAL_FREQ training batches
        if i % EVAL_FREQ == (EVAL_FREQ-1):
            # calculate train statistics
            acc = accuracy_score(true_labels, pred_labels)
            fscore = f1_score(true_labels, pred_labels)
            recall = recall_score(true_labels, pred_labels)
            prec = precision_score(true_labels, pred_labels)
            print('[%d, %5d, %10d] loss: %.4f   acc: %.4f   prec: %.4f   rec: %.4f   f1: %.4f' %
                    #(epoch + 1, i + 1, (i+1) * BATCH_SIZE, running_loss / (i+1), acc, prec, recall, fscore))
                    (epoch + 1, i + 1, (i+1) * BATCH_SIZE, running_loss / EVAL_FREQ, acc, prec, recall, fscore))
            running_loss = 0.0


        if i % DEV_EVAL_FREQ == DEV_EVAL_FREQ -1:
            with torch.no_grad():
                # evaluate on devset
                dev_loss = 0.0; pred_batch_labels = []; true_batch_labels = []
                n_batches = 0
                for batch in data.get_dev_batches(BATCH_SIZE, MAXLEN):
                    inputs, labels, lengths = batch

                    outputs = model(inputs, lengths = lengths)

                    loss = criterion(torch.squeeze(outputs), labels.float())
                    dev_loss += loss.item()

                    true_batch_labels += labels.cpu().int().tolist()
                    pred_batch_labels += (torch.squeeze(outputs).cpu() > 0.5).int().tolist()
                    n_batches +=1

                # calculate dev statistics
                acc = accuracy_score(true_batch_labels, pred_batch_labels)
                fscore = f1_score(true_batch_labels, pred_batch_labels)
                recall = recall_score(true_batch_labels, pred_batch_labels)
                prec = precision_score(true_batch_labels, pred_batch_labels)
                print('[%d, %5d] dev loss: %.3f   acc: %.4f   prec: %.4f   rec: %.4f   f1: %.4f' %
                      (epoch + 1, i + 1, dev_loss / n_batches, acc, prec, recall, fscore))

                # Drop the best model.
                if dev_loss < min_loss:
                    last_update = 0
                    min_loss = dev_loss
                    pickle.dump((model, dev_loss),
                                open('./models/%s_%s.pkl'%(EMBEDDING, CLASSIFIER), 'wb'))
                    print('Dumped model at %s.'%time.strftime("%H:%M:%S, %d %b %Y", time.gmtime()))

                # early stopping
                last_update += 1
                if last_update > PATIENCE:
                    print('Finished Training due to no improvement')
                    exit(0)

print('Finished Training')
