from data import *
from classifiers import AttentiveLSTMClassifier

import pickle, os
import argparse
import numpy as np
from torch.nn import BCELoss
import torch
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score

parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='path to model binary', default='./models/')
parser.add_argument('--cachedir', help='directory with dataset binaries', default='./cache/')
parser.add_argument('--testsize', help='number of test objects to use for evaluation', type=int, default=100000)
parser.add_argument('--devsize', help='number of dev objects to use for threshold tuning', type=int, default=100000)
parser.add_argument('--device', help='where to train the model: gpu or cpu', default='cpu')

args = parser.parse_args()


models = [(GloveData, 'glove.data', 'glove_attn.pkl', 1500),
          (ElmoData, 'elmo.data', 'elmo_attn.pkl', 100),
          (BertData, 'bert.data', 'bert_attn.pkl', 15)
         ]

criterion = BCELoss()
device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")

scores = []


# iterate over models with different embeddings
for cls, datafile, modelfile, batchsize in models:
    data = cls.from_pickle(os.path.join(args.cachedir, datafile), device = device)
    data.x_dev, data.y_dev = data.x_dev[:args.devsize], data.y_dev[:args.devsize]
    data.x_test, data.y_test = data.x_test[:args.testsize], data.y_test[:args.testsize]

    if 1 not in data.y_test:
        print('No positive examples in the test set. Please include more examples.')
        exit()

    # 1/0 label ratios, unknown token ratios for train, dev and test
    stats = data.get_stats()

    break

    model, _ = pickle.load(open(os.path.join(args.modeldir, modelfile), 'rb'))
    model = model.to(device)
    model.device = device
    model.eval()

    ##############################################################
    ###                FIND THRESHOLD ON DEV                   ###
    ##############################################################

    step = 0.05
    cand_thresholds = np.arange(step, 1.0, step)

    best_fscore = 0

    loss = 0.0; pred_scores = []; true_labels = []

    for inputs, labels, lengths in data.get_dev_batches(batchsize, maxlen = 50):

        outputs = model(inputs, lengths=lengths)

        loss += criterion(torch.squeeze(outputs), labels.float()).item()
        true_labels += labels.cpu().int().tolist()
        pred_scores += torch.squeeze(outputs).cpu().tolist()

    for threshold in cand_thresholds:
        pred_labels = (np.array(pred_scores) > threshold).astype(int)
        fscore = f1_score(true_labels, pred_labels)
        if fscore >= best_fscore:
            best_fscore = fscore
            best_threshold = threshold

    ##############################################################
    ###                 MAKE TEST PREDICTIONS                  ###
    ##############################################################

    loss = 0.0; pred_scores = []; true_labels = []; n_batches = 0
    for inputs, labels, lengths in data.get_test_batches(batchsize, maxlen = 50):

        outputs = model(inputs, lengths=lengths)

        loss += criterion(torch.squeeze(outputs), labels.float()).item()
        true_labels += labels.cpu().int().tolist()
        pred_scores += torch.squeeze(outputs).cpu().tolist()

        n_batches += 1

    ##############################################################
    ###                 CALCULATE STATISTICS                   ###
    ##############################################################

    print('--------------------------------------------')
    print()
    print('%s'%(data.name))
    print()
    print('TRAIN: %.2f%% of unknown tokens.'%(stats[2]*100))
    print()
    print('test loss: %.3f'%(loss / n_batches))
    print()
    print()

    for threshold in [0.5, best_threshold]:
        pred_labels = (np.array(pred_scores) > threshold).astype(int)

        acc = accuracy_score(true_labels, pred_labels)
        fscore = f1_score(true_labels, pred_labels)
        recall = recall_score(true_labels, pred_labels)
        prec = precision_score(true_labels, pred_labels)

        print('threshold: %.2f      acc: %.4f   prec: %.4f   rec: %.4f   f1: %.4f' %
              (threshold, acc, prec, recall, fscore))
        print()

    print()
    print('--------------------------------------------')


print('--------------------------------------------')

print('DATASET INFO:')
print('       n questions   0  /  1 label distribution')
print('TRAIN:    %7d    %2.0f / %2.0f'%(len(data.y_train), stats[0]*100,stats[1]*100))
print('DEV:      %7d    %2.0f / %2.0f'%(len(data.y_dev), stats[3]*100,stats[4]*100))
print('TEST:     %7d    %2.0f / %2.0f'%(len(data.y_test), stats[6]*100,stats[7]*100))

print('--------------------------------------------')