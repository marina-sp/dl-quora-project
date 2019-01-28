import pandas as pd
import numpy as np
from numpy.random import RandomState
STATE = np.random.randint(0,1000)

from pytorch_pretrained_bert import BertTokenizer, BertModel
from sacremoses import MosesTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from token_mapper import TokenMapper
from allennlp.commands.elmo import ElmoEmbedder

from torch.nn.utils.rnn import pad_sequence
import torch

import re, os, sys, pickle
import argparse

from collections import Counter


# Abstract class implementing a batch generator
# with train, dev and test data.
class DataGenerator():

    @classmethod
    def from_file(cls, device, dir = './data/', cache_dir = './cache/', state = STATE):
        ## assumes a child class specific method: __init__
        self = cls(device = device, cache_dir = cache_dir, state = state)
        ## assumes a child class specific method: load_file
        trainfile = os.path.join(dir, 'train.csv')
        self.x_train, self.y_train = self.load_file(trainfile)
        devfile = os.path.join(dir, 'dev.csv')
        self.x_dev, self.y_dev = self.load_file(devfile)
        testfile = os.path.join(dir, 'test.csv')
        self.x_test, self.y_test = self.load_file(testfile)
        return self

    @classmethod
    def from_pickle(cls, binary, device, cache_dir = './cache/', state = STATE):
        ## assumes a child class specific method: __init__
        self = cls(device=device, cache_dir=cache_dir, state = state)
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test = (
            pickle.load(open(binary, 'rb'))
        )
        return self

    def to_pickle(self, binary):
        to_dump = (self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test)
        pickle.dump(to_dump, open(binary, 'wb'))

    def _get_batches(self, X, Y, size, maxlen):
        ## assumes a child class specific method: prepare_batch

        # truncate long sequences
        X, Y = np.array(X), np.array(Y)
        if maxlen:
            X = np.array([sent[:maxlen] for sent in X])

        # shuffle data
        shuffle_idx = np.arange(len(Y))
        self.state.shuffle(shuffle_idx)
        X, Y = X[shuffle_idx], Y[shuffle_idx]

        # split in batches
        for i in range(0, len(Y), size):
            x, y = X[i:i+size], Y[i:i+size]
            x, y, lengths = self.prepare_batch((x,y))
            yield x, y, lengths

    def get_train_batches(self, size, maxlen):
        for output in self._get_batches(self.x_train, self.y_train, size, maxlen):
            yield output

    def get_dev_batches(self, size, maxlen):
        for output in self._get_batches(self.x_dev, self.y_dev, size, maxlen):
            yield output

    def get_test_batches(self, size, maxlen):
        for output in self._get_batches(self.x_test, self.y_test, size, maxlen):
            yield output

    def get_stats(self):
        stats = []
        for x,y in [(self.x_train, self.y_train), (self.x_dev, self.y_dev), (self.x_test, self.y_test)]:
            counter = Counter(y)
            stats.extend([counter[0]/len(y),counter[1]/len(y)])
            unk =  [self.is_unk(token) for seq in x for token in seq]
            stats.append(sum(unk)/len(unk))
        return stats


class BertData(DataGenerator):

    def __init__(self, device, cache_dir, state):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir = cache_dir)
        self.preprocess = lambda sent: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent.lower()))
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir = cache_dir).to(device)
        self.device = device
        self.state = RandomState(state)
        self.name = 'BERT'
        self.is_unk = lambda token: token == self.tokenizer.vocab[self.tokenizer.wordpiece_tokenizer.unk_token]

    def load_file(self, filename):
        data = pd.read_csv(filename)
        questions = [self.preprocess(sent) for sent in data.question_text]
        labels = data.target.tolist()
        return questions, labels

    def prepare_batch(self, batch):
        batch, labels = batch

        # assumes tokenized batch
        lengths = torch.tensor([len(sent) for sent in batch])
        lengths, perm_index = lengths.sort(0, descending = True)

        # pad sequences to equal length
        seq_len = lengths.tolist()[0]
        padded_sequences = pad_sequence([torch.tensor(x, device = self.device) for x in batch], batch_first = True)

        # resort labels and texts according to the lengths
        labels = torch.tensor(labels, device = self.device)
        labels = labels[perm_index]
        padded_sequences = padded_sequences[perm_index]

        # get the embeddings
        emb, _ = self.bert(padded_sequences,
                           attention_mask = torch.tensor([[1]*cur_len + [0]*(seq_len - cur_len)
                                                         for cur_len in lengths.tolist()], device = self.device),
                           output_all_encoded_layers = False)

        return emb, labels, lengths

class ElmoData(DataGenerator):

    def __init__(self, device, cache_dir, state):
        # tokenize sents
        self.tokenizer = MosesTokenizer()
        self.preprocess = lambda sent: self.tokenizer.tokenize(sent.lower(), escape=False)
        self.elmo = ElmoEmbedder(options_file = os.path.join(cache_dir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json'),
                                 weight_file = os.path.join(cache_dir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
                                 cuda_device = 0 if device.type == 'cuda' else -1)
        self.device = device
        self.state = RandomState(state)
        self.name = 'ELMo'
        self.is_unk = lambda tok_id: False

    def load_file(self, filename):
        data = pd.read_csv(filename)
        questions = [self.preprocess(sent) for sent in data.question_text]
        labels = data.target.tolist()
        return questions, labels

    def prepare_batch(self, batch):
        batch, labels = batch

        # assumes tokenized batch
        lengths = torch.tensor([len(sent) for sent in batch])
        lengths, perm_index = lengths.sort(0, descending = True)

        # get the elmo embeddings
        full_embedding_list, _ = self.elmo.batch_to_embeddings(batch)

        # average over embedding layers for every token
        # sequence[-1,:,:] for the last layer only
        #last_layer_list = [torch.tensor(sequence.sum(axis = 0), device = self.device)
        padded_embeddings = torch.stack([sequence.float().mean(0)
                            for sequence in full_embedding_list])
        #padded_embeddings = pad_sequence(last_layer_list, batch_first = True)

        # resort sentences and labels
        labels = torch.tensor(labels, device = self.device)
        labels = labels[perm_index]
        padded_embeddings = padded_embeddings[perm_index]

        return padded_embeddings, labels, lengths


class GloveData(DataGenerator):
    def __init__(self, device, state):
        self.tokenize = lambda text: [token for sent in sent_tokenize(text)
                                            for token in word_tokenize(sent)]
        self.mapper = TokenMapper(num_words = 100000, oov_token = '<unk>')
        self.preprocess = lambda text: self.mapper.texts_to_sequences([self.tokenize(text)])[0]
        self.state = RandomState(state)
        self.device = torch.device(device)
        self.name = 'GloVe'
        self.is_unk = lambda tok_id: tok_id == self.mapper.word_index.get(self.mapper.oov_token)

    @classmethod
    def from_file(cls, glovefile, device, dir = './data/', state = STATE):
        self = cls(device = device, state = state)
        # load dataset
        print('Reading train data...')
        trainfile = os.path.join(dir, 'train.csv')
        self.x_train, self.y_train = self.load_file(trainfile, fit_mapper = True)
        print('Reading dev data...')
        devfile = os.path.join(dir, 'dev.csv')
        self.x_dev, self.y_dev = self.load_file(devfile)
        print('Reading test data...')
        testfile = os.path.join(dir, 'test.csv')
        self.x_test, self.y_test = self.load_file(testfile)

        # create embedding matrix
        self._set_matrix(glovefile)
        return self

    @classmethod
    def from_pickle(cls, binary, device, state = STATE):
        self = cls(device = device, state = state)
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test, self.embedding_matrix, self.mapper = (
            pickle.load(open(binary, 'rb'))
        )
        return self

    def to_pickle(self, binary):
        to_dump = (self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test, self.embedding_matrix, self.mapper)
        pickle.dump(to_dump, open(binary, 'wb'))


    def _set_matrix(self, vecfile):
        # init embedding matrix for words in TokenMapper
        word_index = self.mapper.word_index
        print(len(word_index), list(word_index.keys())[:50])
        self.embedding_matrix = torch.zeros((self.mapper.num_words + 1, 300))

        # fill matrix with corresponding vectors from file
        def get_coefs(word,*arr):
            return word, torch.tensor(np.asarray(arr, dtype='float32'))
        count = 0
        with open(vecfile, 'r', encoding='utf-8') as f:
            for line in f:
                word, vec = get_coefs(*line.split(" "))
                idx = word_index.get(word, self.mapper.num_words + 1)
                if idx < self.mapper.num_words:
                    self.embedding_matrix[idx] = vec
                    count += 1
            self.coverage = (count / self.mapper.num_words)


    def load_file(self, filename, fit_mapper = False):
        data = pd.read_csv(filename)

        # tokenize sents
        print('   Tokenizing...')
        tokenized = [self.tokenize(question) for question in data.question_text]
        if fit_mapper:
            print('   Fitting...')
            self.mapper.fit_on_texts(tokenized)

        # transform to integer ids
        print('   Transforming...')
        questions = self.mapper.texts_to_sequences(tokenized)
        labels = data.target.tolist()
        return questions, labels

    def prepare_batch(self, batch):
        batch, labels = batch

        # assumes tokenized batch
        lengths = torch.tensor([len(sent) for sent in batch])
        lengths, perm_index = lengths.sort(0, descending = True)

        # pad sequences to equal length
        padded_sequences = pad_sequence([torch.tensor(x, device = self.device) for x in batch], batch_first = True)

        # resort labels and texts according to the lengths
        labels = torch.tensor(labels, device = self.device)
        labels = labels[perm_index]
        padded_sequences = padded_sequences[perm_index]

        # get the embeddings
        embeddings = torch.stack([self.embedding_matrix[words_idx] for words_idx in padded_sequences]).to(self.device)
        return embeddings, labels, lengths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', help='directory with train, dev and test data', default='./data/')
    parser.add_argument('--glovefile', help='path to the glove file to load', default='./cache/glove.840B.300d.txt')
    args = parser.parse_args()

    data = BertData.from_file(dir=args.datadir)
    data.to_pickle('./cache/bert.data')

    data = ElmoData.from_file(dir=args.datadir)
    data.to_pickle('./cache/elmo.data')

    data = GloveData.from_file(dir=args.datadir, glovefile=args.glovefile)
    data.to_pickle('./cache/glove.data')
