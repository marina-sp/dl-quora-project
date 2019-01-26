import pandas as pd
import numpy as np

from pytorch_pretrained_bert import BertTokenizer, BertModel
from sacremoses import MosesTokenizer
from nltk.tokenize import word_tokenize, sent_tokenize
from token_mapper import TokenMapper
from allennlp.commands.elmo import ElmoEmbedder

from torch.nn.utils.rnn import pad_sequence
import torch

import re, os, sys, pickle
#import dill as pickle
import argparse, tqdm
from numpy.random import RandomState


# Abstract class implementing a batch generator
# with train, dev and test data.
class DataGenerator():

    @classmethod
    def from_file(cls, dir = './data/', device = 'cuda', cache_dir = './cache/', state = 333):
        ## assumes a child class specific method: __init__
        self = cls(device = device, cache_dir = cache_dir)
        self.state = RandomState(state)
        ## assumes a child class specific method: load_file
        trainfile = os.path.join(dir, 'train.csv')
        self.x_train, self.y_train = self.load_file(trainfile)
        devfile = os.path.join(dir, 'dev.csv')
        self.x_dev, self.y_dev = self.load_file(devfile)
        testfile = os.path.join(dir, 'test.csv')
        self.x_test, self.y_test = self.load_file(testfile)
        return self

    @classmethod
    def from_pickle(cls, binary, device = 'cuda', cache_dir = './cache/', state = 333):
        ## assumes a child class specific method: __init__
        self = cls(device=device, cache_dir=cache_dir)
        self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test = (
            pickle.load(open(binary, 'rb'))
        )
        return self

    def to_pickle(self, binary):
        to_dump = (self.x_train, self.y_train, self.x_dev, self.y_dev, self.x_test, self.y_test)
        pickle.dump(to_dump, open(binary, 'wb'))

    def _get_batches(self, X, Y, size):
        ## assumes a child class specific method: prepare_batch

        # shuffle data
        X, Y = np.array(X), np.array(Y)
        shuffle_idx = np.arange(len(Y))
        self.state.shuffle(shuffle_idx)
        X, Y = X[shuffle_idx], Y[shuffle_idx]

        # split in batches
        for i in range(0, len(Y), size):
            x, y = X[i:i+size], Y[i:i+size]
            x, y, lengths = self.prepare_batch((x,y))
            yield x, y, lengths

    def get_train_batches(self, size):
        for output in self._get_batches(self.x_train, self.y_train, size):
            yield output

    def get_dev_batches(self, size):
        for output in self._get_batches(self.x_dev, self.y_dev, size):
            yield output

    def get_test_batches(self, size):
        for output in self._get_batches(self.x_test, self.y_test, size):
            yield output

    def prepare_sentence(self, text):
        ## assumes a child class specific method: prepare_batch
        tok_batch = [self.preprocess(text)]
        batch, _, _ = self.prepare_batch((tok_batch, [0]))
        return batch

    def get_stats(self):
        return


class BertData(DataGenerator):

    def __init__(self, device = 'cuda', cache_dir = './cache/'):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir = cache_dir)
        self.preprocess = lambda sent: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sent.lower()))
        self.bert = BertModel.from_pretrained('bert-base-uncased', cache_dir = cache_dir).to(device)
        self.device = torch.device(device)

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

    def __init__(self, device = 'cuda', cache_dir = './cache/'):
        # tokenize sents
        self.tokenizer = MosesTokenizer()
        self.preprocess = lambda sent: self.tokenizer.tokenize(sent.lower(), escape=False)
        self.elmo = ElmoEmbedder(options_file = os.path.join(cache_dir, 'elmo_2x4096_512_2048cnn_2xhighway_options.json'),
                                 weight_file = os.path.join(cache_dir, 'elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'),
                                 cuda_device = 0 if device == 'cuda' else -1)
        self.device = torch.device(device)

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
        full_embedding_list = self.elmo.embed_batch(batch)

        # average over embedding layers for every token
        # sequence[-1,:,:] for the last layer only
        last_layer_list = [torch.tensor(sequence.sum(axis = 0), device = self.device)
                           for sequence in full_embedding_list]
        padded_embeddings = pad_sequence(last_layer_list, batch_first = True)

        # resort sentences and labels
        labels = torch.tensor(labels, device = self.device)
        labels = labels[perm_index]
        padded_embeddings = padded_embeddings[perm_index]

        return padded_embeddings, labels, lengths


## TODO: embedding matrix and DataGenerator format
class GloveData(DataGenerator):
    def __init__(self, device, state):
        self.tokenize = lambda text: [token for sent in sent_tokenize(text)
                                            for token in word_tokenize(sent)]
        self.mapper = TokenMapper(num_words = 100000, oov_token = '<unk>')
        self.preprocess = lambda text: self.mapper.texts_to_sequences([self.tokenize(text)])
        self.state = RandomState(state)
        self.device = torch.device(device)

    @classmethod
    def from_file(cls, glovefile, device = 'cuda', dir = './data/', state = 333):
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
    def from_pickle(cls, binary, device = 'cuda', state = 333):
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

    #data = BertData(args.datadir)
    #pickle.dump(data, open('cache/data.bert.test','wb'))

    #data = ElmoData()
    #pickle.dump(data, open('cache/data.elmo.test','wb'))
    #data = BertData.from_file(dir = args.datadir)
    #data.to_pickle('./cache/data.bert')
    #loaded = BertData.from_pickle('./cache/data.bert')
    #assert (data.x_train == loaded.x_train)
    #assert (data.y_train == loaded.y_train)

    #data = ElmoData.from_file(dir = args.datadir)
    #data.to_pickle('./cache/data.elmo')
    #loaded = ElmoData.from_pickle('./cache/data.elmo')
    #assert (data.x_train == loaded.x_train)
    #assert (data.y_train == loaded.y_train)

    data = GloveData.from_file(args.glovefile)
    data.to_pickle('./cache/data.glove')
    print(data.coverage)
    loaded = GloveData.from_pickle('./cache/data.glove')
    assert (data.x_train == loaded.x_train)
    assert (data.y_train == loaded.y_train)