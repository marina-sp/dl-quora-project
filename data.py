import pandas as pd

from pytorch_pretrained_bert import BertTokenizer

from torch.utils.data.dataset import *
import torch
import tensorflow as tf


class BertData(Dataset):

    def __init__(self, trainfile = 'data\\train.csv'):
        data = pd.read_csv(trainfile)
        # tokenize sents
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.questions = [bert_tokenizer.convert_tokens_to_ids(bert_tokenizer.tokenize(sent))
                          for sent in data.question_text]
        self.labels = data.target.tolist()

    def __getitem__(self, i):
        return self.questions[i], self.labels[i]

    def __len__(self):
        return len(self.labels)

class ElmoData(Dataset):

    def __init__(self, trainfile = 'data\\train.csv'):
        data = pd.read_csv(trainfile)
        # tokenize sents
        self.questions = [sent.split()
                          for sent in data.question_text]
        self.labels = data.target.tolist()

    def __getitem__(self, i):
        return self.questions[i], self.labels[i]

    def __len__(self):
        return len(self.labels)


def prepare_bert_batch(bert, batch):
    # assumes tokenized batch
    lengths = sorted([len(sent) for sent in batch], reverse=True)

    seq_len = lengths[0]
    padded = pad_sequence([torch.tensor(x, device=device) for x in batch], batch_first=True)

    # get the input embeddings
    emb, _ = bert(padded,
                  attention_mask = torch.tensor([[1]*cur_len + [0]*(seq_len - cur_len)
                                                 for cur_len in lengths], device=device),
                  output_all_encoded_layers = False)
    return emb, lengths

def prepare_elmo_batch(elmo, batch):
    # assumes whitespace split sentences
    tokens_length = [len(sent) for sent in batch]
    seq_len = max(tokens_length)
    batch_size = len(batch)

    # get the input embeddings
    tokens_input = [sent + ['']*(seq_len - len(sent))
                    for sent in batch]
    #print(tokens_input, tokens_length)

    with tf.Session() as sess:
        tokens_ph = tf.placeholder(shape=(None, None), dtype=tf.string, name='tokens')
        len_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='lengths')
        embeddings = elmo(
                inputs={
                    "tokens": tokens_ph,
                    "sequence_len": len_ph
                        },
                signature="tokens",
                as_dict=True)["elmo"]

        sess.run(tf.tables_initializer())
        sess.run(tf.global_variables_initializer())

        emb = sess.run(embeddings,
                       feed_dict={
                            tokens_ph: tokens_input,
                            len_ph: tokens_length
                        }
                      )
        return emb, tokens_length


if __name__ == '__main__':

    data = BertData()
    pickle.dump(data, open('data/data.bert.tokens','wb'))

    data = ElmoData()
    pickle.dump(data, open('data/data.elmo.tokens','wb'))
