import numpy as np
from tensorflow import keras
import tensorflow as tf
import math

with open("language.txt", "r+") as f:
    language = f.read()

if language=="python":
    from data.python.paramters import *
else:
    from data.java.paramters import *

class AttCodeEncoder(keras.Model):

    def __init__(self):
        super(AttCodeEncoder, self).__init__()
        self.emb_size = code_emceding_len
        self.hidden_size = lstm_uint
        self.embedding = keras.layers.Embedding(vocab_code_desc, code_emceding_len)
        # self.word_weights = get_word_weights(vocab_size)
        self.attention = keras.layers.Dense(1)
        self.drop=keras.layers.Dropout(0.25)

    def call(self, inputs, training=None, mask=None):
        embedded = self.embedding(inputs)  # input: [batch_sz x seq_len x 1]  embedded: [batch_sz x seq_len x emb_sz]
        embedded = self.drop(embedded)  # [batch_size x seq_len x emb_size]
        inital_value = tf.math.exp(tf.squeeze(self.attention(embedded),axis=2))
        attention_weight = tf.divide(inital_value, tf.reduce_sum(inital_value, 1, True))
        attention_weight = tf.expand_dims(attention_weight,-1)
        output = tf.squeeze(tf.matmul(tf.transpose(embedded,[0,2, 1]), attention_weight),axis=2)
        return output


class SeqEncoder(keras.Model):
    def __init__(self):
        super(SeqEncoder, self).__init__()
        self.emb_size = desc_emceding_len
        self.hidden_size = lstm_uint
        self.n_layers = 1
        self.embedding = keras.layers.Embedding(vocab_code_desc, code_emceding_len)
        self.drop = keras.layers.Dropout(0.25)

    def call(self, inputs, training=None, mask=None):
        batch_size, seq_len = inputs.shape
        inputs = self.embedding(inputs)  # input: [batch_sz x seq_len]  embedded: [batch_sz x seq_len x emb_sz]
        inputs = self.drop(inputs)
        encoding = tf.reduce_sum(inputs, 1, True) / seq_len
        encoding = tf.squeeze(encoding,axis=1)

        return encoding


def get_word_weights(vocab_size, padding_idx=0):
    '''contruct a word weighting table '''

    def cal_weight(word_idx):
        return 1 - math.exp(-word_idx)

    weight_table = np.array([cal_weight(w) for w in range(vocab_size)])
    if padding_idx is not None:
        weight_table[padding_idx] = 0.  # zero vector for padding dimension
    return tf.convert_to_tensor(weight_table,dtype=tf.float32)

class UNFI(keras.Model):
    def __init__(self):
        super(UNFI, self).__init__()

        self.code_present=AttCodeEncoder()
        self.desc_present=AttCodeEncoder()


    def call(self, inputs, training=None, mask=None):
        code,api,fun_name,neg_code,neg_api,neg_fun_name,code_desc,qurey,desc = inputs

        neg_desc_vect = self.code_present(neg_code)
        desc_vect=self.code_present(code_desc)
        qurey_vect=self.desc_present(qurey)

        return neg_desc_vect,desc_vect,qurey_vect

    def qurey(self,inputs):
        qurey,desc=inputs
        desc_vect = self.code_present(desc)
        qurey_vect = self.desc_present(qurey)

        return qurey_vect, desc_vect

    def get_topk_desc(self,inputs):
        qurey, desc=inputs
        desc_vect = self.code_present(desc)
        qurey_vect = self.desc_present(qurey)

        tmp=[np.argsort(tf.keras.losses.cosine_similarity([i], desc_vect, axis=1))[:10] for i in qurey_vect]
        return tmp