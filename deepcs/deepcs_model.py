import numpy as np
from tensorflow import keras
from tensorflow_core.python.keras.layers import Lambda
from tensorflow.keras import backend as K
import tensorflow as tf
with open("language.txt","r+") as f:
    language = f.read()

if language=="python":
    from data.python.paramters import *
else:
    from data.java.paramters import *

class deepcs(keras.Model):
    def __init__(self):
        super(deepcs, self).__init__()

        self.maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]),
                              name='maxpool_methname')
        #---------------api_repersent-----------------------
        self.api_embeding = keras.layers.Embedding(vocab_api, api_emceding_len)
        self.api_lstm_1 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)
        self.api_lstm_2 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)

        # ---------------funcname_repersent-----------------------
        self.funcname_embeding = keras.layers.Embedding(vocab_fun_name, fun_name_emceding_len)
        self.funcname_lstm_1 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)
        self.funcname_lstm_2 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)

        # ---------------code_repersent-----------------------
        self.code_embeding = keras.layers.Embedding(vocab_code, code_emceding_len)
        self.code_lstm_1 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)
        self.code_lstm_2 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)

        # ---------------code_desc_repersent-----------------------
        self.code_desc_embeding = keras.layers.Embedding(vocab_code_desc, code_desc_emceding_len)
        self.code_desc_lstm_1 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)
        self.code_desc_lstm_2 = keras.layers.LSTM(lstm_uint, return_sequences=True, recurrent_dropout=0.2)

        #---------------CONCATE--------------
        self.dense=keras.layers.Dense(2*lstm_uint)

    def api_repersent(self, api):
        api=self.api_embeding(api)
        l1 = self.maxpool(self.api_lstm_1(api))
        l2 = self.maxpool(self.api_lstm_2(api))
        api=keras.layers.concatenate([l1,l2],axis=1)
        return keras.layers.Activation('tanh')(api)

    def funcname_repersent(self, funcname):
        funcname=self.funcname_embeding(funcname)
        l1 = self.maxpool(self.funcname_lstm_1(funcname))
        l2 = self.maxpool(self.funcname_lstm_2(funcname))
        funcname=keras.layers.concatenate([l1,l2],axis=1)
        return keras.layers.Activation('tanh')(funcname)

    def code_repersent(self, code):
        code=self.code_embeding(code)
        l1 = self.maxpool(self.code_lstm_1(code))
        l2 = self.maxpool(self.code_lstm_2(code))
        code=keras.layers.concatenate([l1,l2],axis=1)
        return keras.layers.Activation('tanh')(code)

    def code_desc_repersent(self, code_desc):
        code_desc=self.code_desc_embeding(code_desc)
        l1 = self.maxpool(self.code_desc_lstm_1(code_desc))
        l2 = self.maxpool(self.code_desc_lstm_2(code_desc))
        code_desc=keras.layers.concatenate([l1,l2],axis=1)
        return keras.layers.Activation('tanh')(code_desc)

    def concate(self,code,api,fun_name):
        return self.dense(keras.layers.concatenate([code,api,fun_name], axis=1))

    def call(self, inputs, training=None, mask=None):
        code,api,fun_name,neg_qurey,code_desc,qurey,desc = inputs

        neg_code_desc_repersent=self.code_desc_repersent(qurey)

        code = self.code_repersent(code)
        api=self.api_repersent(api)
        fun_name = self.funcname_repersent(fun_name)
        code_repersent=self.concate(code,api,fun_name)

        code_desc_repersent = self.code_desc_repersent(qurey)

        return neg_code_desc_repersent,code_repersent,code_desc_repersent

    def qurey(self,inputs):
        code, api, fun_name,qurey=inputs

        code = self.code_repersent(code)
        api = self.api_repersent(api)
        fun_name = self.funcname_repersent(fun_name)

        code_repersent = self.concate(code, api, fun_name)
        qurey_repersent = self.code_desc_repersent(qurey)
        return qurey_repersent,code_repersent

    def qurey_single(self, inputs):
        code, api, fun_name, qurey, lists = inputs
        code = self.code_repersent(code)
        api = self.api_repersent(api)
        fun_name = self.funcname_repersent(fun_name)

        code_vect = self.concate(code, api, fun_name)

        desc_vect = self.code_desc_repersent(qurey)

        for i in lists:
            desc_vect = desc_vect + self.code_desc_repersent(i) * 0.2

        return code_vect, desc_vect

    def qurey2(self, inputs):
        code, api, fun_name, qurey, sortednum, code_desc = inputs
        code = self.code_repersent(code)
        api = self.api_repersent(api)
        fun_name = self.funcname_repersent(fun_name)

        code_vect = self.concate(code, api, fun_name)
        desc_vect = self.code_desc_repersent(qurey)

        desc = []
        for i in range(sortednum.shape[1]):
            desc.append(
                self.code_desc_repersent(tf.concat([np.expand_dims(qurey[j], axis=0) for j in sortednum[:, i]], axis=0)))

        for i in range(len(desc)):
            desc_vect = desc_vect + desc[i]

        return code_vect, desc_vect




