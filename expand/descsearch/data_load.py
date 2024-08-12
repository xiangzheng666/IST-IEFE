import tensorflow as tf
import numpy as np

with open("language.txt", "r+") as f:
    language = f.read()

if language=="python":
    from data.python.paramters import *
else:
    from data.java.paramters import *

train="../../data/"+language+"/train/"
valid="../../data/"+language+"/valid/"
test="../../data/"+language+"/test/"


with open("../../data/"+language+"/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
    qurey=f.read().split("\n")
    qurey_dict={i:qurey[i] for i in range(len(qurey))}
qurey_dict_back={qurey_dict[i]:i for i in qurey_dict.keys()}

class loader():
    def __init__(self, path):
        self.qurey = np.load(path + "qurey.npy")
        self.desc = np.load(path + "desc.npy")

        self.code = np.load(path + "code.npy")
        self.api = np.load(path + "api.npy")
        self.func = np.load(path + "fun_name.npy")

        self.code_desc = np.load(path + "code_desc.npy")

        # self.code_desc=[[qurey_dict[j] for j in i if j !=0] for i in self.code_desc]
        # self.code_desc = [[qurey_dict_back[j] for j in i if j not in stopwords] for i in self.code_desc]
        #
        # self.code_desc=tf.keras.preprocessing.sequence.pad_sequences(self.code_desc, maxlen=desc_len, padding='post',
        #                                               truncating='post', value=0)

        self.len = self.qurey.shape[0]
        self.num=self.len/bs
        self.mark = [-1]

        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)

    def rest_index(self):
        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.code[item, :]), \
                   tf.convert_to_tensor(self.api[item, :]),\
                   tf.convert_to_tensor(self.func[item, :]), \
                   tf.convert_to_tensor(self.desc[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.api[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.func[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.code_desc[item, :]), \
                   tf.convert_to_tensor(self.qurey[item, :]),\
                   tf.convert_to_tensor(self.desc[item, :])
        else:
            return self.mark[4]

def get_train_dataset():
    a=loader(train)
    a.rest_index()
    return a.num,tf.data.Dataset.from_generator( lambda :a,(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(bs)

def get_valid_dataset():
    return tf.data.Dataset.from_generator( lambda : loader(valid),(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

def get_test_dataset(poolsize=poolsize):
    a=loader(test)
    return tf.data.Dataset.from_generator( lambda : a,(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

def get_desc_search():
    a=loader(test)
    return tf.data.Dataset.from_generator(lambda: a, (
    tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64, tf.int64)).batch(poolsize)
