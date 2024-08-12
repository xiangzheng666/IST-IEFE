import tensorflow as tf
import numpy as np

with open("language.txt","r+") as f:
    language = f.read()

if language=="python":
    from data.python.paramters import *
else:
    from data.java.paramters import *

train="../data/"+language+"/train/"
valid="../data/"+language+"/valid/"
test="../data/"+language+"/test/"

class loader():
    def __init__(self, path):
        self.qurey = np.load(path + "qurey.npy")
        self.desc = np.load(path + "desc.npy")

        self.code = np.load(path + "code.npy")
        self.api = np.load(path + "api.npy")
        self.func = np.load(path + "fun_name.npy")

        self.code_desc = np.load(path + "code_desc.npy")

        self.len = self.qurey.shape[0]
        self.num=self.len/bs
        self.mark = [-1]

        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def rest_index(self):
        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def change_qurey(self,path):
        self.qurey = np.load(path)
    def reas_qurey(self):
        self.qurey=tf.keras.preprocessing.sequence.pad_sequences(self.qurey, maxlen=100, padding='post',
                                                                   truncating='post', value=0)
    def get_qurey_func(self):

        out = []
        for i, j in zip(self.qurey, self.func):
            out.append(np.concatenate((i[:np.sum(i != 0)], j[:np.sum(j != 0)])).tolist())

        self.qurey = tf.keras.preprocessing.sequence.pad_sequences(out, maxlen=100, padding='post',
                                                                   truncating='post', value=0)

    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.code[item, :]), \
                   tf.convert_to_tensor(self.api[item, :]),\
                   tf.convert_to_tensor(self.func[item, :]), \
                   tf.convert_to_tensor(self.qurey[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.code_desc[item, :]), \
                   tf.convert_to_tensor(self.qurey[item, :]),\
                   tf.convert_to_tensor(self.desc[item, :])
        else:
            return self.mark[4]

def get_train_dataset():
    a=loader(train)
    a.rest_index()
    return a.num,tf.data.Dataset.from_generator( lambda :a,(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(bs)

def get_valid_dataset():
    return tf.data.Dataset.from_generator( lambda : loader(valid),(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

def get_test_dataset(qurey=False):
    a = loader(test)
    if qurey:
        a.reas_qurey()
    return tf.data.Dataset.from_generator( lambda : a,(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

class wordnet():
    def __init__(self, path):
        self.qurey = np.load(path + "qurey.npy")
        self.desc = np.load(path + "desc.npy")

        self.code = np.load(path + "code.npy")
        self.api = np.load(path + "api.npy")
        self.func = np.load(path + "fun_name.npy")

        self.code_desc = np.load(path + "code_desc.npy")
        self.wordnet = np.load("../expand/wordnet/out/"+language+"_qurey_wordnet.npy")
        self.len = self.qurey.shape[0]
        self.num=self.len/bs
        self.mark = [-1]

        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def rest_index(self):
        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)

    def get_qurey_func(self):

        out = []
        for i, j in zip(self.qurey, self.func):
            out.append(np.concatenate((i[:np.sum(i != 0)], j[:np.sum(j != 0)])).tolist())

        self.qurey = tf.keras.preprocessing.sequence.pad_sequences(out, maxlen=100, padding='post',
                                                                   truncating='post', value=0)
    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.code[item, :]), \
                   tf.convert_to_tensor(self.api[item, :]),\
                   tf.convert_to_tensor(self.func[item, :]), \
                   tf.convert_to_tensor(self.code_desc[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.code_desc[item, :]), \
                   tf.convert_to_tensor(self.qurey[item, :]),\
                   tf.convert_to_tensor(self.desc[item, :]), \
                   tf.convert_to_tensor(self.wordnet[item, :])
        else:
            return self.mark[4]
def get_wordnet_dataset(poolsize=poolsize):

    return tf.data.Dataset.from_generator( lambda : wordnet(test),(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

class FP():
    def __init__(self, path):
        self.qurey = np.load(path + "qurey.npy")
        self.desc = np.load(path + "desc.npy")

        self.code = np.load(path + "code.npy")
        self.api = np.load(path + "api.npy")
        self.func = np.load(path + "fun_name.npy")

        self.code_desc = np.load(path + "code_desc.npy")
        self.wordnet = np.load("../expand/FP/out/"+language+"_qurey_fp.npy")
        self.len = self.qurey.shape[0]
        self.num=self.len/bs
        self.mark = [-1]

        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def rest_index(self):
        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)

    def get_qurey_func(self):

        out = []
        for i, j in zip(self.qurey, self.func):
            out.append(np.concatenate((i[:np.sum(i != 0)], j[:np.sum(j != 0)])).tolist())

        self.qurey = tf.keras.preprocessing.sequence.pad_sequences(out, maxlen=100, padding='post',
                                                                   truncating='post', value=0)
    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.code[item, :]), \
                   tf.convert_to_tensor(self.api[item, :]),\
                   tf.convert_to_tensor(self.func[item, :]), \
                   tf.convert_to_tensor(self.code_desc[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.code_desc[item, :]), \
                   tf.convert_to_tensor(self.qurey[item, :]),\
                   tf.convert_to_tensor(self.desc[item, :]), \
                   tf.convert_to_tensor(self.wordnet[item, :])
        else:
            return self.mark[4]
def get_FP_dataset(poolsize=poolsize):
    return tf.data.Dataset.from_generator( lambda : FP(test),(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)

class wordnet_FP():
    def __init__(self, path):
        self.qurey = np.load(path + "qurey.npy")
        self.desc = np.load(path + "desc.npy")

        self.code = np.load(path + "code.npy")
        self.api = np.load(path + "api.npy")
        self.func = np.load(path + "fun_name.npy")

        self.code_desc = np.load(path + "code_desc.npy")
        self.fp = np.load("../expand/FP/out/"+language+"_qurey_fp.npy")
        self.wordnet = np.load("../expand/wordnet/out/" + language + "_qurey_wordnet.npy")
        self.len = self.qurey.shape[0]
        self.num=self.len/bs
        self.mark = [-1]

        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)
    def rest_index(self):
        self.neg_index = np.random.randint(low=1, high=self.len - 1, size=self.len)

    def get_qurey_func(self):

        out = []
        for i, j in zip(self.qurey, self.func):
            out.append(np.concatenate((i[:np.sum(i != 0)], j[:np.sum(j != 0)])).tolist())

        self.qurey = tf.keras.preprocessing.sequence.pad_sequences(out, maxlen=100, padding='post',
                                                                   truncating='post', value=0)
    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.code[item, :]), \
                   tf.convert_to_tensor(self.api[item, :]),\
                   tf.convert_to_tensor(self.func[item, :]), \
                   tf.convert_to_tensor(self.code_desc[(item+self.neg_index[item])%self.len, :]), \
                   tf.convert_to_tensor(self.code_desc[item, :]), \
                   tf.convert_to_tensor(self.qurey[item, :]),\
                   tf.convert_to_tensor(self.desc[item, :]), \
                   tf.convert_to_tensor(self.wordnet[item, :]), \
                   tf.convert_to_tensor(self.fp[item, :])
        else:
            return self.mark[4]
def get_wordnet_FP_dataset(poolsize=poolsize):
    return tf.data.Dataset.from_generator( lambda : wordnet_FP(test),(tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64,tf.int64)).batch(poolsize)
