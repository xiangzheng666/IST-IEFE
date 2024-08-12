import numpy as np
import tensorflow as tf

with open("language.txt","r+") as f:
    language = f.read()

if language=="python":
    from data.python.paramters import *
else:
    from data.java.paramters import *

train="../../data/"+language+"/train/"
valid="../../data/"+language+"/valid/"
test="../../data/"+language+"/test/"

class loader():
    def __init__(self,path):
        self.qurey = np.load(path+"qurey.npy")
        self.code = np.load(path + "code.npy")
        self.code_desc = np.load(path+"code_desc.npy")
        self.len=self.qurey.shape[0]
        self.num = self.len / 1000
        self.mark = [-1]
    def __getitem__(self, item):
        if item < self.len:
            return tf.convert_to_tensor(self.qurey[item,:]),\
                   tf.convert_to_tensor(self.code[item,:]),\
                   tf.convert_to_tensor(self.code_desc[item,:])
        else:
            return self.mark[4]


def get_test_dataset():
    a = loader(test)
    return a.num,tf.data.Dataset.from_generator( lambda : a,(tf.int64,tf.int64,tf.int64)).batch(1000,drop_remainder=False)