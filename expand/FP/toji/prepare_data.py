import numpy as np

with open("../language.txt", "r+") as f:
    language = f.read()

train = "../../data/" + language + "/train/"
valid = "../../data/" + language + "/valid/"
test = "../../data/" + language + "/test/"

class loader():
    def __init__(self):
        self.train_qurey = np.load(train+"qurey.npy")
        self.train_code = np.load(train + "desc.npy")
        self.train_code_desc = np.load(train+"code_desc.npy")
        self.valid_qurey = np.load(valid + "qurey.npy")
        self.valid_code = np.load(valid + "desc.npy")
        self.valid_code_desc = np.load(valid + "code_desc.npy")
        self.test_qurey = np.load(test + "qurey.npy")
        self.test_code = np.load(test + "desc.npy")
        self.test_code_desc = np.load(test + "code_desc.npy")
        self.desc=np.concatenate([self.train_code_desc,self.valid_code_desc,self.test_code_desc],axis=0)
    def get_data(self):
        data=[]
        for i in self.desc:
            a=np.logical_not(np.equal(i,0))
            data.append((i[a].tolist()))
        return data