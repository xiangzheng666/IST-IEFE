import pyfpgrowth

from prepare_data import *
from gensim import corpora

import numpy as np
import nltk

data=loader().get_data()


with open("../language.txt", "r+") as f:
    language = f.read()

with open("../../../data/"+language+"/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
    qurey=f.read().split("\n")
    qurey_dict={i:qurey[i] for i in range(len(qurey))}
    qurey_dict_back={qurey_dict[i]:i for i in qurey_dict.keys()}


data=[[qurey_dict[int(z)] for z in j ] for j in data]

def get_vocab_2_least(text):
    dicts = corpora.Dictionary(text)
    once_ids = [tokenid for tokenid, docfreq in dicts.dfs.items() if docfreq <= 100]
    dicts.filter_tokens(once_ids)
    return dicts.token2id

def isnn(word):
    if nltk.pos_tag([word])[0][1] in ['NN','NNP','NNS','NNPS']:
        return True
    else:
        return False

def get_maxk(l,tmp):
    frq=[tmp[i] for i in l]
    index=np.argsort(frq)[-10:]
    return [l[i] for i in index]

vocab=get_vocab_2_least(data)
word100=[[j for j in i if j in vocab.keys()] for i in data]
print("pares nn")
nn=[[j for j in i if isnn(j)]for i in word100]
print('finish')
set_nn=[list(set(i)) for i in nn]
nn_set_num=[[vocab[j] for j in i] for i in set_nn]
d=corpora.Dictionary(data)

tmp={}
for tokenid, docfreq in  d.dfs.items():
    tmp[tokenid]=docfreq

out=[get_maxk(i,tmp) for i in nn_set_num]
v={d.token2id[i]:i for i in d.token2id.keys()}
out=[[v[j] for j in i] for i in out]
patterns = pyfpgrowth.find_frequent_patterns(out, 2)
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)

with open(language+"_pattern.txt", "w+") as f:
    for k,v in patterns.items():
        f.write("_".join(list(k))+' '+str(v)+"\n")
with open(language+"_rules.txt", "w+") as f:
    for k,v in rules.items():
        f.write("_".join(list(k))+"+"+"_".join(list(v[0]))+"\n")