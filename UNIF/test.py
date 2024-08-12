import os
import random
import sys

import numpy as np

sys.path.append("../../qurey")

from data_load import *
from UNFI_model import UNFI
from copy import deepcopy

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

def comput_mrr(list_t):
    mrr=[]
    for index,i in enumerate(list_t):
        i=i.numpy().tolist()
        target=i[index]
        i.sort()
        mrr.append(1/(i.index(target)+1))
    return np.mean(mrr)

def comput_mrr2(list_t,k=3):
    mrr=[]
    sortnum=[]
    for index,i in enumerate(list_t):
        i=i.numpy().tolist()
        origin = deepcopy(i)
        target=i[index]
        i.sort()
        mrr.append(1/(i.index(target)+1))
        sortnum.append([origin.index(num) for num in i[:k]])

    return np.mean(mrr),sortnum

def comput_topk(list_t,k):
    topk = []
    for index, i in enumerate(list_t):
        i = i.numpy().tolist()
        target = i[index]
        i.sort()
        if i.index(target)<= k-1:
            topk.append(1)
        else:
            topk.append(0)
    return np.mean(topk)

setup_seed(42)

test_db = get_test_dataset()

model=UNFI()

try:
    model.load_weights("../model/UNIF/"+language+"/UNIF.pth")
    print("load model sucess")
except:
    pass

def test_ours(k=3):
    print("======================ours==========================")
    for code, api, fun_name, neg_code_desc, code_desc, qurey, desc in get_test_dataset():
        qurey_repersent, code_repersent = model.qurey([code, api, fun_name, qurey])
        tmp = [tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
        mrr, sortednum = comput_mrr2(tmp,k=k)
    for i in range(2):
        test_db = get_test_dataset()
        test_mrr = []
        test_topk = []
        for code,api,fun_name,neg_code,code_desc,qurey,desc in test_db:
            qurey_repersent,code_repersent=model.qurey2([code,api,fun_name,qurey,np.array(sortednum),code_desc])
            tmp=[tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
            mrr,sortednum=comput_mrr2(tmp)
            test_mrr.append(mrr)
            k_tmp = []
            for k in range(10):
                score_k = comput_topk(tmp, k + 1)
                k_tmp.append(score_k)
            test_topk=k_tmp

        print("test_mrr:", np.mean(test_mrr))
        for k in range(10):
            print("test:@precison", k + 1, ":", test_topk[k])

def test_wordnet():
    test_db=get_wordnet_dataset()
    test_mrr = []
    test_topk = []
    for code,api,fun_name,neg_code_desc,code_desc,qurey,desc,wordnet in test_db:
        qurey_repersent,code_repersent=model.qurey_single([code,api,fun_name,qurey,[wordnet]])
        tmp=[tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
        mrr,sortednum=comput_mrr2(tmp)
        test_mrr.append(mrr)
        k_tmp = []
        for k in range(10):
            score_k = comput_topk(tmp, k + 1)
            k_tmp.append(score_k)
        test_topk=k_tmp

    print("======================wordnet==========================")
    print("test_mrr:", np.mean(test_mrr))
    for k in range(10):
        print("test:@precison", k + 1, ":", test_topk[k])

def test_FP():
    test_db=get_FP_dataset()
    test_mrr = []
    test_topk = []
    for code,api,fun_name,neg_code_desc,code_desc,qurey,desc,FP in test_db:
        qurey_repersent,code_repersent=model.qurey_single([code,api,fun_name,qurey,[FP]])
        tmp=[tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
        mrr,sortednum=comput_mrr2(tmp)
        test_mrr.append(mrr)
        k_tmp = []
        for k in range(10):
            score_k = comput_topk(tmp, k + 1)
            k_tmp.append(score_k)
        test_topk=k_tmp

    print("======================FP==========================")
    print("test_mrr:", np.mean(test_mrr))
    for k in range(10):
        print("test:@precison", k + 1, ":", test_topk[k])

def test_code_desc():
    # =============================================================================================
    test_db = get_test_dataset()
    test_mrr = []
    test_topk = []
    for code, api, fun_name, neg_code_desc, code_desc, qurey, desc in test_db:
        qurey_repersent, code_repersent = model.qurey([code, api, fun_name, code_desc])
        tmp = [tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
        mrr, sortednum = comput_mrr2(tmp)
        test_mrr.append(mrr)
        k_tmp = []
        for k in range(10):
            score_k = comput_topk(tmp, k + 1)
            k_tmp.append(score_k)
        test_topk = k_tmp

    print("======================code_desc==========================")
    print("test_mrr:", np.mean(test_mrr))
    for k in range(10):
        print("test:@precison", k + 1, ":", test_topk[k])

def test_qurey():
    # =============================================================================================
    test_db = get_test_dataset()
    test_mrr = []
    test_topk = []
    for code, api, fun_name, neg_code_desc, code_desc, qurey, desc in test_db:
        qurey_repersent, code_repersent = model.qurey([code, api, fun_name,qurey])
        tmp = [tf.keras.losses.cosine_similarity([i], code_repersent, axis=1) for i in qurey_repersent]
        mrr, sortednum = comput_mrr2(tmp)
        test_mrr.append(mrr)
        k_tmp = []
        for k in range(10):
            score_k = comput_topk(tmp, k + 1)
            k_tmp.append(score_k)
        test_topk = k_tmp
    print("======================qurey==========================")
    print("test_mrr:", np.mean(test_mrr))
    for k in range(10):
        print("test:@precison", k + 1, ":", test_topk[k])

for i in range(1,10):
    test_ours(i)