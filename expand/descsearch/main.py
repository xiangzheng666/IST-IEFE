import os
import random
import sys
sys.path.append("../../../qurey")

from tqdm import tqdm
from data_load import *
from UNFI_model import UNFI
from bert import bertmodel

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first

setup_seed(42)

valid_db = get_valid_dataset()
test_db = get_test_dataset()

model=bertmodel()
#optimizer=tf.keras.optimizers.Adam(lr)
optimizer=tf.keras.optimizers.Adamax()

try:
    model.load_weights("../../model/desc_search/"+language+"/bertmodel.pth")
    print("load model sucess")
except:
    pass
def simlity_loss(neg_repersent,code_repersent,code_desc_repersent):
    neg_score=tf.keras.losses.cosine_similarity(neg_repersent,code_desc_repersent ,axis=1)  # [batchsize]
    good_score=tf.keras.losses.cosine_similarity(code_repersent,code_desc_repersent ,axis=1)
    return  tf.reduce_mean(tf.clip_by_value(tf.add(good_score-neg_score,0.5),0,10))

def comput_mrr(list_t):
    mrr=[]
    for index,i in enumerate(list_t):
        i=i.numpy().tolist()
        target=i[index]
        i.sort()
        mrr.append(1/(i.index(target)+1))
    return np.mean(mrr)

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

score=0
for epoch in range(epoch_num):
    num, train_db = get_train_dataset()
    loss_score = []
    with tqdm(total=int(num+1)) as pbar:
        for code,api,fun_name,neg_code,neg_api,neg_fun_name,code_desc,qurey,desc in train_db:
            pbar.update(1)
            with tf.GradientTape() as tape:
                neg_repersent,code_repersent,code_desc_repersent=model([qurey,desc,neg_code])
                epoch_loss=simlity_loss(neg_repersent, code_repersent, code_desc_repersent)
            grad = tape.gradient(epoch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grad, model.trainable_variables))
            loss_score.append(epoch_loss.numpy())

    valid_mrr=[]
    valid_topk=[]
    for code,api,fun_name,neg_code,neg_api,neg_fun_name,code_desc,qurey,desc in valid_db:
        qurey_repersent,code_repersent=model.qurey([qurey,desc])
        tmp=[tf.keras.losses.cosine_similarity([i], tf.cast(code_repersent,dtype=tf.float32), axis=1) for i in tf.cast(qurey_repersent,dtype=tf.float32)]
        valid_mrr.append(comput_mrr(tmp))
        k_tmp=[]
        for k in range(10):
            score_k=comput_topk(tmp,k+1)
            k_tmp.append(score_k)
        valid_topk=k_tmp

    print("epoch:", epoch, "loss:", np.mean(loss_score), "valid_mrr:", np.mean(valid_mrr))
    for k in range(10):
        print("valid:@precison",k+1,":",valid_topk[k])

    test_mrr = []
    test_topk = []
    for code,api,fun_name,neg_code,neg_api,neg_fun_name,code_desc,qurey,desc in test_db:
        qurey_repersent,code_repersent=model.qurey([qurey,desc])
        tmp=[tf.keras.losses.cosine_similarity([i], tf.cast(code_repersent,dtype=tf.float32), axis=1) for i in tf.cast(qurey_repersent,dtype=tf.float32)]
        test_mrr.append(comput_mrr(tmp))
        k_tmp = []
        for k in range(10):
            score_k = comput_topk(tmp, k + 1)
            k_tmp.append(score_k)
        test_topk=k_tmp

    print("test_mrr:", np.mean(test_mrr))
    for k in range(10):
        print("test:@precison", k + 1, ":", test_topk[k])

    if np.mean(test_mrr) > score:
        score=np.mean(test_mrr)
        model.save_weights("../../model/desc_search/"+language+"/bertmodel.pth",overwrite=True)

