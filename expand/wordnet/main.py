from wordnet_model import util
import sys
sys.path.append("../../../qurey")

from data_load import *
with open("language.txt","r+") as f:
    language = f.read()

with open("../../data/"+language+"/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
    qurey=f.read().split("\n")
    qurey_dict={i:qurey[i] for i in range(len(qurey))}

with open("../../data/"+language+"/vocab//vocab_code.txt", "r+", encoding="utf-8") as f:
    code=f.read().split("\n")
    code_dict={i:code[i] for i in range(len(code))}

qurey_dict_back={qurey_dict[i]:i for i in qurey_dict.keys()}
_,db=get_test_dataset()
for qurey,code ,code_desc in db:
    orignal_qureys=[" ".join([qurey_dict[word] for word in sentensce if word!=0]) for sentensce in qurey.numpy().tolist()]
    data = [" ".join(" ".join(j.split("_")) for j in i.split(" ")) for i in [util(j) for j in orignal_qureys]]
    num = [[qurey_dict_back[j] for j in i.split(" ") if j in qurey_dict_back.keys()] for i in data]
    with open("out/"+language+"_wordnet.txt", 'w+') as f:
        f.write("\n".join(data))
    with open("out/"+language+"_test.txt", 'w+') as f:
        f.write("\n".join(orignal_qureys))

qurey_wordnet=tf.keras.preprocessing.sequence.pad_sequences(num,maxlen=desc_len, padding='post',
                                              truncating='post', value=0)
np.save("out/"+language+"_qurey_wordnet",qurey_wordnet)
