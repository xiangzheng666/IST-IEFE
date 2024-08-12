import nltk
import sys
sys.path.append("../../../qurey")

with open("language.txt", "r+") as f:
    language = f.read()

with open("toji/"+language+"_pattern.txt", "r+") as f:
    pattern=f.read().split("\n")
with open("toji/"+language+"_rules.txt", "r+") as f:
    rules=f.read().split("\n")

pattern=[[i.split(" ")[0].split("_"),i.split(" ")[1]] for i in pattern if len(i.split(" ")[0].split("_"))>1]
pattern=sorted(pattern,key=lambda x:int(x[1]),reverse=True)
pattern=[i[0] for i in pattern]


with open("../../data/"+language+"/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
    qurey=f.read().split("\n")
    qurey_dict={i:qurey[i] for i in range(len(qurey))}
    qurey_dict_back={qurey_dict[i]:i for i in qurey_dict.keys()}

with open("../../data/"+language+"/vocab/vocab_code.txt", "r+", encoding="utf-8") as f:
    code=f.read().split("\n")
    code_dict={i:code[i] for i in range(len(code))}

def isnn(word):
    if nltk.pos_tag([word])[0][1] in ['NN','NNP','NNS','NNPS']:
        return True
    else:
        return False

from data_load import *

_,db=get_test_dataset()
for qurey,code ,code_desc in db:
    qureys_old=[[qurey_dict[word] for word in sentensce if word!=0] for sentensce in qurey.numpy().tolist()]
    qureys=[[w for w in sentensce if isnn(w)] for sentensce in qureys_old]
    out=[]
    txt=[]
    for old,sentencs in zip(qureys_old,qureys):
        tmp=[]
        expand = []
        for word in sentencs:
            for pa in pattern:
                if word in pa:
                    expand.extend(pa)
                    break
        tmp.extend(old)
        tmp.extend(expand)
        txt.append(tmp)
        out.append([qurey_dict_back[i] for i in tmp])
    with open("out/"+language+"_fp.txt", 'w+') as f:
        f.write("\n".join([" ".join(i) for i in txt]))
    with open("out/"+language+"_test.txt", 'w+') as f:
        f.write("\n".join([" ".join(i) for i in qureys_old]))


qurey_fp=tf.keras.preprocessing.sequence.pad_sequences(out,maxlen=desc_len, padding='post',
                                              truncating='post', value=0)
np.save("out/"+language+"_qurey_fp",qurey_fp)
