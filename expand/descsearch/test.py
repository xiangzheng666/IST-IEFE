import numpy as np
from data_load import get_desc_search,k
from UNFI_model import UNFI


model=UNFI()

with open("language.txt", "r+") as f:
    language = f.read()

with open("../../data/"+language+"/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
    qurey=f.read().split("\n")
    qurey_dict={i:qurey[i] for i in range(len(qurey))}

model.load_weights("../../model/desc_search/" + language + "/UNIF.pth")
print("load model sucess")

for code, api, fun_name, neg_code, neg_api, neg_fun_name, code_desc, qurey, desc in get_desc_search():
    loss=[]
    text=[]
    for i in range(10):
        loss.append([])
        text.append([])
    indexs=model.get_topk_desc([qurey,code_desc])
    t=[" ".join([qurey_dict[j] for j in i if j != 0]) for i in code_desc.numpy()]
    for i in range(len(qurey)):
        for j in range(len(indexs[i])):
            tmp=np.expand_dims(code_desc[indexs[i][j]],axis=0)
            text[j].append(" ".join([qurey_dict[i] for i in code_desc[indexs[i][j]].numpy() if i!=0]))


    f=open("out/"+language+"_qurey.txt","w+")
    f.write("\n".join([" ".join([qurey_dict[i] for i in j if i!=0]) for j in qurey.numpy()]))
    f.close()
    f = open("out/" + language + "_true.txt", "w+")
    f.write("\n".join(t))
    f.close()
    for i in range(len(text)):
        f=open("out/"+language+"_desc_" + str(i + 1)+".txt",'w+')
        f.write("\n".join(text[i]))
        f.close()