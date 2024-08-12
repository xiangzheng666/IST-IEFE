import tensorflow as tf
import tensorflow_hub as hub

from tensorflow import keras

with open("language.txt", "r+") as f:
    language = f.read()


class bertmodel(keras.Model):
    def __init__(self):
        super(bertmodel, self).__init__()
        self.qurey_bert_encoder = hub.KerasLayer("bert",trainable=True)
        self.desc_bert_encoder = hub.KerasLayer("bert", trainable=True)

        with open("bert/assets/vocab.txt", "r+",encoding="utf-8") as f:
            tmp = f.read().split("\n")
            self.bert_dict = {tmp[i]:i for i in range(len(tmp))}

        with open("../../data/" + language + "/vocab/vocab_qurey_desc.txt", "r+", encoding="utf-8") as f:
            qurey = f.read().split("\n")
            self.dict = {i: qurey[i] for i in range(len(qurey))}

    def transfer_input(self,input):
        num2str=[[self.dict[j] for j in i if j!=0] for i in input.numpy()]
        str2nm=[[self.bert_dict[j] for j in i if j in self.bert_dict.keys()] for i in num2str]
        input_ids=tf.keras.preprocessing.sequence.pad_sequences(str2nm, maxlen=20, padding='post',
                                                      truncating='post', value=0)
        input_mask=tf.cast(tf.math.not_equal(input_ids, 0), tf.int32)
        input_segment=tf.zeros_like(input_ids)
        return {"input_word_ids":input_ids,"input_mask":input_mask,"input_type_ids":input_segment}

    def call(self, inputs, training=None, mask=None):
        qurey, desc, neg_desc = [self.transfer_input(i) for i in inputs]

        qurey=self.qurey_bert_encoder(qurey)["pooled_output"]
        desc = self.desc_bert_encoder(desc)["pooled_output"]
        neg_desc = self.desc_bert_encoder(neg_desc)["pooled_output"]

        return neg_desc,desc,qurey

    def qurey(self,inputs):
        qurey,desc=[self.transfer_input(i) for i in inputs]

        desc_vect = self.desc_bert_encoder(desc)["pooled_output"]
        qurey_vect = self.qurey_bert_encoder(qurey)["pooled_output"]

        return qurey_vect, desc_vect