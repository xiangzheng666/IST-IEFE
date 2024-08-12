import string,nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# f = open("QUERIES_for_training.302-450","r")
# fout = open("QUERIES_for_training_Expanded.302-450","w", encoding="utf-8")
stop_words = set(stopwords.words("english"))

# nltk.download("omw-1.4")

def util(line):
    # if not line:
    #     break
    _line = line
    line = line.replace('\n', '')
    line = line.split(" ", 1)
    new_line = line[0]
    # line = ["i", "have", "a", "pen"]
    line[1] = line[1].lower()
    line[1] = line[1].translate(str.maketrans('', '', string.punctuation))
    # print(line)
    word_tokens = word_tokenize(line[1])
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    synonyms = []
    for x in filtered_sentence:
        m=0
        for syn in wordnet.synsets(x):
            count = 0
            for l in syn.lemmas():
                if count < 1 and m < 3:
                    if l.name() not in synonyms:
                        tmp=l.name()
                        synonyms.extend(str(tmp).split("_"))
                        count += 1
                        m +=1

    synonyms_string = ' '.join(synonyms)
    # new_line=" ".join([str(new_line),synonyms_string])
    synonyms = []
    return synonyms_string
    #     fout.write(new_line)
    #     fout.write('\n')
    # f.close()
    # fout.close()
if __name__ == '__main__':
        print(util("how do i plot bars using pandas"))