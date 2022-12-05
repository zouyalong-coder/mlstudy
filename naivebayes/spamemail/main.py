import numpy as np
import re
import os
import csv
import random


class Doc(object):
    def __init__(self, label: int, text: str) :
        self.label = label
        self.text = text
        self.words = textParse(text)


DocSet = list[Doc]


def buildVocablist(ds: DocSet) -> list[str]: 
    vs = set([])
    for doc in ds:
        vs |= set(doc.words)
    return list(vs)


def doc2Vec(vs: list[str], doc: Doc) -> list[int]:
    '''
        计算文档的词向量，仅记是否出现，不管次数
    '''
    vec = [0] * len(vs)
    for word in doc.words:
        if word in vs:
            idx = vs.index(word)
            vec[idx] = 1
    return vec

class Predictor(object):
    def __init__(self, pSpam, pdSpamLog, pdHamLog, vs):
        self.pSpam = pSpam
        self.pdSpamLog = pdSpamLog
        self.pdHamLog = pdHamLog
        self.vs = vs
        
    def predict(self, doc: Doc) -> int:
        vec = np.array(doc2Vec(self.vs, doc))
        pSpamPredict = np.log(self.pSpam) + sum(vec * self.pdSpamLog) # 左右两边都使用了 log ，所以乘法变加法了
        pHamPredict = np.log(1.0-self.pSpam) + sum(vec * self.pdHamLog)
        if pSpamPredict > pHamPredict:
            return 1
        return 0

    def test(self, testSet: DocSet):
        total = len(testSet)
        err = 0
        for doc in testSet:
            predict = self.predict(doc)
            if doc.label != predict:
                err += 1
        return err, total, float(total - err) / total
         
    @classmethod
    def train(cls, data: DocSet):
        vs = buildVocablist(data)
        trainSize = len(data)
        wordsNum = len(vs)
        trainMat = []
        trainCls = []
        print("transform doc to vector now")
        for idx, doc in enumerate(data):
            if idx % 20 == 1 :
                print("handle {}/{} now".format(idx, trainSize))
            trainMat.append(np.array(doc2Vec(vs, doc)))
            trainCls.append(doc.label)
        pSpam = sum(trainCls) / float(trainSize)
        # PHam = 1.0 - PSpam
        # 平滑处理：不能用 zeros 的原因是，在独立计算时，由于是连乘，如果出现了一个0（即语料库中的某个词不出现），就会导致其整体概率为0
        spamWordsNum = np.ones((wordsNum)) 
        hamWordsNum = np.ones((wordsNum))
        totalSpamWordNum = 2 # 通常设置成类别个数
        totalHamWordNum = 2
        print("count words ...")
        for idx, vec in enumerate(trainMat):
            if idx % 20 == 1 :
                print("handle {}/{} now".format(idx, trainSize))
            cls = trainCls[idx]
            if cls == 1: # spam
                spamWordsNum += vec
                totalSpamWordNum += sum(vec)
            else:
                hamWordsNum += vec
                totalHamWordNum += sum(vec)
        # 各个词在分类邮件中的词频，表示其概率(一个词在一个邮件中出现最多只算一次)
        pdSpam = np.log(spamWordsNum/totalSpamWordNum) # 本身概率数据可能太小，但它们都在(0,1]之间，故考虑使用对数进行归一化
        pdHam = np.log(hamWordsNum/totalHamWordNum)
        print("train over: pSpam={}\n\tpdSpam={}\n\tpdHam={}", pSpam, pdSpam, pdHam)
        return Predictor(pSpam, pdSpam, pdHam, vs)


def textParse(content: str):
    listOfWords = re.split(r'\W+', content)
    return [word.lower for word in listOfWords if len(listOfWords) >= 2]


def dealInput(file: str):
    trainSet = []
    testSet = []
    convertLabel = lambda label: 1 if label == "spam" else 0
    with open(file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        cnt = 0
        for row in reader:
            if len(row) < 3:
                continue
            d = Doc(label=convertLabel(row[1]), text=row[2])
            if random.randint(0, 99) <=90:
                trainSet.append(d)
            else:
                testSet.append(d)
    return (trainSet, testSet)


def main(input: str):
    trainSet, testSet = dealInput(input)
    predictor = Predictor.train(trainSet)
    err, total, rightRatio = predictor.test(testSet)
    print("trainSet: {}, testSet: {}".format(len(trainSet), len(testSet)))
    print("ratio: err={}, total={}, {:.2f}%".format(err, total, rightRatio*100.0))


if __name__ == "__main__":
    file = os.path.dirname(__file__) + "/spam_ham_dataset.csv"
    main(file)