from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
from math import log
import operator
import pickle

def createData():
    dataSet = [
        [0, 0, 0, 0, 'no'], # 最后一列为结果：是否贷款
        [0, 0, 0, 1, 'no'],
        [0, 1, 0, 1, 'yes'],
        [0, 0, 0, 0, 'no'],
        [1, 0, 0, 0, 'no'],
        [1, 0, 0, 1, 'no'],
        [1, 1, 1, 1, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [1, 0, 1, 2, 'yes'],
        [2, 0, 1, 2, 'yes'],
        [2, 0, 1, 1, 'yes'],
        [2, 1, 0, 1, 'yes'],
        [2, 1, 0, 2, 'yes'],
        [2, 0, 0, 0, 'no'],
    ]
    labels = ['F1-AGE', 'F2-WORK', 'F3-HOME', 'F4-LOAN']
    return dataSet, labels

def createTree(dataset, labels, featLabels):
    # 如果已经是同一类了，则分好了
    clsList = [row[-1] for row in dataset]
    if clsList.count(clsList[0]) == len(clsList):
        return clsList[0]
    # 只有最后一列，则不需要再分了，此时采用少数服从多数的原则，使用其众数。
    if len(dataset[0]) == 1:
        return majority(clsList)
    bestFeatIdx = chooseBestFeature(dataset)
    bestFeatLabel = labels[bestFeatIdx]
    featLabels.append(bestFeatLabel)
    tree = {bestFeatLabel:{}}
    # 删除当前 label，继续对下一级 label 进行分割
    del labels[bestFeatIdx]
    curFeatValues = [row[bestFeatIdx] for row in dataset]
    uniqVal = set(curFeatValues)
    for val in uniqVal:
        subLabels = labels[:]
        tree[bestFeatLabel][val] = createTree(splitDataSet(dataset, bestFeatIdx, val), subLabels, featLabels)
    return tree

def chooseBestFeature(dataset):
    featNum = len(dataset[0]) - 1 # 最后一列为 label
    baseEntropy = calcShannonEntropy(dataset) # 基准熵，用于计算增益率。这里主要是避免纯熵运算面临的稀疏问题。
    bestInfoGain = 0
    bestFeat = -1
    # 下面分别算如果采用某属性特征进行分割后，对应的熵，以此比较哪种属性分割会带来更好的收益
    for i in range(featNum):
        values = [row[i] for row in dataset]
        uniqVal = set(values)
        newEntropy = 0
        for val in uniqVal:
            subDataset = splitDataSet(dataset, i, val)
            prob = len(subDataset)/float(len(dataset))
            newEntropy += prob * calcShannonEntropy(subDataset) # 各子节点的熵乘该节点的概率（类似于求加权平均），得到分裂后的熵
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeat = i
    return bestFeat


def splitDataSet(dataset, axis, val):
    '''
    分裂数据集。
    val: 此节点的属性值。
    '''
    subSet = []
    for row in dataset:
        if row[axis] == val:
            newRow = row[:axis]
            newRow.extend(row[axis+1:])
            subSet.append(newRow)
    return subSet

def calcShannonEntropy(dataset):
    '''
    计算熵值。
    '''
    total = len(dataset)
    cntPerLabel = {}
    for label in [row[-1] for row in dataset]:
        if label not in cntPerLabel:
            cntPerLabel[label] = 0
        cntPerLabel[label] += 1
    entropy = 0.0
    for label, cnt in cntPerLabel:
        p = cnt/float(total)
        entropy -= p * log(p, 2) # 对数底不影响运算，仅是为了使定义域 (0, 1] 映射到较大的值域。负号是为了抵消 log 引起的取反.
    return entropy


if __name__ == "__main__":
    dataset, labels = createData()
    featLabels = []
    tree = createTree(dataset, labels, featLabels)
    