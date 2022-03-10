import os
import random
from pathlib import Path

def makeTXT(imgfilepath, txtsavepath, trainval_percent=0.1, train_percent=0.9):
    total_img = os.listdir(imgfilepath)
    num = len(total_img)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    ftrainval = open(txtsavepath + '/trainval.txt', 'w')
    ftest     = open(txtsavepath + '/test.txt', 'w')
    ftrain    = open(txtsavepath + '/train.txt', 'w')
    fval      = open(txtsavepath + '/val.txt', 'w')

    for i in list:
        name = os.path.abspath(imgfilepath + '\\' + total_img[i]) + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftest.write(name)
            else:
                fval.write(name)
        else:
            ftrain.write(name)
    print("Finished {} images".format(num))
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

if __name__ == '__main__':
    trainval_percent = 0.1 # set 10% of testing data
    train_percent = 0.9 # set 90% of training data
    imgfilepath = '../datasets/blemish/images'
    txtsavepath = '../datasets/blemish'
    makeTXT(imgfilepath, txtsavepath, trainval_percent, train_percent)