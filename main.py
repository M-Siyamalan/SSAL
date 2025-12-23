import os
import argparse
import sys
import numpy as np
import time
from logger import *
from CNNTrain import CNNTrain
from datetime import timedelta

desc = ''
print(100*'*')
print(desc)
print(100*'*')

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default='BCCD') #'BCCD','RabinWBC','PBC'
parser.add_argument('--lr', default=5e-4, type=float) #5e-4 for ResNet, 1e-4 for DenseNet
parser.add_argument('--GPU_id', default=0, type=int)
parser.add_argument('--n_epochs', default=30, type=int) #30
parser.add_argument('--bs', default=64, type=int)
parser.add_argument('--pTe', default=0.2, type=float)
parser.add_argument('--pL', default=0.01, type=float)
parser.add_argument("--b", default=25, type=int)
parser.add_argument("--ModelName", default='resnet18')  #'resnet18' 'resnet50' 'DenseNet' 'mobilenet', 'ShuffleNet'
parser.add_argument("--useSSL", default=False)
parser.add_argument("--useAL", default=False)
parser.add_argument("--thr", default=0.99, type=float)
parser.add_argument("--useEMA", default=True)
parser.add_argument("--num_ite", default=5)
args = parser.parse_args()

if args.ModelName == 'DenseNet':
    args.lr = 5e-4
    args.n_epochs = 30
elif args.ModelName == 'resnet18' or args.ModelName == 'resnet50':
    args.lr = 5e-4
    args.n_epochs = 30
elif args.ModelName == 'mobilenet':
    args.lr = 1e-3
    args.n_epochs = 30
elif args.ModelName == 'ShuffleNet':
    args.lr = 1e-4
    args.n_epochs = 20

dirname = os.path.join('/home/siyam/SemiSupActiveLearning/Results/', args.dataset+'_'+args.ModelName)
if not os.path.exists(dirname):
    os.makedirs(dirname)
fn = os.path.join(dirname, desc +  'useSSL-' + str(args.useSSL) + '-useAL-' + str(args.useAL) + '.txt')
args.dirname = dirname
print(fn)
sys.stdout = Logger(fn)

def printVals(desc, paraArr, mvArr, stdvArr):
    paraArr = np.stack(paraArr, axis=0)
    mvArr = np.stack(mvArr, axis=0)
    stdvArr = np.stack(stdvArr, axis=0)
    print(150*'-')
    print(desc)
    for r in range(len(paraArr)):
        para = paraArr[r]
        for e in para:
            print(e, ':', end='')
        # print()
        mv = mvArr[r]
        stdv = stdvArr[r]
        for i in range(len(mv)):
            if i % len(desc) == 0:
                print(' \n', end='')
            print('&$', mv[i], '\pm', stdv[i], end = '$ \t')
        print()
        print()
    print(150 * '-')


def testOne(opt):
    print(10 * '-')
    re_all = []
    for i in range(3):
        st = time.time()
        print(30*'-')
        print(i)
        print(30 * '-')
        opt.itr = i
        opt.seed = 500 * (i + 1)
        print(opt)
        fm = CNNTrain(opt)
        re, desc = fm.iterate()

        re_all.append(re)
        print(opt)
        elapsed_time = (time.time() - st)
        print("execution time: " + str(timedelta(seconds=elapsed_time)))

        exit()


    re_all = np.stack(re_all, axis=0)
    re_all_mean = np.round(np.mean(re_all, axis=0), 2)
    re_all_std = np.round(np.std(re_all, axis=0), 2)
    return re_all_mean, re_all_std, desc



# for p in [0.01,0.02,0.05,1]:
#     args.pL = p
#     args.seed = 1
#     fm = CNNTrain(args)
#
# exit()



para = []
re_mvArr = []
re_stdArr = []
arr = [[1,1]]

for p in [0.05]:
    args.p = p
    for a in arr:
        args.pL = p
        args.useSSL = a[0]
        args.useAL = a[1]

        print(desc)
        para.append([p, args.useSSL, args.useAL, args.thr])
        testOne(args)
        re_m, re_std, desc = testOne(args)
        re_mvArr.append(re_m)
        re_stdArr.append(re_std)
        printVals(desc, para, re_mvArr, re_stdArr)