from pyflann import *
import numpy as np

#earliest ann implementation

def submatrix(wanted, test):
    wanted_set = set(wanted)

    @np.vectorize
    def selected(elmt): return elmt in wanted_set

    return test[selected(test[:, 4])]
if(True):
    only=None
    index = np.reshape(np.loadtxt('C:/Users/DFCTech/Downloads/mu/all.idx', usecols=range(1),max_rows=only,dtype='i4'),(-1,1))
    print('crazy')
    data = np.loadtxt('C:/Users/DFCTech/Downloads/mu/all.dta', usecols=range(4),max_rows=only,dtype='i4')


    together = np.concatenate((data,index),1)
    np.save("together.npy",together)
else:
    together=np.load('together.npy')
testindexes = [5]
trainidexes = [1,2,3,4]

testtog = submatrix(testindexes,together)
traintog = submatrix(trainidexes,together)

trainscores = traintog[:,[3]].flatten()
trainpar = traintog[:,[0,1,2]]

testpar = testtog[:,[0,1,2]]
print('data read')

#do knn

dataset = trainpar
testset = testpar
flann = FLANN()
k=5

result, dists = flann.nn(
    dataset, testset, k, target_precision=0.1, algorithm="kdtree",build_weight=0.01,memory_weight=1)

n=np.size(testset,0)
scores = np.empty(n)
for i in range(n):
    s=0
    nk = 0
    for j in range(k):
        if result[i,j]<len(scores):
            s+=trainscores[result[i,j]]
            nk=nk+1
    if nk==0:
        scores[i]=-1
    else:
        scores[i]=s/nk

print('knn done')

#output result
np.savetxt('C:/Users/DFCTech/Desktop/testout.dta', scores, fmt='%.2f',delimiter='\n')

print('done')

