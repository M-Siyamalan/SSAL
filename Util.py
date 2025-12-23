import numpy as np

def calWeights(lbl_arr):
    weights = []
    unique_lbls = set(lbl_arr)
    for lbl in unique_lbls:
        idx = [i for i, x in enumerate(lbl_arr) if x == lbl]  #store the x=lbl index are in the idx array
        weights.append(1/len(idx))  #len(idx) =length of the array
    weights = np.asarray(weights)   #all weights for the lbl are stored here using numpy array
    weights = weights/sum(weights)  #   normalize to one
    return weights

def printStatDataloaders(dl_L, dl_U, dl_test, weights=None):
    y_L, y_te = dl_L.dataset.lbls, dl_test.dataset.lbls
    y_U = None
    totU = 0
    if dl_U is not None:
        y_U = dl_U.dataset.lbls
        totU = len(y_U)
    unique_lbls = set(y_te)

    def countLbls(lblArr, lbl):
        idx = [i for i, x in enumerate(lblArr) if x == lbl]
        return len(idx)

    print('lbl \t Label \t Unlbl \t Test')
    for lbl in unique_lbls:
        tmp_L = countLbls(y_L, lbl)
        tmp_U = 0
        if y_U is not None:
            tmp_U = countLbls(y_U, lbl)
        tmp_te = countLbls(y_te, lbl)
        print('%1d\t%5d\t%5d\t%5d'%(lbl, tmp_L, tmp_U, tmp_te))
    print(' \t%5d\t%5d\t%5d' % (len(y_L), totU, len(y_te)))

    if weights is not None:
        print("Weights :", end='')
        for val in weights:
            print(np.round(val.cpu().numpy(), 3), end=',')
    print('\n-----------------\n')


