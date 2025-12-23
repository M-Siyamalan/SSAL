import os
import torch
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from sklearn.utils import shuffle
from Util import *

def getDatasetFromDir(root_dir, class_labels):
    fnArr, lblArr = [], []
    class_dirs = os.listdir(root_dir)
    for classNo, className in enumerate(class_dirs):
        class_path = os.path.join(root_dir, className)
        fn_class = os.listdir(class_path)
        for fn in fn_class:
            fnArr.append(os.path.join(class_path, fn))
            lblArr.append(class_labels[className])
    return fnArr, lblArr


def train_test_split(fnArr, lblArr, pLa, random_state=None):
    assert pLa<1, 'test size must be less than 1'
    classes, class_counts = np.unique(lblArr, return_counts=True)
    train_indices = []
    test_indices = []

    for cls in classes:
        cls_indices = [i for i, lbl in enumerate(lblArr) if lbl == cls]
        if pLa <= 1:
            n_train = max(1, int(pLa * len(cls_indices)))  # At least 1 sample in test
        else:
            n_train = pLa
        cls_indices = shuffle(cls_indices, random_state=random_state)
        train_indices.extend(cls_indices[:n_train])
        test_indices.extend(cls_indices[n_train:])

    fnArr_train = [fnArr[i] for i in train_indices]
    fnArr_test = [fnArr[i] for i in test_indices]
    lblArr_train = [lblArr[i] for i in train_indices]
    lblArr_test = [lblArr[i] for i in test_indices]

    return fnArr_train, lblArr_train, fnArr_test, lblArr_test


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, fnArr, lblArr, transform_w, trans_strong=None):
        self.fnArr, self.lbls = fnArr, lblArr
        self.classLbls = set(self.lbls)  # select the unique labels
        self.transform_w = transform_w
        self.trans_strong = trans_strong

    def __getitem__(self, index):
        img_fn = self.fnArr[index]
        y = self.lbls[index]
        I = pil_loader(img_fn)
        if self.trans_strong is None:
            return self.transform_w(I), self.transform_w(I), y, img_fn
        else:
            return self.transform_w(I), self.trans_strong(I), y, img_fn

    def __len__(self):
        return len(self.fnArr)

# def printStatFinalDataloaders(dl_L, dl_test, dl_unlabeled):
#     if dl_unlabeled!=None:
#         y_L, y_te, y_unlabeled = dl_L.dataset.lbls, dl_test.dataset.lbls, dl_unlabeled.dataset.lbls
#         print('LabeledTrain:%5d\nTest:%5d\nUnlabeledTrain:%5d\n' % (len(y_L), len(y_te), len(y_unlabeled)))
#     else :
#         y_L, y_te= dl_L.dataset.lbls, dl_test.dataset.lbls
#         print('LabeledTrain:%5d\nTest:%5d\n' % (len(y_L), len(y_te)))

def get_datasets(dataset, seed, pTe, pLa, bs):
    normalize = transforms.Normalize(mean=[0.7951, 0.7951, 0.7951], std=[0.1879, 0.1879, 0.1879])
    imsize1 = 128 #280
    imsize2 = 96 #224

    tr_w = transforms.Compose([
        transforms.Resize([imsize1, imsize1]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop([imsize2, imsize2]),
        transforms.RandomRotation([0, 180]),
        transforms.ToTensor(),
        normalize
    ])

    tr_s = transforms.Compose([
        transforms.Resize([imsize1, imsize1]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([0, 180]),
        transforms.RandomCrop([imsize2, imsize2]),
        # transforms.RandAugment(),
        # transforms.RandomAffine(degrees=[-180, 180], translate=[0.1, 0.1], scale=[0.8, 1.2]),
        transforms.ToTensor(),
        normalize
    ])

    tt = transforms.Compose([
        transforms.Resize([imsize1, imsize1]),
        transforms.CenterCrop([imsize2, imsize2]),
        transforms.ToTensor(),
        normalize])

    fnArr_U, unlabeledset = None, None
    if dataset == 'BCCD':
        root_dir = '/home/siyam/DATA/BCCD/'
        class_labels = {'EOSINOPHIL': 0, 'LYMPHOCYTE': 1, 'MONOCYTE': 2, 'NEUTROPHIL': 3}
        fnArr_L, lblArr_L = getDatasetFromDir(os.path.join(root_dir, 'TRAIN'), class_labels)
        fnArr_Te, lblArr_Te = getDatasetFromDir(os.path.join(root_dir, 'TEST'), class_labels)
        fnArr_all = fnArr_L + fnArr_Te
        lblArr_all = lblArr_L + lblArr_Te

        # split
        fnArr_L, lblArr_L, fnArr_Te, lblArr_Te = train_test_split(fnArr_all, lblArr_all, 1-pTe, seed)

    elif dataset == 'PBC':
        root_dir = '/home/siyam/DATA/PBC_dataset_normal_DIB/'
        class_labels = {'platelet': 0, 'neutrophil': 1, 'monocyte': 2, 'lymphocyte': 3,
                       'ig':4, 'erythroblast':5,  'eosinophil':6, 'basophil':7}
        fnArr_all, lblArr_all = getDatasetFromDir(root_dir, class_labels)
        fnArr_L, lblArr_L, fnArr_Te, lblArr_Te = train_test_split(fnArr_all, lblArr_all, 1-pTe, seed)
    elif dataset == 'RabinWBC':
        root_dir = '/home/siyam/DATA/Raabin_WBC_Dataset/'
        class_labels = {'Basophil': 0, 'Eosinophil': 1, 'Lymphocyte': 2, 'Monocyte': 3,
                       'Neutrophil':4}
        fnArr_L, lblArr_L = getDatasetFromDir(os.path.join(root_dir, 'Train'), class_labels)
        fnArr_Te, lblArr_Te = getDatasetFromDir(os.path.join(root_dir, 'TestA'), class_labels)
        fnArr_all = fnArr_L + fnArr_Te
        lblArr_all = lblArr_L + lblArr_Te

        # split
        fnArr_L, lblArr_L, fnArr_Te, lblArr_Te = train_test_split(fnArr_all, lblArr_all, 1 - pTe, seed)
    if pLa != 1:
        fnArr_L, lblArr_L, fnArr_U, lblArr_U = train_test_split(fnArr_L, lblArr_L, pLa, seed)

    trainset = MyDataset(fnArr_L, lblArr_L, tr_w, tr_s)
    testset = MyDataset(fnArr_Te, lblArr_Te, tt)

    if pLa != 1:
        unlabeledset = MyDataset(fnArr_U, lblArr_U, tr_w, tr_s)
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size=bs,
                                             num_workers=8,
                                             shuffle=False)

    return trainset, unlabeledset, testloader


