import os
import os.path
import glob
import shutil
from tqdm import tqdm

# run under the top level directory

def spliter(sz):
    train_sz = sz * 8 / 10
    test_sz = sz - train_sz
    return train_sz, test_sz

realpaths = glob.glob('./real/real/*.jpg')
pixpaths  = glob.glob('./real/pix/*.jpg')

train_sz, test_sz = spliter(len(realpaths))


for i, realpath in tqdm(enumerate(realpaths)):
    filename = os.path.basename(realpath)

    pixpath = pixpaths[i]

    if i < train_sz:    # train
        shutil.copyfile(realpath, os.path.join('./dataset/train/real', filename))
        shutil.copyfile(pixpath, os.path.join('./dataset/train/pix', filename))
    else:               # test
        shutil.copyfile(realpath, os.path.join('./dataset/test/real', filename))
        shutil.copyfile(pixpath, os.path.join('./dataset/test/pix', filename))