import cv2 as cv
import glob
import os

cnt = 1
paths = glob.glob('./real/*.jpg')
for path in tqdm(paths):
    image=cv.imread(path)
    out = cv.imread('./pix/'+os.path.basename(path))
    con=np.concatenate((image,out),axis=1)
    cv.imwrite(str(cnt)+'.jpg',con)
    cnt = cnt+1