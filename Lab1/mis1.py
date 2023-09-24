import cv2
import numpy as np

def showImg(IMREAD_OPT,WINDOW):
    img = cv2.imread(r'../imgs/1.png', IMREAD_OPT)
    alpha = 1.5
    beta = 10
    adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    Hori = np.concatenate((img, adjusted), axis=1)
    cv2.namedWindow('Display window', WINDOW)
    cv2.imshow('Display window', Hori)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    showImg(cv2.IMREAD_GRAYSCALE,cv2.WINDOW_KEEPRATIO)
    showImg(cv2.IMREAD_ANYCOLOR,cv2.WINDOW_NORMAL)
    showImg(cv2.IMREAD_COLOR,cv2.WINDOW_AUTOSIZE)

#Flag creeate WIndow
#WINDOW_KEEPRATIO
#WINDOW_AUTOSIZE
#WINDOW_NORMAL

#Flag read IMG
#IMREAD_GRAYSCALE
#IMREAD_COLOR=1
#IMREAD_ANYDEPTH =2