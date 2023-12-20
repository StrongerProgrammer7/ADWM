import cv2
import numpy as np

if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png')
    color = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resize = cv2.resize(img, dim,interpolation=cv2.INTER_AREA)
    resize1 = cv2.resize(color, dim,interpolation=cv2.INTER_AREA)

    horizontal = np.concatenate((resize, resize1), axis=1)
    cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Display window', horizontal)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
