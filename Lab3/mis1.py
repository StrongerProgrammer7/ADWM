import math

import numpy as np
import cv2

def showGassmethod(img,kernel_s,sig):

    kernel_size = kernel_s

    sigma = sig

    center = (kernel_size - 1) // 2

    kernel = np.zeros((kernel_size, kernel_size))


    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center

            kernel[i, j] = np.exp(
                -((x ** 2 - math.ceil(kernel_size / 2)) + (y ** 2 - math.ceil(kernel_size / 2))) / (2 * sigma ** 2)) / (
                                   2 * np.pi * sigma ** 2)

    # print(kernel)

    kernel /= np.sum(kernel) #для того чтобы средняя интенсивность оставалась не изменой.

    # print("Гауссова матрица свертки:")
    # print(kernel)

    result_image = img.copy()

    for i in range(center, result_image.shape[0] - center):
        for j in range(center, result_image.shape[1] - center):

            new_saturation = np.sum(result_image[i - center:i + center + 1, j - center:j + center + 1] * kernel)


            result_image[i, j] = np.clip(new_saturation, 0, 255)

    #Hori = np.concatenate((img, result_image), axis=1)
    #cv2.imshow('Original Image ks=' + str(kernel_s) + '; sigma=' + str(sigma), Hori)
    return result_image

def libGuss(img,kernel_size,sigma):
    return cv2.GaussianBlur(img, kernel_size, sigma)


if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png', cv2.IMREAD_GRAYSCALE)

    scale_percent = 30  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    myGauss = showGassmethod(img,3,1.0)
    libGauss =libGuss(img,(7,7),1.0)


    Hori = np.concatenate((img, myGauss,libGauss), axis=1)
    cv2.imshow('Original Image ks=3  + sigma=1.0', Hori)
    #showGassmethod(img, 3, 1.0)
    #showGassmethod(img, 3, 10.0)
    #showGassmethod(img, 5, 1.0)
    #showGassmethod(img, 5, 10.0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
