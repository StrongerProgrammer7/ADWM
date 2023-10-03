import cv2
import numpy as np

matrGx = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

matrGy = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])
def nothing(x):
    pass

cv2.namedWindow('res')
cv2.createTrackbar('min','res',0,25,nothing)
cv2.createTrackbar('max','res',0,25,nothing)


def non_max_suppression(grad_len,grad_angle):
    rows,cols = grad_len.shape
    result = np.zeros_like(grad_len)

    for i in range(1,rows-1):
        for j in range(1, cols-1):
            angle = grad_angle[i,j] * 180 / np.pi
            if angle < 0:
                angle += 180

            q,r = 0,0
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q,r = grad_len[i,j + 1], grad_len[i,j-1]
            elif 22.5 <= angle < 67.5:
                q,r = grad_len[i+1,j-1], grad_len[i-1,j+1]
            elif 67.5 <= angle < 112.5:
                q,r = grad_len[i+1,j], grad_len[i-1,j]
            elif 112.5 <= angle < 157.5:
                q,r = grad_len[i-1,j-1], grad_len[i+1,j+1]

            if grad_len[i,j] >= q and grad_len[i,j] >= r:
                result[i,j] = grad_len[i,j]

    return result

def hysteresis_thresholding(nms,low_border,hight_border):
    rows,cols = nms.shape
    result = np.zeros_like(nms)

    weak_pixels = 25

    strong_i,strong_j = np.where(nms >= hight_border)
    zeros_i,zeros_j = np.where(nms < low_border)

    result[strong_i,strong_j] = 255
    for i,j in zip(zeros_i,zeros_j):
        if np.any(nms[i-1:i+2,j-1:j+2] >= hight_border):
            result[i,j] = 255
        else:
            result[i,j] = weak_pixels

    return result


if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png', cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    img = cv2.GaussianBlur(img, (13, 13), 0.2)


    calc_grad_x = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    calc_grad_y = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    #print(calc_grad_y)
    #print(calc_grad_x)
    grad_len = np.sqrt(calc_grad_y**2.0 + calc_grad_x**2.0)
    grad_angle = np.arctan2(calc_grad_y,calc_grad_x)
    #print(grad_len)
    #print(grad_angle)

    nms = non_max_suppression(grad_len,grad_angle)

    while (1):
        if cv2.waitKey(1) & 0xFF == 27:
            break
        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        canny = hysteresis_thresholding(nms,minVal*10,maxVal*10)
        cv2.imshow('res', canny)
    #canny_img = hysteresis_thresholding(nms,45,255) #(10,200)
    #cv2.imshow('GRAY', img)
    #cv2.imshow('GRAY', canny_img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
