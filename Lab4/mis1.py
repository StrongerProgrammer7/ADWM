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

def changeSizeImg(img):
    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return img

def isAngel_nearby_bound(angle):
    return (0 <= angle < 22.5) or (157.5 <= angle <= 180)
def isPixel_localMaxByDirectionGradient(grad_curPixel,gradNeighborByDir_1,gradNeighborByDir_2):
    return grad_curPixel >= gradNeighborByDir_1 and grad_curPixel >= gradNeighborByDir_2

def isAnyPixel_higher_highBorder(nms,i,j,high_border):
    return np.any(nms[i-1:i+1,j-1:j+1] >= high_border)

def getAngelByDeg(angel_by_value):
    return angel_by_value * 180 / np.pi

def get_intesitiesNeighbors(
        right_neighbor, left_neighbor,
        up_neighbor, down_neighbor,
    left_up_neighbor, left_down_neighbor,
    right_up_neighbor,right_down_neighbor,angle):
    if isAngel_nearby_bound(angle):
        return [right_neighbor,left_neighbor]#[grad_len[i, j + 1], grad_len[i, j - 1]]
    elif 22.5 <= angle < 67.5:
        return [left_down_neighbor,right_up_neighbor] # grad_len[i + 1, j - 1], grad_len[i - 1, j + 1]
    elif 67.5 <= angle < 112.5:
        return [down_neighbor,up_neighbor]#[grad_len[i + 1, j], grad_len[i - 1, j]]
    elif 112.5 <= angle < 157.5:
        return  [left_up_neighbor,right_down_neighbor]#grad_len[i - 1, j - 1], grad_len[i + 1, j + 1]
    return [0,0]

def non_max_suppression(grad_len, directionsVeryFastChangeFunc):
    rows,cols = grad_len.shape
    result = np.zeros_like(grad_len)

    for i in range(1,rows-1):
        for j in range(1, cols-1):
            angle = getAngelByDeg(directionsVeryFastChangeFunc[i,j])
            if angle < 0:
                angle += 180

            #хранения значений интенсивности пикселей в двух соседних пикселях относительно текущего пикселя
            intensities_neighbor,intensities_neighbor2 = get_intesitiesNeighbors(
                grad_len[i, j + 1], grad_len[i, j - 1],
                grad_len[i - 1, j],grad_len[i + 1, j],
                grad_len[i - 1, j - 1],grad_len[i + 1, j - 1],
                grad_len[i - 1, j + 1],grad_len[i + 1, j + 1],
                angle
            )

            if isPixel_localMaxByDirectionGradient(grad_len[i,j],intensities_neighbor,intensities_neighbor2):
                result[i,j] = grad_len[i,j]

    return result

def double_thresh_filter(nms, low_border, high_border):
    result = np.zeros_like(nms)

    weak_border_pixel = 100

    arr_index_by_i_high,arr_index_by_j_high = np.where(nms >= high_border)
    arr_index_by_i_lower,arr_index_by_j_lower = np.where(nms < low_border)

    result[arr_index_by_i_high,arr_index_by_j_high] = 255

    for i,j in zip(arr_index_by_i_lower,arr_index_by_j_lower):
        if isAnyPixel_higher_highBorder(nms,i,j,high_border):
            result[i,j] = 255
        else:
            result[i,j] = weak_border_pixel

    return result



if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png', cv2.IMREAD_COLOR)

    img_hsv= cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_BGRA2GRAY)
    img = changeSizeImg(img_gray)

    img_hsv_gray_gauss = cv2.GaussianBlur(img, (13, 13), 0.2)

    #cv2.imshow("Blur",img)

    calc_grad_x = cv2.Sobel(img_hsv_gray_gauss,cv2.CV_64F,1,0,ksize=3)
    calc_grad_y = cv2.Sobel(img_hsv_gray_gauss,cv2.CV_64F,0,1,ksize=3)

    # cv2.imshow("GradX", calc_grad_x)
    #cv2.imshow("GradY", calc_grad_y)
    #print(calc_grad_y)
    #print(calc_grad_x)
    grad_len = np.sqrt(calc_grad_y**2.0 + calc_grad_x**2.0)
    grad_angle = np.arctan2(calc_grad_y,calc_grad_x)
    #print(grad_len)
    #print(grad_angle)

    nms = non_max_suppression(grad_len,grad_angle)
   # cv2.imshow('NMS',nms)

    while (1):
        if cv2.waitKey(1) & 0xFF == 27:
            break
        maxVal = cv2.getTrackbarPos('max', 'res')
        minVal = cv2.getTrackbarPos('min', 'res')
        canny = double_thresh_filter(nms, minVal * 10, maxVal * 10)
        cv2.imshow('res', canny)
    #canny_img = hysteresis_thresholding(nms,45,255) #(10,200)
    #cv2.imshow('GRAY', img)
    #cv2.imshow('GRAY', canny_img)
    #cv2.waitKey(0)
    cv2.destroyAllWindows()

#cv2.CV_64F: Тип данных, в котором хранятся результаты вычислений (64-битное число с плавающей запятой).
#1: Порядок производной по горизонтали (горизонтальный градиент).
#0: Порядок производной по вертикали (вертикальный градиент).