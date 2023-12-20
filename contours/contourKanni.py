import cv2
import numpy as np
import time

kernelSobelX = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

kernelSobelY = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])
# kernelPrewwitX = np.array([
#     [-1, -1, -1],
#     [-1, 8, -1],
#     [-1, -1, -1]
# ])
#
# kernelPrewwitY = np.array([
#     [-1, -1, -1],
#     [-1, 8, -1],
#     [-1, -1, -1]
# ])
kernel_kirsch = np.array([[5, 5, 5],
                          [-3, 0, -3],
                          [-3, -3, -3]])
kernelLaplacian = [[0,1,0],[1,-4,1],[0,1,0]]

kernelPrewwitY = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
kernelPrewwitX = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

kernelRobertX = np.array([[0,1],[-1,0]])
kernelRobertY = np.array([[-1,0],[0,1]])

kernelScharrX = np.array([[3,10,3],[0,0,0],[-3,-10,-3]])
kernelScharrY = np.array([[3,0,-3],[10,0,-10],[-3,-10,-3]])

listCanny = []
listSobel = []
listPrewitt = []
listScharr = []

# def nothing(x):
#     pass
#
# cv2.namedWindow('res')
# cv2.createTrackbar('min','res',0,25,nothing)
# cv2.createTrackbar('max','res',0,25,nothing)

def changeSizeImgPercent(img,percent=50):
    scale_percent = percent  # percent of original size
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

    weak_border_pixel = 255

    arr_index_by_i_high,arr_index_by_j_high = np.where(nms >= high_border)
    arr_index_by_i_lower,arr_index_by_j_lower = np.where(nms < low_border)

    result[arr_index_by_i_high,arr_index_by_j_high] = 0

    for i,j in zip(arr_index_by_i_lower,arr_index_by_j_lower):
        if isAnyPixel_higher_highBorder(nms,i,j,high_border):
            result[i,j] = 0
        else:
            result[i,j] = weak_border_pixel
    result = 255 - result
    return result

def kirsch(image):
    m,n = image.shape
    kirsch = np.zeros((m,n))
    for i in range(2,m-1):
        for j in range(2,n-1):
            d1 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d2 = np.square((-3) * image[i - 1, j - 1] + 5 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d3 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] + 5 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d4 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] -
                  3 * image[i, j - 1] + 5 * image[i, j + 1] - 3 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d5 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] - 3
                  * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] + 5 * image[i + 1, j + 1])
            d6 = np.square((-3) * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] +
                  5 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d7 = np.square(5 * image[i - 1, j - 1] - 3 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] + 5 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])
            d8 = np.square(5 * image[i - 1, j - 1] + 5 * image[i - 1, j] - 3 * image[i - 1, j + 1] +
                  5 * image[i, j - 1] - 3 * image[i, j + 1] - 3 * image[i + 1, j - 1] -
                  3 * image[i + 1, j] - 3 * image[i + 1, j + 1])

            # Первый способ: возьмите максимальное значение в каждом направлении, эффект плохой, используйте другой метод
            list=[d1, d2, d3, d4, d5, d6, d7, d8]
            #kirsch[i,j]= int(np.sqrt(max(list)))
                         # Второй способ: округлить длину модуля в каждом направлении
            kirsch[i, j] =int(np.sqrt(d1+d2+d3+d4+d5+d6+d7+d8))
    return kirsch
    # for i in range(m):
    #     for j in range(n):
    #         if kirsch[i,j]>127:
    #             kirsch[i,j]=255
    #         else:
    #             kirsch[i,j]=0


#-----------------For test optimal values
def gaussBlur(img,ksize,sigmaX):
    return cv2.GaussianBlur(img, ksize, sigmaX)

def checkOptimalGaussBlur():
    i = 0
    while i <= 1:

        img = cv2.imread(f'../imgs/Contours/auto{i}.jpg', cv2.IMREAD_COLOR)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_BGRA2GRAY)
        img = changeSizeImgPercent(img_gray)
        cv2.imshow(f"Blur5x5-{i}", gaussBlur(img,(5,5),0.2))

        i += 1
    cv2.waitKey(0)
    cv2.destroyAllWindows()
#------------------------------------

#----------------------Fot only test work with dataset
#-------------------Test Canny Sobel
def recordColorImgWithContour(canny,img):
    img_cont = img.copy()
    canny = np.array(canny,np.uint8)
    contours, hierarchy = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    min_black = 255
    cnt_black = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for cnt in contours:
        c_area = cv2.contourArea(cnt) + 1e-7
        if cv2.contourArea(cnt) + 1e-7 > 10:
            cv2.drawContours(img_cont, [cnt], -1, (255, 23, 216))
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [cnt], -1, (255, 23, 216), -1)
            temp_mask = cv2.bitwise_and(gray, mask)
            temp_col = np.sum(temp_mask).real / (cv2.contourArea(cnt) + 1e-7)
            if (temp_col < min_black) or (len(cnt_black) == 0):
                cnt_black = cnt
                min_black = temp_col

    if len(cnt_black) != 0:
        cv2.drawContours(img_cont, [cnt_black], -1, (0, 0, 255), 3)
    return img_cont

# using Canny
def testAlgorithmWithDifferentValue(path,count_img,ksize,low_border,hight_border,color=False):
    i=0
    color_path = 'canny_auto'
    global listCanny
    if color == True:
        color_path = 'canny_auto_color'
    while (i < count_img):
        img = cv2.imread(f'{path}/auto{i}.jpg', cv2.IMREAD_COLOR)

        new_img = cv2.Canny(img,low_border,hight_border,apertureSize=ksize)
        if(color==True):
            new_img = recordColorImgWithContour(new_img,img)
        cv2.imwrite(f'{path}/{color_path}/auto_{i}/({ksize})-{low_border}_{hight_border}Auto.jpg',new_img)
        listCanny.append(new_img)
        i+=1


def runTestCanny(ksize, color=False):
    l_h_b = [(10,100),(100,200),(150,230)]
    hb = l_h_b[0][1]
    lb = l_h_b[0][0]
    i = 1
    arr_ib = 0
    t0 = time.time()
    while i <= 9:
        testAlgorithmWithDifferentValue('../imgs/Contours', 3, ksize, lb, hb, color=color)
        if(i%3 == 0 and i < 9):
            arr_ib +=1
            hb = l_h_b[arr_ib][1]
            lb = l_h_b[arr_ib][0]
            print(f"Stage : {arr_ib - 1} completed! ")
        i+=1
    t1 = time.time()
    print("Task(Canny) completed " + str(ksize) + "! Processing took: " + str(t1 - t0) + "sec")

#----------------------------------------------------------

#--- Using realisation Canny with different operator
def call_x_y(operator,img,operatorKSize=3,otherKernelX=None,otherKernelY=None):
    global listSobel,listPrewitt,listScharr
    if (operator == 'Sobel'):
        x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=operatorKSize)
        y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=operatorKSize)
        listSobel.append(x+y)
        return x,y
    else:
        x = cv2.filter2D(img,-1, otherKernelX)
        y = cv2.filter2D(img, -1, otherKernelY)
        if operator=='Prewitta':
            listPrewitt.append(x+y)
        elif operator =='Scharr':
            listScharr.append(x+y)
        return x,y

def callGradLen_angle(operator,img,operatorKSize=3,otherKernelX=None,otherKernelY=None):
    x ,y = call_x_y(operator,img,operatorKSize,otherKernelX,otherKernelY)
    grad_len = np.sqrt(y ** 2.0 + x ** 2.0)
    grad_angle = np.arctan2(y, x)
    return grad_len,grad_angle


def callKanni(path,operator,ksize, border,sigma=(0.5,0.5),operatorKSize=3,color=False,count_img=5):
    color_path = 'canny_auto'
    if color == True:
        color_path = 'canny_auto_color'
    i = 0
    while(i<count_img):
        img = cv2.imread(f'{path}/auto{i}.jpg', cv2.IMREAD_COLOR)
        imgCopy = img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        if(operator != 'Sobel'):
            img = cv2.GaussianBlur(img, ksize,sigmaX=sigma[0],sigmaY=sigma[1])

        if(operator == 'Prewitta'):
            grad_len, grad_angle = callGradLen_angle(operator, img, otherKernelX=kernelPrewwitX, otherKernelY=kernelPrewwitY)
        elif(operator== 'Kirsh'):
            nms = kirsch(img)
        elif operator =='Scharr':
            grad_len,grad_angle = callGradLen_angle(operator,img,otherKernelX=kernelScharrX,otherKernelY=kernelScharrY)
        else:
            grad_len, grad_angle = callGradLen_angle(operator, img,operatorKSize)

        if(operator != 'Kirsh'):
            nms = non_max_suppression(grad_len, grad_angle)

        new_img = double_thresh_filter(nms,border[0],border[1]) #0 90

        if(color == True):
            new_img = recordColorImgWithContour(new_img, imgCopy)
        #cv2.imwrite(f'{path}/{color_path}/{operator}/auto_{i}/({ksize})-{border[0]}_{border[1]}Auto.jpg', new_img)
        cv2.imshow(f"Operator-{operator}-{i}", img)
        i+=1


def runTestCannyOtherOperator(operator,ksize,sigma,color=False):
    t0 = time.time()
    l_h_b = [(10, 100), (100, 200), (150, 230)]
    hb = l_h_b[0][1]
    lb = l_h_b[0][0]
    i = 1
    arr_ib = 0
    while i <= 9:
        callKanni('../imgs/Contours', operator, ksize, (lb, hb),sigma, color=color)
        if 0 == i%3 and i < 9:
            arr_ib +=1
            hb = l_h_b[arr_ib][1]
            lb = l_h_b[arr_ib][0]
            print(f"Stage : {arr_ib-1} completed! ")
        i+=1
    t1 = time.time()
    print("Task completed ksize="+ str(ksize) + "! Processing took: " + str(t1 - t0) + "sec")


def averageSquaredDifference(listA, listB):
    sum = 0
    for imageA, imageB in zip(listA, listB):
        sum += squaredDifference(imageA, imageB)
    return sum / len(listA)


def squaredDifference(imageA, imageB):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    img1 = cv2.resize(imageA,(640,480))
    img2 = cv2.resize(imageB, (640, 480))
    err = np.sum((img1.astype("float") - img2.astype("float")) ** 2)
    err /= float(img1.shape[0] * img2.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    #print(f"MSE - {err}")
    return err

if __name__ == '__main__':
    #callKanni('../imgs/Contours', "Sobel", (3, 3), (150, 230), (0.1, 0.1), color=True)
    callKanni('../imgs/Contours', "Prewitta", (3, 3), (150, 230), (0.1, 0.1), color=False)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #callKanni('../imgs/Contours', "Prewitta", (3, 3), (150, 230), (0.1, 0.1), color=True)
    #callKanni('../imgs/Contours', "Scharr", (3, 3), (150, 230), (0.1, 0.1), color=False)
    #callKanni('../imgs/Contours', "Scharr", (3, 3), (150, 230), (0.1, 0.1), color=True)
    #callKanni('../imgs/Contours', "Kirsh", (3, 3), (150, 230), (0.1, 0.1), color=False)
    #callKanni('../imgs/Contours', "Kirsh", (3, 3), (150, 230), (0.1, 0.1), color=True)
    '''
    runTestCannyOtherOperator('Sobel', (3, 3), (0.5, 0.5))
    runTestCannyOtherOperator('Sobel', (5, 5), (0.3, 0.3))
    runTestCannyOtherOperator('Sobel', (7, 7), (0.1, 0.1))
    '''
    '''
    runTestCannySobel(3)
    runTestCannyOtherOperator('Scharr', (3, 3), (0.5, 0.5))
    runTestCannyOtherOperator('Prewitta', (3, 3), (0.5, 0.5))
    SobelPrewitt = averageSquaredDifference(listCanny,listPrewitt)
    SobelScharr = averageSquaredDifference(listCanny,listScharr)
    PrewittScharr = averageSquaredDifference(listPrewitt,listScharr)
    
    print(f"MSE Sobel Prewitt {SobelPrewitt}")
    print(f"MSE Sobel Scharr {SobelScharr}")
    print(f"MSE Prewitt Scharr {PrewittScharr}")'''
    '''
    runTestCannyOtherOperator('Prewitta',(3,3),(0.5,0.5))
    runTestCannyOtherOperator('Prewitta', (5, 5), (0.3, 0.3))
    runTestCannyOtherOperator('Prewitta', (7, 7), (0.1, 0.1))'''
    '''
    runTestCanny(3, True)
    runTestCanny(5, True)
    runTestCanny(7, True)
    '''
    '''
    runTestCanny(3)
    runTestCanny(5)
    runTestCanny(7)
    '''
