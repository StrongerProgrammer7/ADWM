import cv2
import numpy
import numpy as np

if __name__ == '__main__':
    img = cv2.imread(r'../imgs/2.jpg')
    # cv2.namedWindow('Display window', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)

    scale_percent = 8  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    height, width, _ = img.shape

    rect1_x = width // 2 - 100
    rect1_y = height // 2 - 40
    rect1_width = width - 200
    rect1_height = rect1_y + 30

    start_point1 = rect1_x, rect1_y
    end_point1 = rect1_width, rect1_height
    color = (255, 0, 0)
    thickness = 4

    rect2_x = rect1_x + (rect1_x // 2) - 18
    rect2_y = 50
    rect2_width = rect2_x + 30
    rect2_height = height - 100

    start_point = rect2_x, rect2_y
    end_point = rect2_width, rect2_height

    cat1 = img[start_point1[1]:end_point1[1], start_point1[0]:end_point1[0]]  # cut image
    cat2 = img[start_point[1]:end_point[1], start_point[0]:end_point[0]]
    # cv2.imshow('Cut', cat1)
    cat1 = cv2.blur(cat1, (10, 10))
    cat2 = cv2.blur(cat2, (10, 10))

    cv2.rectangle(img, start_point, end_point, color, thickness)
    cv2.rectangle(img, start_point1, end_point1, color, thickness)

    img[start_point1[1]:end_point1[1], start_point1[0]:end_point1[0]] = cat1
    img[start_point[1]:end_point[1], start_point[0]:end_point[0]] = cat2

    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    '''
    image = cv2.imread(r'./imgs/2.jpg')

    scale_percent = 8  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    newImage = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    blank_image = np.zeros(newImage.shape, dtype=np.uint8)

    rect1_x = width // 2 - 100
    rect1_y = height // 2 - 40
    rect1_width = width - 200
    rect1_height = rect1_y + 30

    rect2_x = rect1_x + (rect1_x // 2) - 18
    rect2_y = 50
    rect2_width = rect2_x + 30
    rect2_height = height - 100

    blur_kernel_size = (31, 31)

    overlay = newImage.copy()
    overlay[rect2_y:rect2_y+5 + rect2_height-65, rect2_x:rect2_x + 28] = cv2.GaussianBlur(
        overlay[rect2_y:rect2_y+5 + rect2_height-65, rect2_x:rect2_x + 28], (51, 51), 0)

    #rect2 = np.zeros((rect2_x, rect2_y, 4), dtype=np.uint8)
    rect2 = newImage.copy()
    rect2 = cv2.rectangle(rect2, (rect2_x, rect2_y), (rect2_width, rect2_height),
                  (0, 255, 0),
                  thickness=2,
                   lineType=8 , shift=0)
    #rect2 = cv2.GaussianBlur(rect2, blur_kernel_size, 0)

    rect1 =  newImage.copy()
    cv2.rectangle(rect1, (rect1_x, rect1_y), (rect1_width, rect1_height),
                  (0, 0, 255),
                  thickness=3,
                  lineType=8
                  , shift=0)
    rect1 = cv2.blur(rect1, blur_kernel_size, 0)


    image_new = cv2.addWeighted(overlay,0.5,rect2,1-.5,1)
    image_new = cv2.addWeighted(image_new, 0.5, rect1, 1 - .5, 1)

    cv2.namedWindow('Display window', cv2.WINDOW_AUTOSIZE)
   # cv2.imshow('Display window', numpy.hstack((newImage, image_new)))
    cv2.imshow('Display window', image_new)

    cv2.waitKey(0)
    cv2.destroyAllWindows()'''