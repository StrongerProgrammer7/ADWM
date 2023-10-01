import cv2

matrix_G = [[]]
if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png', cv2.IMREAD_GRAYSCALE)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    cv2.imshow('GRAY', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

