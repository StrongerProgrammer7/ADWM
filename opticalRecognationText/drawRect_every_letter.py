import pytesseract as pt
import cv2
from pytesseract import Output
#import enchant

#dictionary = enchant.Dict('en_US')
path_to_tesseract = r"../tesseract.exe"

pt.pytesseract.tesseract_cmd = (path_to_tesseract)

if __name__ == '__main__':
    img = cv2.imread("../imgs/captcha/2.png")
    cv2.imshow('Original',img)

    h,w,c = img.shape
    boxes = pt.image_to_boxes(img)
    for b in boxes.splitlines():
        b = b.split(' ')
        img = cv2.rectangle(img,(int(b[1]),h-int(b[2])),(int(b[3]),h-int(b[4])),(0,255,0),1)
    cv2.imshow('BOxes',img)
    text = pt.image_to_string(img,lang='rus+eng')
    print(text)
    cv2.waitKey(0)