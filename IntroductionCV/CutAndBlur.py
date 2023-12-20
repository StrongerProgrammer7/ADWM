import cv2

if __name__ == '__main__':
    img = cv2.imread(r'2.jpg')
    # cv2.namedWindow('Display window', cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO | cv2.WINDOW_GUI_EXPANDED)
    img = cv2.resize(img ,(640,480))
    height, width, _ = img.shape

    #Draw rectangle horizontal
    start_x = width // 2 - 100
    start_y = height // 2 - 40
    end_x = width - 200
    end_y = start_y + 30

    rectHoriz_startPoint = start_x, start_y
    rectHoriz_endPoint = end_x, end_y
    color = (255, 0, 0)
    thickness = 4

    #Draw rectangle vertical
    start_x = start_x + (start_x // 2) - 18
    start_y = 50
    end_x = start_x + 30
    end_y = height - 100

    rectVertical_startPoint = start_x, start_y
    rectVertical_endPoint = end_x, end_y

    cat_horizontal = img[rectHoriz_startPoint[1]:rectHoriz_endPoint[1], rectHoriz_startPoint[0]:rectHoriz_endPoint[0]]

    cat_vertical = img[rectVertical_startPoint[1]:rectVertical_endPoint[1], rectVertical_startPoint[0]:rectVertical_endPoint[0]]

    cat_horizontal = cv2.blur(cat_horizontal, (10, 10))
    cat_vertical = cv2.blur(cat_vertical, (10, 10))

    cv2.rectangle(img, rectVertical_startPoint, rectVertical_endPoint, color, thickness)
    cv2.rectangle(img, rectHoriz_startPoint, rectHoriz_endPoint, color, thickness)

    img[rectHoriz_startPoint[1]:rectHoriz_endPoint[1], rectHoriz_startPoint[0]:rectHoriz_endPoint[0]] = cat_horizontal
    img[rectVertical_startPoint[1]:rectVertical_endPoint[1], rectVertical_startPoint[0]:rectVertical_endPoint[0]] = cat_vertical

    cv2.imshow('Display window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
