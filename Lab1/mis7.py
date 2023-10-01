import cv2
import numpy
import numpy as np
def closest_color(c):
    arr_val = np.array(c)
    return np.linalg.norm(arr_val - center_pixel_color)
#Вычисляет норму разности
#выражение оценивает "расстояние" между цветом c и цветом центрального пикселя center_pixel_color.
# Чем меньше это расстояние, тем ближе цвет c к цвету центрального пикселя.

if __name__ == '__main__':
    image = cv2.imread(r'../imgs/2.jpg')

    scale_percent = 8  # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)

    newImage = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    height, width, _ = newImage.shape

    # Найти координаты центрального пикселя
    center_x = width // 2
    center_y = height // 2

    # Получить цвет центрального пикселя
    center_pixel_color = newImage[center_y, center_x]
    print(center_pixel_color)
    # Определить ближайший из трех цветов (красный, зеленый, синий)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_blue = (255, 0, 0)
    colors = [color_red, color_green, color_blue]  # Красный, зеленый, синий
    closest_color = min(colors, key=closest_color)#min(colors, key=lambda c: np.linalg.norm(np.array(c) - center_pixel_color))

    cross_thickness = 5  # Толщина креста

    # Вертикальная линия креста
    cv2.line(newImage, (center_x, center_y - 50), (center_x, center_y + 50), closest_color, cross_thickness)

    # Горизонтальная линия креста
    cv2.line(newImage, (center_x - 50, center_y), (center_x + 50, center_y), closest_color, cross_thickness)


    cv2.imshow('Result', newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


