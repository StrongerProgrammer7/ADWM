import math

import numpy as np
import cv2

if __name__ == '__main__':
    img = cv2.imread(r'../imgs/1.png', cv2.IMREAD_GRAYSCALE)

    scale_percent = 50  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


    # Размер матрицы свертки (нечетное число, чтобы был центр)
    kernel_size = 3

    # Среднеквадратичное отклонение (стандартное отклонение) для Гауссовой функции
    sigma = 1.0

    # Вычисление центра матрицы свертки
    center = (kernel_size - 1) // 2

    # Создание пустой матрицы свертки
    kernel = np.zeros((kernel_size, kernel_size))

    # Заполнение матрицы свертки значениями функции Гаусса
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            # Функция Гаусса для текущей координаты (x, y)
            kernel[i, j] = np.exp(-((x**2-math.ceil(kernel_size/2)) + (y**2-math.ceil(kernel_size/2))) / (2 * sigma**2)) / (2 * np.pi * sigma**2)

    # Нормализация матрицы свертки, чтобы сумма всех значений была равна 1
    kernel /= np.sum(kernel)

    # Вывод матрицы свертки
    print("Гауссова матрица свертки:")
    print(kernel)
    image_copy = img.copy()
    # Примените свертку к изображению с использованием функции filter2D
    result_image = cv2.filter2D(image_copy, -1, kernel)

    # Вычислите новое значение насыщенности для каждого пикселя внутри изображения
    # и запишите это значение в пиксель нового изображения
    for i in range(1, result_image.shape[0] - 1):
        for j in range(1, result_image.shape[1] - 1):
            # Примените формулу для расчета новой насыщенности
            new_saturation = np.sum(result_image[i - 1:i + 2, j - 1:j + 2] * kernel)

            # Запишите новое значение насыщенности в пиксель нового изображения
            result_image[i, j] = np.clip(new_saturation, 0, 255)  # Ограничьте значение в диапазоне 0-255

    # Отобразите исходное и результирующее изображения
    cv2.imshow('Original Image', img)
    cv2.imshow('Result Image', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
