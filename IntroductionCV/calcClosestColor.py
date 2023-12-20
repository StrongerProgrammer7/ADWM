import cv2
import numpy
import numpy as np
def closest_color(c):
    arr_val = np.array(c)
    return np.linalg.norm(arr_val - center_pixel_color)
#Calculates the rate of difference
# expression evaluates "distance" between the color c and the color of the center pixel center_pixel_color.
# The smaller this distance, the closer the color of c is to the color of the center pixel.

if __name__ == '__main__':
    img = cv2.imread(r'/2.jpg')

    newImage = cv2.resize(img ,(640,480))

    height, width, _ = newImage.shape

    # Find the coordinates of the center pixel
    center_x = width // 2
    center_y = height // 2

    # Get the color of the center pixel
    center_pixel_color = newImage[center_y, center_x]
    print(center_pixel_color)

    #Identify the closest of the three colors (red, green, blue)
    color_red = (0, 0, 255)
    color_green = (0, 255, 0)
    color_blue = (255, 0, 0)
    colors = [color_red, color_green, color_blue]  # Red, green, blue
    closest_color = min(colors, key=closest_color)#min(colors, key=lambda c: np.linalg.norm(np.array(c) - center_pixel_color))

    cross_thickness = 5

    # Vertical line of the cross
    cv2.line(newImage, (center_x, center_y - 50), (center_x, center_y + 50), closest_color, cross_thickness)

    # horizontal line of the cross
    cv2.line(newImage, (center_x - 50, center_y), (center_x + 50, center_y), closest_color, cross_thickness)

    cv2.imshow('Result', newImage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


