#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;
int main()
{
    Mat img = imread("./1.jpg", IMREAD_COLOR);

    if (img.empty())
    {
        std:: cerr << "Erro load img" << std::endl;
        return -1;
    }
  
    cvtColor(img, img, COLOR_BGR2HSV);
    extractChannel(img, img, 2);
    resize(img, img,{ 500, 500 }, 0, 0, cv::INTER_NEAREST);
    // Шаг 1: Уменьшение шума с помощью гауссова размытия
    Mat blurred;
    GaussianBlur(img, blurred, Size(5, 5), 0);

    // Шаг 2: Расчет градиентов по горизонтали и вертикали с использованием операторов Собеля
    Mat sobelX, sobelY;
    Sobel(blurred, sobelX, CV_64F, 1, 0, 3);
    Sobel(blurred, sobelY, CV_64F, 0, 1, 3);

    // Шаг 3: Вычисление магнитуды и угла градиента
    Mat gradLen, gradientAngle;
    magnitude(sobelX, sobelY, gradLen);
    phase(sobelX, sobelY, gradientAngle);

    // Шаг 4: Подавление немаксимумов
    Mat nmsImage = Mat::zeros(gradLen.size(), gradLen.type());

    for (int i = 1; i < gradLen.rows - 1; i++) 
    {
        for (int j = 1; j < gradLen.cols - 1; j++) 
        {
            double angle = gradientAngle.at<double>(i, j) * 180 / 3.144456845;
            if (angle < 0) 
                angle += 180;

            double q = 0, r = 0;
            if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180)) 
            {
                q = gradLen.at<double>(i, j + 1);
                r = gradLen.at<double>(i, j - 1);
            }
            else if (22.5 <= angle && angle < 67.5) 
            {
                q = gradLen.at<double>(i + 1, j - 1);
                r = gradLen.at<double>(i - 1, j + 1);
            }
            else if (67.5 <= angle && angle < 112.5) 
            {
                q = gradLen.at<double>(i + 1, j);
                r = gradLen.at<double>(i - 1, j);
            }
            else if (112.5 <= angle && angle < 157.5) 
            {
                q = gradLen.at<double>(i - 1, j - 1);
                r = gradLen.at<double>(i + 1, j + 1);
            }

            if (gradLen.at<double>(i, j) >= q && gradLen.at<double>(i, j) >= r) 
            {
                nmsImage.at<double>(i, j) = gradLen.at<double>(i, j);
            }
        }
    }

    // Шаг 5: Пороговая фильтрация и подавление слабых пикселей
    Mat cannyImage = Mat::zeros(nmsImage.size(), nmsImage.type());

    double lowBorder = 1;  
    double highBorder = 130;  

    for (int i = 0; i < nmsImage.rows; i++) 
    {
        for (int j = 0; j < nmsImage.cols; j++) 
        {
            double pixelValue = nmsImage.at<double>(i, j);
            if (pixelValue >= highBorder) 
                cannyImage.at<double>(i, j) = 255;
            else if (pixelValue >= lowBorder) 
                for (int m = -1; m <= 1; m++) 
                {
                    for (int n = -1; n <= 1; n++) 
                        if (nmsImage.at<double>(i + m, j + n) >= highBorder) 
                        {
                            cannyImage.at<double>(i, j) = 255;
                            break;
                        }
                    if (cannyImage.at<double>(i, j) == 255) {
                        break;
                    }
                }
            
        }
    }

    // Отобразите результат
    imshow("Canny Edge Detection", cannyImage);
    waitKey(0);
    destroyAllWindows();
  
}

