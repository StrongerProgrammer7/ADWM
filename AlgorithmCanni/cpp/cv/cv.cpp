#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

using namespace cv;

double get_angleByDeg(double angle)
{
    return angle * 180.0 / 3.144456845;
}

bool isPixel_localMaxByDirectionGradient(double grad_curPixel,double* gradNeighbor)
{
    return grad_curPixel >= gradNeighbor[0] and grad_curPixel >= gradNeighbor[1];
}
   

double* get_intesitiesNeighbors(double right_neighbor, double left_neighbor,
    double up_neighbor, double down_neighbor,
    double left_up_neighbor, double left_down_neighbor,
    double right_up_neighbor, double right_down_neighbor, double angle)
{
    if ((0 <= angle && angle < 22.5) || (157.5 <= angle && angle <= 180))
        return new double[2]{ right_neighbor,left_neighbor };
    else if (22.5 <= angle && angle < 67.5)
        return new double[2]{ left_down_neighbor, right_up_neighbor };
    else if (67.5 <= angle && angle < 112.5)
        return new double[2]{ down_neighbor, up_neighbor };
    else if (112.5 <= angle && angle < 157.5)
        return new double[2] {left_up_neighbor, right_down_neighbor};
    return new double[2]{ 0,0 };
}

void non_maximal(Mat &nmsImage,const Mat &gradLen,const Mat &gradientAngle)
{
    for (int i = 1; i < gradLen.rows - 1; i++)
    {
        for (int j = 1; j < gradLen.cols - 1; j++)
        {
            double angle = get_angleByDeg(gradientAngle.at<double>(i, j));
            if (angle < 0)
                angle += 180;

            double* intensities_neighbors = get_intesitiesNeighbors(
                gradLen.at<double>(i, j + 1), gradLen.at<double>(i, j - 1),
                gradLen.at<double>(i - 1, j), gradLen.at<double>(i + 1, j),
                gradLen.at<double>(i - 1, j - 1), gradLen.at<double>(i + 1, j - 1),
                gradLen.at<double>(i - 1, j + 1), gradLen.at<double>(i + 1, j + 1),
                angle
            );

            if (isPixel_localMaxByDirectionGradient(gradLen.at<double>(i, j), intensities_neighbors) == true)
                nmsImage.at<double>(i, j) = gradLen.at<double>(i, j);
        }
    }
}

void findPixelUpHighBorder_set255(Mat &cannyImage,const Mat &nmsImage,const double &highBorder,const int &i,const int &j)
{
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

void double_thresh_filter(Mat &cannyImage, const Mat &nmsImage)
{
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
                findPixelUpHighBorder_set255(cannyImage, nmsImage, highBorder, i, j);

        }
    }
}
int main()
{
    Mat img = imread("./1.jpg", IMREAD_COLOR);

    if (img.empty())
    {
        std:: cerr << "Erro load img" << std::endl;
        return -1;
    }
  
    cvtColor(img, img, COLOR_BGR2HSV);
    /* Extracts the blue color channel from the img image and replaces the original img image with the extracted blue channel.*/
    extractChannel(img, img, 2);
    resize(img, img,{ 500, 500 }, 0, 0, cv::INTER_NEAREST);

    Mat blurred;
    GaussianBlur(img, blurred, Size(5, 5), 0.5);


    Mat sobelX, sobelY;
    Sobel(blurred, sobelX, CV_64F, 1, 0, 3);
    Sobel(blurred, sobelY, CV_64F, 0, 1, 3);


    Mat gradLen, gradientAngle;
    magnitude(sobelX, sobelY, gradLen);
    phase(sobelX, sobelY, gradientAngle);

    /* a new matrix (image) is created with dimensions and data type, all pixels =0 */
    Mat nmsImage = Mat::zeros(gradLen.size(), gradLen.type());
                                /*width and height of the matrix., data type storage format color and pixel precision*/

    non_maximal(nmsImage, gradLen, gradientAngle);

    Mat cannyImage = Mat::zeros(nmsImage.size(), nmsImage.type());

    double_thresh_filter(cannyImage, nmsImage);

    imshow("Canny Edge Detection", cannyImage);
    waitKey(0);
    destroyAllWindows();
  
}

