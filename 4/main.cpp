#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    std::string image_path = "C:/Users/danil/Downloads/2.png";
    Mat image = imread(image_path);

    if (image.empty())
    {
        cout << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);

    // --- Маска для красного цвета ---
    Mat red_mask, red_result;
    inRange(hsv, Scalar(0, 100, 100), Scalar(10, 255, 255), red_mask); // Низкий красный
    Mat red_mask2;
    inRange(hsv, Scalar(160, 100, 100), Scalar(179, 255, 255), red_mask2); // Верхний красный
    red_mask |= red_mask2;
    bitwise_and(image, image, red_result, red_mask);
    imshow("Красная маска", red_mask);
    imshow("Красные участки", red_result);

    // --- Маска для зелёного цвета ---
    Mat green_mask, green_result;
    inRange(hsv, Scalar(35, 100, 100), Scalar(85, 255, 255), green_mask);
    bitwise_and(image, image, green_result, green_mask);
    imshow("Зеленая маска", green_mask);
    imshow("Зеленые участки", green_result);

    // --- Маска для синего цвета ---
    Mat blue_mask, blue_result;
    inRange(hsv, Scalar(90, 100, 100), Scalar(130, 255, 255), blue_mask);
    bitwise_and(image, image, blue_result, blue_mask);
    imshow("Синяя маска", blue_mask);
    imshow("Синие участки", blue_result);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
