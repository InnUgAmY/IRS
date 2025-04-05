#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>

using namespace cv;

int main()
{
    std::system("chcp 65001 > nul");

    std::string image_path = "C:/Users/danil/Downloads/1.png";
    Mat image = imread(image_path, IMREAD_COLOR);

    if (image.empty())
    {
        std::cout << "Ошибка при загрузке изображения!" << std::endl;
        return -1;
    }

    int thickness = 2;
    double fontScale = 1.0;
    Scalar color(20, 255, 20); // Зеленый цвет

    // 1. Пунктирная линия
    for (int i = 0; i < 200; i += 20)
    {
        line(image, Point(20 + i, 20), Point(30 + i, 70), color, thickness);
    }

    // 2. Закрашенный прямоугольник
    rectangle(image, Point(50, 100), Point(200, 200), color, FILLED);

    // 3. Цветок из маленьких кругов
    Point center(400, 150);
    int radius = 10;
    for (int i = 0; i < 360; i += 45)
    {
        int x = center.x + 50 * cos(i * CV_PI / 180);
        int y = center.y + 50 * sin(i * CV_PI / 180);
        circle(image, Point(x, y), radius, color, FILLED);
    }
    circle(image, center, radius, Scalar(0, 0, 255), FILLED); // центр красным

    // 4. Дуга
    ellipse(image, Point(200, 300), Size(100, 50), 0, 0, 180, color, thickness);

    // 5. Контур звезды
    std::vector<Point> star = {
        {300, 400}, {320, 440}, {370, 440}, {330, 470},
        {350, 520}, {300, 490}, {250, 520}, {270, 470},
        {230, 440}, {280, 440}
    };
    polylines(image, star, true, color, thickness);

    // 6. Текст
    putText(image, "TEXT IS A TEXT!", Point(100, 450), FONT_HERSHEY_SIMPLEX, fontScale, color, thickness);

    imshow("Рисование с OpenCV", image);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
