#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main()
{
    std::string image_path = "C:/Users/danil/Downloads/3.png";
    Mat image = imread(image_path);

    if (image.empty())
    {
        cout << "Не удалось загрузить изображение!" << endl;
        return -1;
    }

    // Показываем оригинал
    imshow("Оригинал", image);

    // --- Цветовые пространства ---
    Mat gray, hsv, lab;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    cvtColor(image, hsv, COLOR_BGR2HSV);
    cvtColor(image, lab, COLOR_BGR2Lab);

    imshow("Grayscale", gray);
    imshow("HSV", hsv);
    imshow("LAB", lab);

    // --- Обнаружение кругов ---
    Mat circlesOutput = image.clone();
    GaussianBlur(gray, gray, Size(9, 9), 2, 2);  // Смягчение
    vector<Vec3f> circles;
    HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows / 8, 100, 30, 0, 0);

    for (size_t i = 0; i < circles.size(); i++)
    {
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
        int radius = cvRound(circles[i][2]);
        circle(circlesOutput, center, radius, Scalar(0, 255, 0), 2);       // круг
        circle(circlesOutput, center, 2, Scalar(0, 0, 255), 3);            // центр
    }

    // --- Обнаружение линий ---
    Mat edges;
    Canny(gray, edges, 50, 150, 3);
    vector<Vec2f> lines;
    HoughLines(edges, lines, 1, CV_PI / 180, 100);

    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        Point pt1, pt2;
        double a = cos(theta), b = sin(theta);
        double x0 = a * rho, y0 = b * rho;
        pt1.x = cvRound(x0 + 1000 * (-b));
        pt1.y = cvRound(y0 + 1000 * (a));
        pt2.x = cvRound(x0 - 1000 * (-b));
        pt2.y = cvRound(y0 - 1000 * (a));
        line(circlesOutput, pt1, pt2, Scalar(255, 0, 0), 1);
    }

    imshow("Обнаружение кругов и линий", circlesOutput);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
