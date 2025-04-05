#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cmath>
#include <map>

using namespace cv;
using namespace std;

double angle(Point pt1, Point pt2, Point pt0) {
    double dx1 = pt1.x - pt0.x;
    double dy1 = pt1.y - pt0.y;
    double dx2 = pt2.x - pt0.x;
    double dy2 = pt2.y - pt0.y;
    return (dx1 * dx2 + dy1 * dy2) /
        sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2));
}

string classifyShape(const vector<Point>& approx) {
    int vertices = static_cast<int>(approx.size());

    if (vertices == 3) return "Triangle";
    else if (vertices == 4) {
        double cos1 = abs(angle(approx[0], approx[1], approx[2]));
        double cos2 = abs(angle(approx[1], approx[2], approx[3]));
        if (cos1 < 0.3 && cos2 < 0.3) {
            double side1 = norm(approx[0] - approx[1]);
            double side2 = norm(approx[1] - approx[2]);
            if (abs(side1 - side2) < 10)
                return "Square";
            else
                return "Rectangle";
        }
        return "Quadrilateral";
    }
    else if (vertices > 4 && vertices <= 7) return "Polygon";
    else return "Circle";
}

int main() {
    // Загрузка изображения
    Mat image = imread("C:/Users/danil/Downloads/6.png");
    if (image.empty()) {
        cout << "Ошибка загрузки изображения!" << endl;
        return -1;
    }

    // Повышение насыщенности
    Mat hsv;
    cvtColor(image, hsv, COLOR_BGR2HSV);
    for (int i = 0; i < hsv.rows; i++) {
        for (int j = 0; j < hsv.cols; j++) {
            hsv.at<Vec3b>(i, j)[1] = saturate_cast<uchar>(hsv.at<Vec3b>(i, j)[1] * 1.5);
        }
    }
    Mat enhanced;
    cvtColor(hsv, enhanced, COLOR_HSV2BGR);

    // Градации серого, блюр, Canny
    Mat gray, blurred, edged;
    cvtColor(enhanced, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(5, 5), 2);
    Canny(blurred, edged, 50, 150);

    // Поиск контуров
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Цвета для разных фигур
    map<string, Scalar> colorMap = {
        {"Triangle", Scalar(0, 0, 255)},
        {"Square", Scalar(255, 0, 0)},
        {"Rectangle", Scalar(0, 255, 0)},
        {"Circle", Scalar(0, 255, 255)},
        {"Polygon", Scalar(255, 0, 255)},
        {"Quadrilateral", Scalar(255, 255, 0)}
    };

    // Классификация и отрисовка
    for (size_t i = 0; i < contours.size(); ++i) {
        vector<Point> approx;
        approxPolyDP(contours[i], approx, arcLength(contours[i], true) * 0.02, true);

        if (fabs(contourArea(approx)) < 100 || !isContourConvex(approx))
            continue;

        string shape = classifyShape(approx);
        Rect r = boundingRect(approx);
        Scalar color = colorMap.count(shape) ? colorMap[shape] : Scalar(255, 255, 255);

        // Обводка и подпись
        drawContours(image, vector<vector<Point>>{approx}, -1, color, 2);
        putText(image, shape, Point(r.x, r.y - 5), FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    // Отображение результатов
    imshow("Original Image with Classified Shapes", image);
    imshow("Canny Edge Detection", edged);

    waitKey(0);
    destroyAllWindows();
    return 0;
}
