#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;
//определяем заданные диапазоны цветов в HSV
// находим круговые объекты с помощью OpenCV.

            
// Обнаружение объектов круглой формы в красной цветовой гамме
int main() {
    // Загрузка изображения
    Mat inputImg = imread("C:/Users/danil/Downloads/5.png");

    if (inputImg.empty()) {
        cout << "Ошибка: изображение не загружено!" << endl;
        return -1;
    }

    // Изменение размера 
    resize(inputImg, inputImg, Size(1000, 700), INTER_LINEAR);

    // Преобразования 
    Mat imgHSV, imgGray;
    cvtColor(inputImg, imgHSV, COLOR_BGR2HSV);
    cvtColor(inputImg, imgGray, COLOR_BGR2GRAY);
    GaussianBlur(imgGray, imgGray, Size(9, 9), 2);

    // Создание масок для выделения красного цвета
    Mat redMaskLow, redMaskHigh, redMaskCombined, redFiltered;
    inRange(imgHSV, Scalar(0, 100, 100), Scalar(10, 255, 255), redMaskLow);      // Низкие тона красного
    inRange(imgHSV, Scalar(160, 100, 100), Scalar(180, 255, 255), redMaskHigh);  // Высокие тона красного

    bitwise_or(redMaskLow, redMaskHigh, redMaskCombined); // Объединение масок
    bitwise_and(inputImg, inputImg, redFiltered, redMaskCombined); // Применение маски к изображению

    // Поиск окружностей методом Хафа
    vector<Vec3f> detectedCircles;
    HoughCircles(imgGray, detectedCircles, HOUGH_GRADIENT, 1, imgGray.rows, 55, 45, 200, 280);

    // Отображение найденных окружностей на исходном изображении
    for (size_t i = 0; i < detectedCircles.size(); i++) {
        Vec3i circ = detectedCircles[i];
        Point circleCenter(circ[0], circ[1]);
        int circleRadius = circ[2];

        // Отметка центра круга
        circle(inputImg, circleCenter, 3, Scalar(0, 255, 0), -1, LINE_AA);  // Зелёный центр
        // Отрисовка окружности
        circle(inputImg, circleCenter, circleRadius, Scalar(255, 0, 0), 2, LINE_AA);  // Синяя линия
    }

    // Окна отображения
    imshow("Изображение с окружностями", inputImg);
    imshow("Маска красного цвета", redMaskCombined);
    imshow("Фрагмент с красными объектами", redFiltered);

    waitKey(0);
    destroyAllWindows();

    return 0;
}
