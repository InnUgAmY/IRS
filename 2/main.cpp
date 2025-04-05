#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = "C:/Users/danil/Downloads/1.png"; // укажи свой путь
    Mat image = imread(image_path);

    if (image.empty())
    {
        std::cout << "Не удалось загрузить изображение!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // Разделим на 4 части
    Rect topLeft(0, 0, width / 2, height / 2);
    Rect topRight(width / 2, 0, width / 2, height / 2);
    Rect bottomLeft(0, height / 2, width / 2, height / 2);
    Rect bottomRight(width / 2, height / 2, width / 2, height / 2);

    Mat tl = image(topLeft).clone();          // Без изменений
    Mat tr = image(topRight).clone();         // Инверсия цветов
    Mat bl = image(bottomLeft).clone();       // Ч/б
    Mat br = image(bottomRight).clone();      // Заливка цветом

    // Применяем эффекты
    bitwise_not(tr, tr);                      // Инверсия

    cvtColor(bl, bl, COLOR_BGR2GRAY);         // Ч/б
    cvtColor(bl, bl, COLOR_GRAY2BGR);         // Обратно в 3 канала

    br = Scalar(0, 255, 0);                   // Зелёный цвет

    // Склеиваем части обратно
    Mat topHalf, bottomHalf, finalImage;
    hconcat(tl, tr, topHalf);
    hconcat(bl, br, bottomHalf);
    vconcat(topHalf, bottomHalf, finalImage);

    // Отображение
    imshow("Обработанное изображение", finalImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
