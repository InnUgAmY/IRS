#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main()
{
    std::string image_path = "C:/Users/danil/Downloads/1.png"; // ����� ���� ����
    Mat image = imread(image_path);

    if (image.empty())
    {
        std::cout << "�� ������� ��������� �����������!" << std::endl;
        return -1;
    }

    int width = image.cols;
    int height = image.rows;

    // �������� �� 4 �����
    Rect topLeft(0, 0, width / 2, height / 2);
    Rect topRight(width / 2, 0, width / 2, height / 2);
    Rect bottomLeft(0, height / 2, width / 2, height / 2);
    Rect bottomRight(width / 2, height / 2, width / 2, height / 2);

    Mat tl = image(topLeft).clone();          // ��� ���������
    Mat tr = image(topRight).clone();         // �������� ������
    Mat bl = image(bottomLeft).clone();       // �/�
    Mat br = image(bottomRight).clone();      // ������� ������

    // ��������� �������
    bitwise_not(tr, tr);                      // ��������

    cvtColor(bl, bl, COLOR_BGR2GRAY);         // �/�
    cvtColor(bl, bl, COLOR_GRAY2BGR);         // ������� � 3 ������

    br = Scalar(0, 255, 0);                   // ������ ����

    // ��������� ����� �������
    Mat topHalf, bottomHalf, finalImage;
    hconcat(tl, tr, topHalf);
    hconcat(bl, br, bottomHalf);
    vconcat(topHalf, bottomHalf, finalImage);

    // �����������
    imshow("������������ �����������", finalImage);
    waitKey(0);
    destroyAllWindows();

    return 0;
}
