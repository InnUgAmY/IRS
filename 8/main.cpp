#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    Mat img1 = imread("C:/Users/danil/Downloads/10.png"); // Правая
    Mat img2 = imread("C:/Users/danil/Downloads/11.png"); // Левая

    if (img1.empty() || img2.empty()) {
        cerr << "Ошибка загрузки изображений!" << endl;
        return -1;
    }

    // ORB детектор
    Ptr<ORB> orb = ORB::create(2000);
    vector<KeyPoint> kp1, kp2;
    Mat desc1, desc2;

    orb->detectAndCompute(img1, noArray(), kp1, desc1);
    orb->detectAndCompute(img2, noArray(), kp2, desc2);

    // Матчер
    BFMatcher matcher(NORM_HAMMING);
    vector<DMatch> matches;
    matcher.match(desc1, desc2, matches);

    // Фильтрация
    double maxDist = 0;
    for (auto& m : matches)
        maxDist = max(maxDist, (double)m.distance);

    vector<DMatch> goodMatches;
    for (auto& m : matches)
        if (m.distance < 0.5 * maxDist)
            goodMatches.push_back(m);

    // Точки
    vector<Point2f> pts1, pts2;
    for (auto& m : goodMatches) {
        pts1.push_back(kp1[m.queryIdx].pt);  // img1
        pts2.push_back(kp2[m.trainIdx].pt);  // img2
    }

    // Гомография
    Mat H = findHomography(pts1, pts2, RANSAC);
    if (H.empty()) {
        cerr << "Не удалось найти гомографию!" << endl;
        return -1;
    }

    // Определяем размер выходного изображения
    vector<Point2f> corners_img1 = { {0, 0}, {(float)img1.cols, 0}, {(float)img1.cols, (float)img1.rows}, {0, (float)img1.rows} };
    vector<Point2f> transformed_corners;
    perspectiveTransform(corners_img1, transformed_corners, H);

    // Находим смещение, если часть уходит в отрицательные координаты
    float min_x = 0, min_y = 0, max_x = img2.cols, max_y = img2.rows;
    for (const auto& pt : transformed_corners) {
        min_x = min(min_x, pt.x);
        min_y = min(min_y, pt.y);
        max_x = max(max_x, pt.x);
        max_y = max(max_y, pt.y);
    }

    // Смещение и размер результата
    int shift_x = -min_x;
    int shift_y = -min_y;
    Size panorama_size((int)(max_x - min_x), (int)(max_y - min_y));

    // Смещенная гомография
    Mat H_translation = (Mat_<double>(3, 3) <<
        1, 0, shift_x,
        0, 1, shift_y,
        0, 0, 1);

    Mat panorama;
    warpPerspective(img1, panorama, H_translation * H, panorama_size);

    // Вставляем левое изображение
    Mat roi(panorama, Rect(shift_x, shift_y, img2.cols, img2.rows));
    img2.copyTo(roi);

    // Отображение
    imshow("Панорама", panorama);
    imwrite("panorama_fixed.png", panorama);

    // Совпадения
    Mat match_img;
    drawMatches(img1, kp1, img2, kp2, goodMatches, match_img);
    imshow("Совпадения", match_img);

    // Оригиналы рядом (img1 справа, img2 слева)
    Mat combined(max(img1.rows, img2.rows), img1.cols + img2.cols, img1.type(), Scalar::all(0));
    img2.copyTo(combined(Rect(0, 0, img2.cols, img2.rows)));
    img1.copyTo(combined(Rect(img2.cols, 0, img1.cols, img1.rows)));
    imshow("Исходные изображения", combined);

    waitKey(0);
    return 0;
}
