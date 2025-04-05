#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// Классификация фигуры по количеству углов
string classifyShape(const vector<Point>& contour) {
    string shape = "Unknown";
    double peri = arcLength(contour, true);
    vector<Point> approx;
    approxPolyDP(contour, approx, 0.04 * peri, true);

    if (approx.size() == 3) shape = "Triangle";
    else if (approx.size() == 4) {
        Rect rect = boundingRect(approx);
        float ratio = (float)rect.width / rect.height;
        if (ratio >= 0.95 && ratio <= 1.05)
            shape = "Square";
        else
            shape = "Rectangle";
    }
    else if (approx.size() > 4 && approx.size() < 15)
        shape = "Polygon";
    else
        shape = "Circle";

    return shape;
}

int main() {
    string videoPath = "C:/Users/danil/Downloads/video.mp4";
    VideoCapture cap(videoPath);

    if (!cap.isOpened()) {
        cerr << "Ошибка открытия видео!" << endl;
        return -1;
    }

    int frameWidth = (int)cap.get(CAP_PROP_FRAME_WIDTH);
    int frameHeight = (int)cap.get(CAP_PROP_FRAME_HEIGHT);
    VideoWriter output("output_video.avi", VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, Size(frameWidth, frameHeight));

    Mat frame, gray, blurred, edged;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Обработка кадра
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, Size(5, 5), 1.5);
        Canny(blurred, edged, 50, 150);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(edged, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours) {
            double area = contourArea(contour);
            if (area < 100) continue;

            string shape = classifyShape(contour);
            Moments M = moments(contour);
            if (M.m00 == 0) continue;
            int cx = int(M.m10 / M.m00);
            int cy = int(M.m01 / M.m00);

            Scalar color;
            if (shape == "Triangle") color = Scalar(0, 255, 0);
            else if (shape == "Square") color = Scalar(255, 0, 0);
            else if (shape == "Rectangle") color = Scalar(0, 0, 255);
            else if (shape == "Circle") color = Scalar(0, 255, 255);
            else color = Scalar(255, 255, 255);

            drawContours(frame, vector<vector<Point>>{contour}, -1, color, 2);
            putText(frame, shape, Point(cx - 20, cy - 10), FONT_HERSHEY_SIMPLEX, 0.5, color, 2);
            circle(frame, Point(cx, cy), 3, Scalar(255, 255, 0), -1);
        }

        imshow("Object Tracking", frame);
        output.write(frame);

        if (waitKey(10) == 27) break;  // ESC для выхода
    }

    cap.release();
    output.release();
    destroyAllWindows();
    return 0;
}
