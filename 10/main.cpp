#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // ==============================================
    // 1. Загрузка каскадов Хаара для обнаружения объектов
    // ==============================================
    cv::CascadeClassifier face_cascade, eyes_cascade, smile_cascade;

    // Пытаемся загрузить XML-файлы с каскадами
    if (!face_cascade.load("haarcascade_frontalface_default.xml") ||
        !eyes_cascade.load("haarcascade_eye.xml") ||
        !smile_cascade.load("haarcascade_smile.xml")) {
        std::cerr << "Ошибка: Не удалось загрузить каскады Хаара!" << std::endl;
        return -1;
    }

    // ==============================================
    // 2. Открытие видеофайла для обработки
    // ==============================================
    cv::VideoCapture cap("ZUA.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Ошибка: Не удалось открыть видеофайл!" << std::endl;
        return -1;
    }

    cv::Mat frame; // Кадр из видео
    while (cap.read(frame)) { // Чтение кадров пока видео не закончится
        if (frame.empty()) break; // Если кадр пустой - выходим

        // ==============================================
        // 3. Предварительная обработка изображения
        // ==============================================
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY); // Конвертация в оттенки серого
        cv::equalizeHist(gray, gray); // Выравнивание гистограммы для улучшения контраста
        cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0); // Размытие для уменьшения шума

        // ==============================================
        // 4. Обнаружение лиц на изображении
        // ==============================================
        std::vector<cv::Rect> faces;
        // Параметры detectMultiScale:
        // 1.1 - масштабный коэффициент
        // 5 - минимальное количество соседей
        // 0 - флаги (0 - по умолчанию)
        // cv::Size(150,150) - минимальный размер объекта
        face_cascade.detectMultiScale(gray, faces, 1.1, 5, 0, cv::Size(150, 150));

        // ==============================================
        // 5. Обработка каждого обнаруженного лица
        // ==============================================
        for (const auto& face : faces) {
            // Рисуем прямоугольник вокруг лица (синий цвет)
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);

            // Область интереса (ROI) - только лицо
            cv::Mat faceROI_gray = gray(face);
            cv::Mat faceROI_color = frame(face);

            // 5.1. Обнаружение глаз в области лица
            std::vector<cv::Rect> eyes;
            eyes_cascade.detectMultiScale(faceROI_gray, eyes, 1.1, 10, 0, cv::Size(50, 50));
            for (const auto& eye : eyes) {
                // Рисуем прямоугольник вокруг глаз (зеленый цвет)
                cv::rectangle(faceROI_color, eye, cv::Scalar(0, 255, 0), 2);
            }

            // 5.2. Обнаружение улыбки в области лица
            std::vector<cv::Rect> smiles;
            smile_cascade.detectMultiScale(faceROI_gray, smiles, 1.24, 15, 0, cv::Size(40, 40));
            for (const auto& smile : smiles) {
                // Рисуем прямоугольник вокруг улыбки (красный цвет)
                cv::rectangle(faceROI_color, smile, cv::Scalar(0, 0, 255), 2);
            }
        }

        // ==============================================
        // 6. Отображение результата
        // ==============================================
        cv::imshow("Детекция лиц, глаз и улыбок", frame);

        // Выход по нажатию любой клавиши (ожидание 30 мс между кадрами)
        if (cv::waitKey(30) >= 0) break;
    }

    // ==============================================
    // 7. Освобождение ресурсов
    // ==============================================
    cap.release(); // Закрытие видеофайла
    cv::destroyAllWindows(); // Закрытие всех окон OpenCV

    return 0;
}