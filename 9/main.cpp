#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <numeric>  // для accumulate

namespace fs = std::filesystem;

struct CardSample {
    std::string name;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

void filterMatches(const std::vector<std::vector<cv::DMatch>>& matches,
    float ratio_threshold,
    std::vector<cv::DMatch>& good_matches) {
    for (const auto& match_pair : matches) {
        if (match_pair.size() < 2) continue;
        if (match_pair[0].distance < ratio_threshold * match_pair[1].distance) {
            good_matches.push_back(match_pair[0]);
        }
    }
}

int main() {
    try {
        const std::string base_path = "C:/Users/danil/Downloads";
        const std::vector<std::string> sample_files = {
            "valet piki.png", "tuz buby.png", "korol kresti.png",
            "dama chervy.png", "9 kresti.png", "9 buby.png",
            "5 kresti.png", "3 piki.png", "2 chervy.png"
        };
        const std::string target_path = base_path + "/target.png";

        if (!fs::exists(target_path)) {
            throw std::runtime_error("Целевое изображение не найдено: " + target_path);
        }

        cv::Ptr<cv::SIFT> detector = cv::SIFT::create();
        if (detector.empty()) {
            throw std::runtime_error("Не удалось инициализировать SIFT детектор");
        }

        std::vector<CardSample> samples;
        for (const auto& filename : sample_files) {
            const std::string full_path = base_path + "/" + filename;

            if (!fs::exists(full_path)) {
                std::cerr << "Предупреждение: файл " << full_path << " не найден, пропускаем..." << std::endl;
                continue;
            }

            cv::Mat img = cv::imread(full_path, cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                std::cerr << "Ошибка загрузки: " << full_path << std::endl;
                continue;
            }

            CardSample sample;
            sample.name = filename.substr(0, filename.find_last_of('.'));
            sample.image = img.clone();  // сохраняем для размеров
            detector->detectAndCompute(img, cv::noArray(), sample.keypoints, sample.descriptors);
            samples.push_back(std::move(sample));
        }

        // Загружаем цветную копию для отображения и ч/б для SIFT
        cv::Mat target_img_color = cv::imread(target_path, cv::IMREAD_COLOR);
        cv::Mat target_img_gray = cv::imread(target_path, cv::IMREAD_GRAYSCALE);
        if (target_img_color.empty() || target_img_gray.empty()) {
            throw std::runtime_error("Не удалось загрузить целевое изображение");
        }

        // SIFT обработка
        std::vector<cv::KeyPoint> target_keypoints;
        cv::Mat target_descriptors;
        detector->detectAndCompute(target_img_gray, cv::noArray(), target_keypoints, target_descriptors);

        cv::Mat result_img = target_img_color.clone();

        cv::BFMatcher matcher(cv::NORM_L2);

        for (const auto& sample : samples) {
            if (sample.descriptors.empty()) continue;

            std::vector<std::vector<cv::DMatch>> knn_matches;
            matcher.knnMatch(sample.descriptors, target_descriptors, knn_matches, 2);

            std::vector<cv::DMatch> good_matches;
            filterMatches(knn_matches, 0.75f, good_matches);

            if (good_matches.size() >= 4) {
                std::vector<cv::Point2f> src_points, dst_points;
                for (const auto& match : good_matches) {
                    src_points.push_back(sample.keypoints[match.queryIdx].pt);
                    dst_points.push_back(target_keypoints[match.trainIdx].pt);
                }

                cv::Mat H = cv::findHomography(src_points, dst_points, cv::RANSAC);
                if (H.empty()) continue; // если гомография не найдена, пропускаем

                std::vector<cv::Point2f> corners = {
                    cv::Point2f(0, 0),
                    cv::Point2f((float)sample.image.cols, 0),
                    cv::Point2f((float)sample.image.cols, (float)sample.image.rows),
                    cv::Point2f(0, (float)sample.image.rows)
                };
                std::vector<cv::Point2f> transformed_corners;
                cv::perspectiveTransform(corners, transformed_corners, H);

                // Рисуем контур карты
                for (size_t i = 0; i < 4; ++i) {
                    cv::line(result_img, transformed_corners[i],
                        transformed_corners[(i + 1) % 4],
                        cv::Scalar(0, 255, 0), 2); // зеленый контур
                }

                // Вычисляем центр для текста
                cv::Point2f sum = std::accumulate(
                    transformed_corners.begin(), transformed_corners.end(),
                    cv::Point2f(0.0f, 0.0f),
                    [](const cv::Point2f& a, const cv::Point2f& b) { return a + b; }
                );
                cv::Point2f center = sum * (1.0f / transformed_corners.size());

                // Рисуем название жирным зеленым текстом
                cv::putText(result_img, sample.name, center,
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Распознанные карты", result_img);
        cv::waitKey(0);

    }
    catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
