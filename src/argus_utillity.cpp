#include <argus_utillity.h>

#include <chrono>
#include <ctime>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace argus {
nlohmann::json load_json(const std::string &file_path) {
  std::ifstream i(file_path);
  nlohmann::json j;
  i >> j;
  i.close();
  return j;
}

void write_json(const std::string &file_path, const nlohmann::ordered_json &j) {
  std::ofstream o(file_path);
  o << std::setw(4) << j;
  o.close();
}

std::string date() {
  std::time_t t = std::time(nullptr);
  char mbstr[100];
  if (std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S",
                    std::localtime(&t))) {
    return std::string(mbstr);
  }
  return std::string();
}

cv::Mat draw_observation(
    const cv::Mat &img,
    const Eigen::aligned_map<size_t, Eigen::Vector2f> &observation) {
  cv::Mat temp;
  cv::cvtColor(img, temp, cv::COLOR_GRAY2BGR);
  std::uniform_int_distribution<int> dis(1, 255);
  for (const auto &ob : observation) {
    std::mt19937 gen(ob.first);
    cv::Scalar color(dis(gen), dis(gen), dis(gen));
    cv::Point2f pt(ob.second.x(), ob.second.y());
    cv::circle(temp, pt, 3, color, -1);
    cv::putText(temp, std::to_string(ob.first), pt, 1, 1, color);
  }
  return temp;
}

} // namespace argus