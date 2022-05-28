#pragma once

#include <fstream>
#include <iostream>
#include <variant>
#include <vector>

#include <json.hpp>
#include <opencv2/calib3d.hpp>
#include <sophus/se3.hpp>
#include <basalt/utils/eigen_utils.hpp>
#include <pangolin/gl/gldraw.h>

namespace argus {
nlohmann::json load_json(const std::string &file_path);
void write_json(const std::string &file_path, const nlohmann::ordered_json &j);
std::string date();

template <typename Scalar>
Sophus::SE3<Scalar> rtvec2SE3(const cv::Vec<Scalar, 3> rvec,
                              const cv::Vec<Scalar, 3> tvec) {
  Eigen::Matrix<Scalar, 3, 1> r(rvec[0], rvec[1], rvec[2]);
  Eigen::Matrix<Scalar, 3, 1> t(tvec[0], tvec[1], tvec[2]);
  return Sophus::SE3<Scalar>(Sophus::SO3<Scalar>::exp(r), t);
}

cv::Mat draw_observation(
    const cv::Mat &img,
    const Eigen::aligned_map<size_t, Eigen::Vector2f> &observation);

std::shared_ptr<cv::Mat> draw_obs_for_pango(
    const std::vector<cv::Mat> &imgs,
    const std::vector<Eigen::aligned_map<size_t, Eigen::Vector2f>>
        &current_obs);

template <typename Model>
std::vector<Model> json2models(const std::string &filename) {
  const auto j = load_json(filename);

  const uint16_t cam_num = j["intrinsics"].size();

  std::vector<Model> camera_models(cam_num);
  for (int cam = 0; cam < cam_num; ++cam) {
    std::unordered_map<std::string, double> params =
        j["intrinsics"][cam]["intrinsics"];
    Eigen::Vector2i img_col_row(j["resolution"][cam][0].get<int>(),
                                j["resolution"][cam][1].get<int>());
    typename Model::VecN parameters;
    const auto param_name_list = Model::param_name_list();
    for (int i = 0; i < Model::N; ++i) {
      parameters[i] = params.at(param_name_list[i]);
    }
    camera_models[cam] = Model(parameters, img_col_row);
  }

  return camera_models;
}

template <typename Scalar>
std::vector<Sophus::SE3<Scalar>> json2extrinsics(const std::string &filename) {
  const auto j = load_json(filename);

  const uint16_t cam_num = j["T_cami_cam0"].size();

  std::array<std::string, 7> se3_name = {"qx", "qy", "qz", "qw",
                                         "px", "py", "pz"};

  std::vector<Sophus::SE3<Scalar>> extrinsics(cam_num);
  for (int cam = 0; cam < cam_num; ++cam) {
    std::vector<Scalar> ext_vec(7);
    for (int i = 0; i < 7; ++i) {
      ext_vec[i] = j["T_cami_cam0"][cam][se3_name[i]].get<Scalar>();
    }

    auto ext = Eigen::Map<Sophus::SE3<Scalar>>(ext_vec.data());
    extrinsics[cam] = Sophus::SE3d(ext);
  }

  return extrinsics;
}

inline void render_camera(const Eigen::Matrix4d &T_w_c, float lineWidth,
                          const uint8_t *color, float sizeFactor = 1.0f) {
  const float sz = sizeFactor;
  const float width = 640, height = 480, fx = 500, fy = 500, cx = 320, cy = 240;

  const Eigen::aligned_vector<Eigen::Vector3f> lines = {
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {0, 0, 0},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (height - 1 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (0 - cx) / fx, sz * (0 - cy) / fy, sz},
      {sz * (width - 1 - cx) / fx, sz * (0 - cy) / fy, sz}};

  glPushMatrix();
  glMultMatrixd(T_w_c.data());
  glColor3ubv(color);
  glLineWidth(lineWidth);
  pangolin::glDrawLines(lines);
  glPopMatrix();
}

} // namespace argus