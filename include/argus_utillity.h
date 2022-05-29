#pragma once

#include <fstream>
#include <vector>

#include <json.hpp>
#include <opencv2/core.hpp>
#include <basalt/camera/generic_camera.hpp>
#include <basalt/utils/eigen_utils.hpp>
#include <pangolin/gl/gldraw.h>

namespace argus {
nlohmann::json load_json(const std::string &file_path);
void write_json(const std::string &file_path, const nlohmann::ordered_json &j);
std::string date();
std::vector<std::string> camera_param_name_list(const std::string &model_name);
std::vector<Eigen::Vector2i> json2img_col_row(const std::string &file_path);

cv::Mat draw_observation(
    const cv::Mat &img,
    const Eigen::aligned_map<size_t, Eigen::Vector2f> &observation);

std::shared_ptr<cv::Mat> draw_obs_for_pango(
    const std::vector<cv::Mat> &imgs,
    const std::vector<Eigen::aligned_map<size_t, Eigen::Vector2f>>
        &current_obs);

// template fnuctions
template <typename Scalar>
Sophus::SE3<Scalar> rtvec2SE3(const cv::Vec<Scalar, 3> rvec,
                              const cv::Vec<Scalar, 3> tvec) {
  Eigen::Matrix<Scalar, 3, 1> r(rvec[0], rvec[1], rvec[2]);
  Eigen::Matrix<Scalar, 3, 1> t(tvec[0], tvec[1], tvec[2]);
  return Sophus::SE3<Scalar>(Sophus::SO3<Scalar>::exp(r), t);
}

template <template <typename> typename BasaltCameraModel, typename Scalar>
Eigen::aligned_vector<basalt::GenericCamera<Scalar>>
json2models(const std::string &filename) {
  const auto j = load_json(filename);

  const size_t cam_num = j["intrinsics"].size();
  Eigen::aligned_vector<basalt::GenericCamera<Scalar>> gcs(cam_num);

  for (int cam = 0; cam < cam_num; ++cam) {
    std::unordered_map<std::string, double> params =
        j["intrinsics"][cam]["intrinsics"];
    std::string camera_type =
        j["intrinsics"][cam].at("camera_type").get<std::string>();
    const auto param_name_list = camera_param_name_list(camera_type);

    typename BasaltCameraModel<Scalar>::VecN parameters;
    for (int i = 0; i < parameters.size(); ++i) {
      parameters[i] = params.at(param_name_list[i]);
    }
    gcs[cam].variant = BasaltCameraModel<Scalar>(parameters);
  }

  return gcs;
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