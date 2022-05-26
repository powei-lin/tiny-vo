#pragma once

#include <fstream>
#include <iostream>
#include <variant>
#include <vector>

#include <json.hpp>
#include <opencv2/calib3d.hpp>
#include <sophus/se3.hpp>
#include <basalt/utils/eigen_utils.hpp>

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

// template <typename ArgusCameraModel>
// Sophus::SE3d solvePnP(const typename ArgusCameraModel::VecN &param,
//                       const std::vector<argus::DetectedPoint<double>> &pts,
//                       bool coplanar = false) {
//   std::vector<cv::Point2d> p2ds;
//   std::vector<cv::Point3d> p3ds;
//   for (const auto &pt : pts) {
//     Eigen::Vector3d undistort_pt;
//     ArgusCameraModel::unproject(param.data(), pt.img_pt, undistort_pt);
//     p2ds.emplace_back(undistort_pt.x(), undistort_pt.y());
//     p3ds.emplace_back(pt.obj_pt.x(), pt.obj_pt.y(), pt.obj_pt.z());
//   }
//   cv::Vec3d rvec, tvec;
//   if (coplanar) {
//     cv::solvePnP(p3ds, p2ds, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec,
//     tvec,
//                  false, cv::SOLVEPNP_IPPE);
//   } else {
//     cv::solvePnP(p3ds, p2ds, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec,
//     tvec);
//   }
//   return rtvec2SE3(rvec, tvec);
// }

// template <typename CameraModelVariant>
// Sophus::SE3d solvePnP(const CameraModelVariant &camera_model_variant,
//                       const std::vector<argus::DetectedPoint<double>> &pts) {
//   std::vector<cv::Point2d> p2ds;
//   std::vector<cv::Point3d> p3ds;
//   for (const auto &pt : pts) {
//     Eigen::Vector3d undistort_pt;

//     std::visit([&](auto &&arg) { arg.unproject(pt.img_pt, undistort_pt); },
//                camera_model_variant);
//     p2ds.emplace_back(undistort_pt.x(), undistort_pt.y());
//     p3ds.emplace_back(pt.obj_pt.x(), pt.obj_pt.y(), pt.obj_pt.z());
//   }
//   cv::Vec3d rvec, tvec;
//   cv::solvePnP(p3ds, p2ds, cv::Mat::eye(3, 3, CV_64F), cv::Mat(), rvec,
//   tvec); cv::solvePnPRefineLM(p3ds, p2ds, cv::Mat::eye(3, 3, CV_64F),
//   cv::Mat(), rvec,
//                        tvec);
//   return rtvec2SE3(rvec, tvec);
// }

// template <int I = 0>
// std::vector<std::string> get_param_name_list(const std::string &target_model)
// {
//   if constexpr (I < std::variant_size_v<camera_model_d>) {
//     using cam_t = typename std::variant_alternative<I, camera_model_d>::type;
//     if (target_model == cam_t::model_name()) {
//       return cam_t::param_name_list();
//     } else {
//       return get_param_name_list<I + 1>(target_model);
//     }
//   } else {
//     std::cout << "no matched model" << std::endl;
//     return std::vector<std::string>();
//   }
// }

// template <typename CameraModelVariant, int I = 0>
// void visit_all_model(const std::string &target_model,
//                      CameraModelVariant &camera_model_variant,
//                      const std::unordered_map<std::string, double> &params,
//                      const Eigen::Vector2i &img_col_row) {
//   if constexpr (I < std::variant_size_v<CameraModelVariant>) {
//     using cam_t =
//         typename std::variant_alternative<I, CameraModelVariant>::type;
//     if (target_model == cam_t::model_name()) {
//       std::cout << cam_t::model_name() << std::endl;
//       const auto param_name_list = cam_t::param_name_list();
//       typename cam_t::VecN parameters;
//       for (int i = 0; i < cam_t::N; ++i) {
//         parameters[i] = params.at(param_name_list[i]);
//       }
//       camera_model_variant = cam_t(parameters, img_col_row);
//     } else {
//       visit_all_model<CameraModelVariant, I + 1>(
//           target_model, camera_model_variant, params, img_col_row);
//     }
//   } else {
//     std::cout << "no matched model" << std::endl;
//   }
// }

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

} // namespace argus