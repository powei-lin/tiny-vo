#pragma once

#include <unordered_set>
#include <sophus/se3.hpp>
#include <vo/landmark.h>

class StereoVo {
private:
  Eigen::aligned_unordered_map<size_t, Landmark> landmark_db;
  Eigen::aligned_vector<Sophus::SE3d> poses;
  size_t count_pose;
  const std::vector<Sophus::SE3d> T_i_0;

public:
  StereoVo(const std::vector<Sophus::SE3d> &);
  ~StereoVo();
  bool track(const std::vector<Eigen::aligned_map<size_t, Eigen::Vector3f>>
                 &current_un_pts,
             std::unordered_set<size_t> &bad_ids);
  Sophus::SE3d get_current_pose() const;
  std::vector<Eigen::Vector3d> get_landmark_p3d() const;
};
