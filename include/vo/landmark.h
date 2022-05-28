#pragma once

#include <basalt/utils/eigen_utils.hpp>

struct Landmark {
  Landmark() {}
  ~Landmark() {}
  size_t id;
  bool initialized = false;
  unsigned start_frame_idx;
  Eigen::Vector3d position;
  Eigen::aligned_vector<Eigen::Vector3d> un_pt0_in_frame;
  Eigen::aligned_vector<Eigen::Vector3d> un_pt1_in_frame;
};
