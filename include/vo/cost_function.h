#pragma once

#include <sophus/se3.hpp>

struct pnpFactor {
  pnpFactor(const Eigen::Vector3d &_obj_pt, const Eigen::Vector3d &_un_pt,
            const Sophus::SE3d &_T_i_0)
      : obj_pt(_obj_pt), un_pt(_un_pt), T_i_0(_T_i_0) {}

  template <typename T>
  bool operator()(const T *const _pose, T *residuals) const {
    //
    Eigen::Map<Sophus::SE3<T> const> const pose(_pose);

    const auto p3d = T_i_0.cast<T>() * pose * obj_pt.cast<T>();
    const auto p3d_norm_gt = un_pt.cast<T>();
    const auto p3d_norm = p3d.normalized();

    residuals[0] = p3d_norm_gt.x() - p3d_norm.x();
    residuals[1] = p3d_norm_gt.y() - p3d_norm.y();
    residuals[2] = p3d_norm_gt.z() - p3d_norm.z();

    return true;
  }

  const Eigen::Vector3d obj_pt, un_pt;
  const Sophus::SE3d T_i_0;
};