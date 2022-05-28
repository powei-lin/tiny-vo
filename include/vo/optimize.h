#pragma once

#include <Eigen/Dense>
#include <sophus/se3.hpp>

template <class Derived>
Eigen::Matrix<typename Derived::Scalar, 4, 1>
triangulate(const Eigen::MatrixBase<Derived> &f0,
            const Eigen::MatrixBase<Derived> &f1,
            const Sophus::SE3<typename Derived::Scalar> &T_0_1) {
  EIGEN_STATIC_ASSERT_VECTOR_SPECIFIC_SIZE(Derived, 3);

  // suffix "2" to avoid name clash with class-wide typedefs
  using Scalar_2 = typename Derived::Scalar;
  using Vec4_2 = Eigen::Matrix<Scalar_2, 4, 1>;

  Eigen::Matrix<Scalar_2, 3, 4> P1, P2;
  P1.setIdentity();
  P2 = T_0_1.inverse().matrix3x4();

  Eigen::Matrix<Scalar_2, 4, 4> A(4, 4);
  A.row(0) = f0[0] * P1.row(2) - f0[2] * P1.row(0);
  A.row(1) = f0[1] * P1.row(2) - f0[2] * P1.row(1);
  A.row(2) = f1[0] * P2.row(2) - f1[2] * P2.row(0);
  A.row(3) = f1[1] * P2.row(2) - f1[2] * P2.row(1);

  Eigen::JacobiSVD<Eigen::Matrix<Scalar_2, 4, 4>> mySVD(A, Eigen::ComputeFullV);
  Vec4_2 worldPoint = mySVD.matrixV().col(3);
  worldPoint /= worldPoint.template head<3>().norm();

  // Enforce same direction of bearing vector and initial point
  if (f0.dot(worldPoint.template head<3>()) < 0)
    worldPoint *= -1;

  return worldPoint;
}

Eigen::Matrix<double, 3, 1> triangulatePoint(Eigen::Vector2d &point0,
                                             Eigen::Vector2d &point1,
                                             Eigen::Matrix<double, 3, 4> Pose1);

bool ceres_solvePnp(Sophus::SE3d &pose, const std::vector<Sophus::SE3d> &T_i_0,
                    const std::vector<std::vector<Eigen::Vector3d>> &un_cam_pts,
                    const std::vector<std::vector<Eigen::Vector3d>> &cam_p3ds);