#include <vo/optimize.h>
#include <queue>
#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>

#include <vo/cost_function.h>

Eigen::Matrix<double, 3, 1>
triangulatePoint(Eigen::Vector2d &point0, Eigen::Vector2d &point1,
                 Eigen::Matrix<double, 3, 4> Pose1) {
  Eigen::Matrix<double, 3, 4> Pose0;
  Pose0.setIdentity();
  Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
  design_matrix.row(0) = point0[0] * Pose0.row(2) - Pose0.row(0);
  design_matrix.row(1) = point0[1] * Pose0.row(2) - Pose0.row(1);
  design_matrix.row(2) = point1[0] * Pose1.row(2) - Pose1.row(0);
  design_matrix.row(3) = point1[1] * Pose1.row(2) - Pose1.row(1);

  Eigen::JacobiSVD<Eigen::Matrix<double, 4, 4>> mySVD(design_matrix,
                                                      Eigen::ComputeFullV);
  Eigen::Vector4d triangulated_point;
  triangulated_point = mySVD.matrixV().rightCols<1>();
  Eigen::Vector3d point_3d;
  point_3d(0) = triangulated_point(0) / triangulated_point(3);
  point_3d(1) = triangulated_point(1) / triangulated_point(3);
  point_3d(2) = triangulated_point(2) / triangulated_point(3);
  return point_3d;
}

bool ceres_solvePnp(Sophus::SE3d &pose, const std::vector<Sophus::SE3d> &T_i_0,
                    const std::vector<std::vector<Eigen::Vector3d>> &un_cam_pts,
                    const std::vector<std::vector<Eigen::Vector3d>> &cam_p3ds) {

  ceres::Problem::Options problem_options;
  problem_options.enable_fast_removal = true;
  ceres::Problem problem(problem_options);
  ceres::Manifold *manifold = new Sophus::Manifold<Sophus::SE3>();
  ceres::LossFunction *loss = new ceres::HuberLoss(0.5);
  std::vector<ceres::ResidualBlockId> res_block;

  constexpr int residual_degree = 3;
  constexpr int pose_degree = 7;

  const size_t cam_num = un_cam_pts.size();
  for (int cam = 0; cam < cam_num; ++cam) {
    const size_t pt_num = un_cam_pts[cam].size();
    for (int p = 0; p < pt_num; ++p) {
      ceres::CostFunction *cost =
          new ceres::AutoDiffCostFunction<pnpFactor, residual_degree,
                                          pose_degree>(
              new pnpFactor(cam_p3ds[cam][p], un_cam_pts[cam][p], T_i_0[cam]));

      res_block.push_back(problem.AddResidualBlock(cost, loss, pose.data()));
    }
  }
  problem.SetManifold(pose.data(), manifold);

  ceres::Solver::Summary summary;
  ceres::Solver::Options solver_options;
  ceres::Solve(solver_options, &problem, &summary);

  // evaluate reprojection error
  std::priority_queue<std::pair<double, ceres::ResidualBlockId>> pq;
  for (auto &rbid : res_block) {
    Eigen::Vector3d r;
    problem.EvaluateResidualBlock(rbid, false, nullptr, r.data(), nullptr);
    pq.emplace(r.norm(), rbid);
  }

  // use best 30 points to refine the pose
  constexpr int best_n_points = 30;
  while (pq.size() > best_n_points) {
    problem.RemoveResidualBlock(pq.top().second);
    pq.pop();
  }
  ceres::Solve(solver_options, &problem, &summary);

  return summary.termination_type == ceres::CONVERGENCE;
}