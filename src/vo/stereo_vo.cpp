#include <vo/stereo_vo.h>
#include <iostream>
#include <argus_utillity.h>
#include <vo/optimize.h>

StereoVo::StereoVo(const std::vector<Sophus::SE3d> &_T_i_0)
    : T_i_0(_T_i_0), count_pose(0){};

StereoVo::~StereoVo() {}

double angle_deg(Eigen::Vector3d v0, Eigen::Vector3d v1) {
  double n0 = v0.norm();
  double n1 = v1.norm();
  return acos(v0.dot(v1) / n0 / n1) * 180.0 / M_PI;
}

bool StereoVo::track(
    const std::vector<Eigen::aligned_map<size_t, Eigen::Vector3f>>
        &current_un_pts,
    std::unordered_set<size_t> &bad_ids) {

  // seperate known and unknown pts
  std::unordered_set<size_t> unknow_id;
  std::vector<std::vector<Eigen::Vector3d>> un_cam_pts, cam_p3ds;

  for (const auto &cam_pts : current_un_pts) {

    std::vector<Eigen::Vector3d> un_pts, p3ds;

    for (const auto &kv : cam_pts) {
      if (auto landmark_ptr = landmark_db.find(kv.first);
          landmark_ptr != landmark_db.end()) {
        un_pts.emplace_back(kv.second.cast<double>());
        p3ds.emplace_back(landmark_ptr->second.position);
      } else {
        unknow_id.insert(kv.first);
      }
    }
    un_cam_pts.push_back(un_pts);
    cam_p3ds.push_back(p3ds);
  }

  // initialize
  if (count_pose == 0) {
    // first pose identity
    poses.push_back(Sophus::SE3d());
  } else {
    // use known pts for pnp
    Sophus::SE3d pose_init = poses.back();
    ceres_solvePnp(pose_init, T_i_0, un_cam_pts, cam_p3ds);
    poses.push_back(pose_init);
  }

  // triangulate unknown pts
  const Sophus::SE3d T_0_1 = T_i_0[1].inverse();
  for (auto id : unknow_id) {
    if (auto kv1_ptr = current_un_pts[1].find(id);
        kv1_ptr != current_un_pts[1].end()) {
      // std::cout << id << std::endl;
      const auto &v0 = current_un_pts[0].at(id);
      const auto &v1 = kv1_ptr->second;

      Eigen::Vector2d p0 = (v0 / v0.z()).head<2>().cast<double>();
      Eigen::Vector2d p1 = (v1 / v1.z()).head<2>().cast<double>();
      auto p3d = triangulatePoint(p0, p1, T_i_0[1].matrix3x4());
      if (p3d.z() > 5.0 || p3d.z() <= 0) {
        bad_ids.insert(id);
        continue;
      }

      auto p3d_w = poses.back().inverse() * p3d;
      landmark_db[id].id = id;
      landmark_db[id].position = p3d_w;
      landmark_db[id].start_frame_idx = count_pose;
      landmark_db[id].un_pt0_in_frame.emplace_back(v0.cast<double>());
    }
  }

  // std::cout << current_un_pts[0].size() << " " << current_un_pts[1].size() <<
  // std::endl;

  ++count_pose;
  return true;
}

Sophus::SE3d StereoVo::get_current_pose() const { return poses.back(); }
std::vector<Eigen::Vector3d> StereoVo::get_landmark_p3d() const {
  std::vector<Eigen::Vector3d> landmark_p3ds;
  landmark_p3ds.reserve(landmark_db.size());
  for (const auto &lm : landmark_db) {
    landmark_p3ds.push_back(lm.second.position);
  }
  return landmark_p3ds;
}