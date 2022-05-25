/**
BSD 3-Clause License

This file is part of the Basalt project.
https://gitlab.com/VladyslavUsenko/basalt.git

Copyright (c) 2019, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <thread>

#include <sophus/se2.hpp>
#include <sophus/se3.hpp>

#include <opencv2/features2d/features2d.hpp>

#include <tbb/blocked_range.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/parallel_for.h>

#include <basalt/patch.h>
#include <basalt/common_types.h>

#include <basalt/camera/generic_camera.hpp>
#include <basalt/utils/sophus_utils.hpp>
#include <basalt/image/image_pyr.h>

namespace basalt {

/// Unlike PatchOpticalFlow, FrameToFrameOpticalFlow always tracks patches
/// against the previous frame, not the initial frame where a track was created.
/// While it might cause more drift of the patch location, it leads to longer
/// tracks in practice.
template <typename Scalar, template <typename> typename Pattern>
class FrameToFrameOpticalFlow {
public:
  typedef OpticalFlowPatch<Scalar, Pattern<Scalar>> PatchT;

  typedef Eigen::Matrix<Scalar, 2, 1> Vector2;
  typedef Eigen::Matrix<Scalar, 2, 2> Matrix2;

  typedef Eigen::Matrix<Scalar, 3, 1> Vector3;
  typedef Eigen::Matrix<Scalar, 3, 3> Matrix3;

  typedef Eigen::Matrix<Scalar, 4, 1> Vector4;
  typedef Eigen::Matrix<Scalar, 4, 4> Matrix4;

  typedef Sophus::SE2<Scalar> SE2;
  using KeypointId = size_t;

  FrameToFrameOpticalFlow(
      const Eigen::aligned_vector<GenericCamera<Scalar>> &gcs,
      const Sophus::SE3d &T_i_j)
      : t_ns(-1), frame_counter(0), last_keypoint_id(0), cam_num(gcs.size()),
        intrinsics(gcs) {

    patch_coord = PatchT::pattern2.template cast<float>();

    if (cam_num > 1) {
      Eigen::Matrix4d Ed;
      // Sophus::SE3d T_i_j = calib.T_i_c[0].inverse() * calib.T_i_c[1];
      computeEssential(T_i_j, Ed);
      E = Ed.cast<Scalar>();
    }
  }

  ~FrameToFrameOpticalFlow() {}

  void showDebug(
      const std::vector<cv::Mat> &imgs,
      const std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
          &obs,
      std::string window_name) {

    cv::Mat allcamshow;
    for (int cam = 0; cam < cam_num; cam++) {
      cv::Mat img_show;
      cv::cvtColor(imgs[cam], img_show, cv::COLOR_GRAY2BGR);
      std::uniform_int_distribution<int> dis(1, 255);
      for (const auto &ob : obs.at(cam)) {
        std::mt19937 gen(ob.first);
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        cv::Point2f pt(ob.second.translation().x(),
                       ob.second.translation().y());
        cv::circle(img_show, pt, 3, color, -1);
        cv::putText(img_show, std::to_string(ob.first), pt, 1, 1, color);
      }
      if (cam == 0) {
        allcamshow = img_show;
      } else {
        cv::hconcat(allcamshow, img_show, allcamshow);
      }
    }
    cv::imshow(window_name, allcamshow);
  }

  std::vector<Eigen::aligned_map<KeypointId, Eigen::Vector2f>>
  getObservations() const {
    std::vector<Eigen::aligned_map<KeypointId, Eigen::Vector2f>> obs(cam_num);
    for (int cam = 0; cam < cam_num; ++cam) {
      for (const auto &kv : observations[cam]) {
        obs[cam][kv.first] = kv.second.translation();
      }
    }
    return obs;
  }

  void processFrame(const std::vector<cv::Mat> &imgs) {
    // std::vector<ImageData> imgData;
    // // MatToImageData(imgs, imgData);
    std::vector<ManagedImage<uint16_t>::Ptr> img_data(imgs.size());
    for (size_t i = 0; i < imgs.size(); ++i) {
      if (imgs[i].type() == CV_8UC1) {
        img_data[i].reset(
            new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));

        const uint8_t *data_in = imgs[i].ptr();
        uint16_t *data_out = img_data[i]->ptr;

        size_t full_size = imgs[i].cols * imgs[i].rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i];
          val = val << 8;
          data_out[i] = val;
        }
      } else if (imgs[i].type() == CV_8UC3) {
        img_data[i].reset(
            new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));

        const uint8_t *data_in = imgs[i].ptr();
        uint16_t *data_out = img_data[i]->ptr;

        size_t full_size = imgs[i].cols * imgs[i].rows;
        for (size_t i = 0; i < full_size; i++) {
          int val = data_in[i * 3];
          val = val << 8;
          data_out[i] = val;
        }
      } else if (imgs[i].type() == CV_16UC1) {
        img_data[i].reset(
            new ManagedImage<uint16_t>(imgs[i].cols, imgs[i].rows));
        std::memcpy(img_data[i]->ptr, imgs[i].ptr(),
                    imgs[i].cols * imgs[i].rows * sizeof(uint16_t));
      }
    }

    if (!initialized) {
      initialized = true;

      observations.resize(cam_num);

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(cam_num);

      tbb::parallel_for(tbb::blocked_range<size_t>(0, cam_num),
                        [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(*img_data[i],
                                                        optical_flow_levels);
                          }
                        });

      addPoints();
      filterPoints();

    } else {

      old_pyramid = pyramid;

      pyramid.reset(new std::vector<basalt::ManagedImagePyr<uint16_t>>);
      pyramid->resize(cam_num);
      tbb::parallel_for(tbb::blocked_range<size_t>(0, cam_num),
                        [&](const tbb::blocked_range<size_t> &r) {
                          for (size_t i = r.begin(); i != r.end(); ++i) {
                            pyramid->at(i).setFromImage(*img_data[i],
                                                        optical_flow_levels);
                          }
                        });

      std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
          new_observations(cam_num);

      for (size_t i = 0; i < cam_num; i++) {
        trackPoints(old_pyramid->at(i), pyramid->at(i), observations[i],
                    new_observations[i]);
      }
      observations = new_observations;

      addPoints();
      showDebug(imgs, observations, "new");
      // cv::waitKey(1);
      filterPoints();
    }

    frame_counter++;
  }

  void trackPoints(const basalt::ManagedImagePyr<uint16_t> &pyr_1,
                   const basalt::ManagedImagePyr<uint16_t> &pyr_2,
                   const Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
                       &transform_map_1,
                   Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>
                       &transform_map_2) const {
    size_t num_points = transform_map_1.size();

    std::vector<KeypointId> ids;
    Eigen::aligned_vector<Eigen::AffineCompact2f> init_vec;

    ids.reserve(num_points);
    init_vec.reserve(num_points);

    for (const auto &kv : transform_map_1) {
      ids.push_back(kv.first);
      init_vec.push_back(kv.second);
    }

    tbb::concurrent_unordered_map<KeypointId, Eigen::AffineCompact2f,
                                  std::hash<KeypointId>>
        result;

    auto compute_func = [&](const tbb::blocked_range<size_t> &range) {
      for (size_t r = range.begin(); r != range.end(); ++r) {
        const KeypointId id = ids[r];

        const Eigen::AffineCompact2f &transform_1 = init_vec[r];
        Eigen::AffineCompact2f transform_2 = transform_1;

        bool valid = trackPoint(pyr_1, pyr_2, transform_1, transform_2);

        if (valid) {
          Eigen::AffineCompact2f transform_1_recovered = transform_2;

          valid = trackPoint(pyr_2, pyr_1, transform_2, transform_1_recovered);

          if (valid) {
            Scalar dist2 = (transform_1.translation() -
                            transform_1_recovered.translation())
                               .squaredNorm();

            if (dist2 < optical_flow_max_recovered_dist2) {
              result[id] = transform_2;
            }
          }
        }
      }
    };

    tbb::blocked_range<size_t> range(0, num_points);

    tbb::parallel_for(range, compute_func);

    transform_map_2.clear();
    transform_map_2.insert(result.begin(), result.end());
  }

  inline bool trackPoint(const basalt::ManagedImagePyr<uint16_t> &old_pyr,
                         const basalt::ManagedImagePyr<uint16_t> &pyr,
                         const Eigen::AffineCompact2f &old_transform,
                         Eigen::AffineCompact2f &transform) const {
    bool patch_valid = true;

    transform.linear().setIdentity();

    for (int level = optical_flow_levels; level >= 0 && patch_valid; level--) {
      const Scalar scale = 1 << level;

      transform.translation() /= scale;

      PatchT p(old_pyr.lvl(level), old_transform.translation() / scale);

      patch_valid &= p.valid;
      if (patch_valid) {
        // Perform tracking on current level
        patch_valid &= trackPointAtLevel(pyr.lvl(level), p, transform);
      }

      transform.translation() *= scale;
    }

    transform.linear() = old_transform.linear() * transform.linear();

    return patch_valid;
  }

  inline bool trackPointAtLevel(const Image<const uint16_t> &img_2,
                                const PatchT &dp,
                                Eigen::AffineCompact2f &transform) const {
    bool patch_valid = true;

    for (int iteration = 0;
         patch_valid && iteration < optical_flow_max_iterations; iteration++) {
      typename PatchT::VectorP res;

      typename PatchT::Matrix2P transformed_pat =
          transform.linear().matrix() * PatchT::pattern2;
      transformed_pat.colwise() += transform.translation();

      patch_valid &= dp.residual(img_2, transformed_pat, res);

      if (patch_valid) {
        const Vector3 inc = -dp.H_se2_inv_J_se2_T * res;

        // avoid NaN in increment (leads to SE2::exp crashing)
        patch_valid &= inc.array().isFinite().all();

        // avoid very large increment
        patch_valid &= inc.template lpNorm<Eigen::Infinity>() < 1e6;

        if (patch_valid) {
          transform *= SE2::exp(inc).matrix();

          const int filter_margin = 2;

          patch_valid &= img_2.InBounds(transform.translation(), filter_margin);
        }
      }
    }

    return patch_valid;
  }

  void addPoints() {
    Eigen::aligned_vector<Eigen::Vector2d> pts0;

    for (const auto &kv : observations.at(0)) {
      pts0.emplace_back(kv.second.translation().template cast<double>());
    }

    KeypointsData kd;

    // detect point from img0
    detectKeypoints(pyramid->at(0).lvl(0), kd, optical_flow_detection_grid_size,
                    1, pts0);

    Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f> new_poses0,
        new_poses1;

    for (size_t i = 0; i < kd.corners.size(); i++) {
      Eigen::AffineCompact2f transform;
      transform.setIdentity();
      transform.translation() = kd.corners[i].cast<Scalar>();
      new_poses0[i] = transform;
    }

    if (cam_num > 1) {
      trackPoints(pyramid->at(0), pyramid->at(1), new_poses0, new_poses1);

      for (const auto &kv : new_poses1) {
        observations.at(0)[last_keypoint_id] = new_poses0.at(kv.first);
        observations.at(1)[last_keypoint_id++] = kv.second;
      }
    } else {
      for (const auto &kv : new_poses0) {
        observations.at(0)[last_keypoint_id++] = kv.second;
      }
    }
  }

  void filterPoints() {
    if (cam_num < 2)
      return;

    std::set<KeypointId> lm_to_remove;

    std::vector<KeypointId> kpid;
    Eigen::aligned_vector<Eigen::Vector2f> proj0, proj1;

    for (const auto &kv : observations.at(1)) {
      auto it = observations.at(0).find(kv.first);

      if (it != observations.at(0).end()) {
        proj0.emplace_back(it->second.translation());
        proj1.emplace_back(kv.second.translation());
        kpid.emplace_back(kv.first);
      }
    }

    Eigen::aligned_vector<Eigen::Vector4f> p3d0, p3d1;
    std::vector<bool> p3d0_success, p3d1_success;

    intrinsics[0].unproject(proj0, p3d0, p3d0_success);
    intrinsics[1].unproject(proj1, p3d1, p3d1_success);

    for (size_t i = 0; i < p3d0_success.size(); i++) {
      if (p3d0_success[i] && p3d1_success[i]) {
        const double epipolar_error =
            std::abs(p3d0[i].transpose() * E * p3d1[i]);

        if (epipolar_error > optical_flow_epipolar_error) {
          lm_to_remove.emplace(kpid[i]);
        }
      } else {
        lm_to_remove.emplace(kpid[i]);
      }
    }

    for (int id : lm_to_remove) {
      observations.at(0).erase(id);
      observations.at(1).erase(id);
    }
  }
  void detectKeypoints(
      const basalt::Image<const uint16_t> &img_raw, KeypointsData &kd,
      int PATCH_SIZE, int num_points_cell,
      const Eigen::aligned_vector<Eigen::Vector2d> &current_points) {
    kd.corners.clear();
    kd.corner_angles.clear();
    kd.corner_descriptors.clear();

    const size_t x_start = (img_raw.w % PATCH_SIZE) / 2;
    const size_t x_stop = x_start + PATCH_SIZE * (img_raw.w / PATCH_SIZE - 1);

    const size_t y_start = (img_raw.h % PATCH_SIZE) / 2;
    const size_t y_stop = y_start + PATCH_SIZE * (img_raw.h / PATCH_SIZE - 1);

    //  std::cerr << "x_start " << x_start << " x_stop " << x_stop << std::endl;
    //  std::cerr << "y_start " << y_start << " y_stop " << y_stop << std::endl;

    Eigen::MatrixXi cells;
    cells.setZero(img_raw.h / PATCH_SIZE + 1, img_raw.w / PATCH_SIZE + 1);

    for (const Eigen::Vector2d &p : current_points) {
      if (p[0] >= x_start && p[1] >= y_start && p[0] < x_stop + PATCH_SIZE &&
          p[1] < y_stop + PATCH_SIZE) {
        int x = (p[0] - x_start) / PATCH_SIZE;
        int y = (p[1] - y_start) / PATCH_SIZE;

        cells(y, x) += 1;
      }
    }

    for (size_t x = x_start; x <= x_stop; x += PATCH_SIZE) {
      for (size_t y = y_start; y <= y_stop; y += PATCH_SIZE) {
        if (cells((y - y_start) / PATCH_SIZE, (x - x_start) / PATCH_SIZE) > 0)
          continue;

        const basalt::Image<const uint16_t> sub_img_raw =
            img_raw.SubImage(x, y, PATCH_SIZE, PATCH_SIZE);

        cv::Mat subImg(PATCH_SIZE, PATCH_SIZE, CV_8U);

        for (int y = 0; y < PATCH_SIZE; y++) {
          uchar *sub_ptr = subImg.ptr(y);
          for (int x = 0; x < PATCH_SIZE; x++) {
            sub_ptr[x] = (sub_img_raw(x, y) >> 8);
          }
        }

        int points_added = 0;
        int threshold = 40;

        while (points_added < num_points_cell && threshold >= 2) {
          std::vector<cv::KeyPoint> points;
          cv::FAST(subImg, points, threshold);

          std::sort(points.begin(), points.end(),
                    [](const cv::KeyPoint &a, const cv::KeyPoint &b) -> bool {
                      return a.response > b.response;
                    });

          //        std::cout << "Detected " << points.size() << " points.
          //        Threshold "
          //                  << threshold << std::endl;

          for (size_t i = 0;
               i < points.size() && points_added < num_points_cell; i++)
            if (img_raw.InBounds(x + points[i].pt.x, y + points[i].pt.y,
                                 edge_threshold)) {
              kd.corners.emplace_back(x + points[i].pt.x, y + points[i].pt.y);
              points_added++;
            }

          threshold /= 2;
        }
      }
    }

    // std::cout << "Total points: " << kd.corners.size() << std::endl;

    //  cv::TermCriteria criteria =
    //      cv::TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 40, 0.001);
    //  cv::Size winSize = cv::Size(5, 5);
    //  cv::Size zeroZone = cv::Size(-1, -1);
    //  cv::cornerSubPix(image, points, winSize, zeroZone, criteria);

    //  for (size_t i = 0; i < points.size(); i++) {
    //    if (img_raw.InBounds(points[i].pt.x, points[i].pt.y, edge_threshold))
    //    {
    //      kd.corners.emplace_back(points[i].pt.x, points[i].pt.y);
    //    }
    //  }
  }

  inline void computeEssential(const Sophus::SE3d &T_0_1, Eigen::Matrix4d &E) {
    E.setZero();
    const Eigen::Vector3d t_0_1 = T_0_1.translation();
    const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

    E.topLeftCorner<3, 3>() = Sophus::SO3d::hat(t_0_1.normalized()) * R_0_1;
  }

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
private:
  int64_t t_ns;
  const int cam_num;
  const Eigen::aligned_vector<GenericCamera<Scalar>> intrinsics;
  static constexpr uint8_t optical_flow_detection_grid_size = 50;
  static constexpr float optical_flow_max_recovered_dist2 = 0.04;
  static constexpr uint8_t optical_flow_max_iterations = 5;
  static constexpr uint8_t optical_flow_levels = 3;
  static constexpr float optical_flow_epipolar_error = 0.005;
  static constexpr int edge_threshold = 19;

  size_t frame_counter;
  bool initialized = false;

  KeypointId last_keypoint_id;

  std::vector<Eigen::aligned_map<KeypointId, Eigen::AffineCompact2f>>
      observations;

  std::vector<std::map<KeypointId, size_t>> pyramid_levels;
  std::shared_ptr<std::vector<basalt::ManagedImagePyr<uint16_t>>> old_pyramid,
      pyramid;

  Matrix4 E;
  Eigen::MatrixXf patch_coord;
};

} // namespace basalt
