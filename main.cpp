#include <filesystem>
#include <iostream>

#include <CLI11.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <basalt/frame_to_frame_optical_flow.h>

#include <argus_utillity.h>
#include <dataset_loader/dataset_loader.h>
#include <camera_model/extended_unified_camera.hpp>

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
  CLI::App app{"Tiny VO"};

  std::string img_folder, camera_calib_file_path, data_config_file_path;
  app.add_option("-i,--img_folder", img_folder,
                 "img folder, containing mav0 folder")
      ->required();
  app.add_option("-c,--calib", camera_calib_file_path, "camera calib file path")
      ->default_val("config/eucm_512.json");
  app.add_option("-d,--data_config", data_config_file_path,
                 "dataset type config file path")
      ->default_val("config/tum_vi_dataset.json");

  CLI11_PARSE(app, argc, argv);
  const auto log_path = "logs/" + argus::date();
  if (!filesystem::exists(log_path)) {
    filesystem::create_directories(log_path);
  }

  // load dataset
  const auto data_type_config = argus::load_json(data_config_file_path);
  const auto dataset_loader_ptr =
      argus::DatasetLoaderFactory::getDataLoader(data_type_config, img_folder);
  const auto frame_num = dataset_loader_ptr->total_frame();
  const auto cam_num = dataset_loader_ptr->total_cam_num();

  // load cameras
  std::vector<Eigen::Vector2i> camera_col_row;
  const auto models = argus::json2models<argus::ExtendedUnifiedCamera<double>>(
      camera_calib_file_path);
  const auto T_cami_cam0 =
      argus::json2extrinsics<double>(camera_calib_file_path);

  // load basalt optical flow tracker
  Eigen::aligned_vector<basalt::GenericCamera<float>> gcs(models.size());
  for (int cam = 0; cam < models.size(); ++cam)
    gcs[cam].variant =
        basalt::ExtendedUnifiedCamera<float>(models[cam].get_param().cast<float>());
  basalt::FrameToFrameOpticalFlow<float, basalt::Pattern51>
      optical_flow_tracker(gcs, T_cami_cam0[1].inverse());

  // main loop
  for (auto frame_ptr = dataset_loader_ptr->get_img_frame(); frame_ptr;
       frame_ptr = dataset_loader_ptr->get_img_frame()) {
    optical_flow_tracker.processFrame(*frame_ptr);
    const auto current_obs = optical_flow_tracker.getObservations();
    // cout << current_obs[0].size() << endl;
    cv::Mat show;
    cv::hconcat(*frame_ptr, show);
    cv::imshow("img", show);
    cv::waitKey(1);
  }

  const std::string result_json_path = log_path + "/calib_result.json";
  const std::string result_poses_json_path = log_path + "/poses.json";
  const std::string result_pts_json_path = log_path + "/pts.json";
  std::vector<std::vector<Sophus::SE3d>> cam_poses;

  return 0;
}