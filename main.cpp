#include <filesystem>
#include <iostream>

#include <CLI11.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <basalt/frame_to_frame_optical_flow.h>

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

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
  const auto models = argus::json2models<argus::ExtendedUnifiedCamera<double>>(
      camera_calib_file_path);
  const auto img_col_row = models[0].get_img_col_row();
  const auto T_cami_cam0 =
      argus::json2extrinsics<double>(camera_calib_file_path);

  // load basalt optical flow tracker
  Eigen::aligned_vector<basalt::GenericCamera<float>> gcs(models.size());
  for (int cam = 0; cam < models.size(); ++cam)
    gcs[cam].variant = basalt::ExtendedUnifiedCamera<float>(
        models[cam].get_param().cast<float>());
  basalt::FrameToFrameOpticalFlow<float, basalt::Pattern51>
      optical_flow_tracker(gcs, T_cami_cam0[1].inverse());

  // Create OpenGL window in single line
  pangolin::CreateWindowAndBind("Main", 1280, 720);
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 240, 0.1, 1000),
      pangolin::ModelViewLookAt(-1, 1, -1, 0, 0, 0, pangolin::AxisY));

  pangolin::View &d_cam = pangolin::Display("cam")
                              .SetBounds(0, 1.0f, 0, 1.0f, -640 / 480.0)
                              .SetHandler(new pangolin::Handler3D(s_cam));

  const int width = img_col_row.x();
  const int height = img_col_row.y() * cam_num;
  pangolin::View &d_image =
      pangolin::Display("image")
          .SetBounds(0, 1.0f, 0, 720.0 / 2 / 1280, (float)width / height)
          .SetLock(pangolin::LockLeft, pangolin::LockTop);

  pangolin::GlTexture imageTexture(width, height, GL_RGB, false, 0, GL_RGB,
                                   GL_UNSIGNED_BYTE);

  // main loop
  for (auto frame_ptr = dataset_loader_ptr->get_img_frame(); frame_ptr;
       frame_ptr = dataset_loader_ptr->get_img_frame()) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glColor3f(1.0, 1.0, 1.0);
    pangolin::glDrawColouredCube();

    // add feature points and track
    optical_flow_tracker.processFrame(*frame_ptr);
    const auto current_obs = optical_flow_tracker.getObservations();

    std::vector<cv::Mat> img_with_points(cam_num);
    for (int cam = 0; cam < cam_num; ++cam) {
      img_with_points[cam] =
          argus::draw_observation((*frame_ptr)[cam], current_obs[cam]);
    }
    // cout << current_obs[0].size() << endl;
    cv::Mat show;
    cv::vconcat(img_with_points, show);

    cv::flip(show, show, 0);
    imageTexture.Upload(show.ptr(), GL_BGR, GL_UNSIGNED_BYTE);
    d_image.Activate();
    glColor3f(1.0, 1.0, 1.0);
    imageTexture.RenderToViewport();

    pangolin::FinishFrame();
    // cv::imshow("img", show);
    // cv::waitKey(1);
  }

  const std::string result_json_path = log_path + "/calib_result.json";
  const std::string result_poses_json_path = log_path + "/poses.json";
  const std::string result_pts_json_path = log_path + "/pts.json";
  std::vector<std::vector<Sophus::SE3d>> cam_poses;

  return 0;
}