#include <filesystem>
#include <iostream>
#include <queue>

#include <CLI11.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>
#include <basalt/frame_to_frame_optical_flow.h>

#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/scene/axis.h>

#include <argus_utillity.h>
#include <dataset_loader/dataset_loader.h>
#include <camera_model/extended_unified_camera.hpp>
#include <vo/stereo_vo.h>

using namespace std;
using namespace cv;

const uint8_t cam_color[3]{250, 0, 26};
const uint8_t state_color[3]{250, 0, 26};
const uint8_t pose_color[3]{0, 50, 255};
const uint8_t gt_color[3]{0, 171, 47};

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
  const int window_width = 1280;
  const int window_height = 720;
  const int image_window_height = window_height;
  const int image_window_width =
      image_window_height / cam_num * img_col_row.x() / img_col_row.y();
  const int main_window_width = window_width - image_window_width;
  const float image_bound = (float)window_height * image_window_width /
                            image_window_height / window_width;

  pangolin::CreateWindowAndBind("Main", window_width, window_height);
  glEnable(GL_DEPTH_TEST);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(main_window_width, window_height, 720, 720,
                                 main_window_width / 2, window_height / 2, 0.1,
                                 100),
      pangolin::ModelViewLookAt(0, -2, -5, 0, 0, 0, pangolin::AxisNegY));

  pangolin::View &d_cam =
      pangolin::Display("cam")
          .SetBounds(0, 1.0f, image_bound, 1.0f,
                     (float)main_window_width / window_height)
          .SetHandler(new pangolin::Handler3D(s_cam));

  pangolin::View &d_image =
      pangolin::Display("image")
          .SetBounds(0, 1.0f, 0, image_bound,
                     (float)image_window_width / image_window_height)
          .SetLock(pangolin::LockLeft, pangolin::LockTop);

  pangolin::GlTexture imageTexture(img_col_row.x(), img_col_row.y() * cam_num,
                                   GL_RGB, false, 0, GL_RGB, GL_UNSIGNED_BYTE);

  pangolin::Renderable tree;
  auto axis_i = std::make_shared<pangolin::Axis>();
  tree.Add(axis_i);

  std::queue<std::shared_ptr<cv::Mat>> img_with_points;
  std::queue<Sophus::SE3d> poses;
  std::queue<std::vector<Eigen::Vector3d>> points_3d;

  StereoVo stereo_vo(T_cami_cam0);

  std::thread opt_flow_thread([&]() {
    for (auto frame_ptr = dataset_loader_ptr->get_img_frame(); frame_ptr;
         frame_ptr = dataset_loader_ptr->get_img_frame()) {

      // add feature points and track
      optical_flow_tracker.processFrame(*frame_ptr);
      const auto current_obs = optical_flow_tracker.getObservations();
      const auto current_un_pts = optical_flow_tracker.getUndistortPoints();
      std::unordered_set<size_t> bad_ids;
      stereo_vo.track(current_un_pts, bad_ids);
      optical_flow_tracker.removeObservations(bad_ids);

      img_with_points.push(argus::draw_obs_for_pango(*frame_ptr, current_obs));
      poses.push(stereo_vo.get_current_pose());
      points_3d.push(stereo_vo.get_landmark_p3d());
    }
    img_with_points.push(nullptr);

    std::cout << "Finished optical flow thread." << std::endl;
  });

  // drawing loop
  while (!pangolin::ShouldQuit()) {
    while (img_with_points.empty())
      this_thread::sleep_for(chrono::milliseconds(5));
    auto img_ptr = img_with_points.front();
    if (img_ptr == nullptr) {
      break;
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glColor3f(1.0, 1.0, 1.0);
    // pangolin::glDrawColouredCube();
    // m << 420, 0, -main_window_width / 2, 0, 420, -window_height / 2, 0, 0, 1;
    // pangolin::glDrawFrustum(m, 640, 480, -0.0001);
    argus::render_camera(poses.front().inverse().matrix(), 2.0f, cam_color,
                         0.1f);
    tree.Render();
    glColor3ubv(pose_color);
    pangolin::glDrawPoints(points_3d.front());

    imageTexture.Upload(img_ptr->ptr(), GL_BGR, GL_UNSIGNED_BYTE);
    d_image.Activate();
    glColor3f(1.0, 1.0, 1.0);
    imageTexture.RenderToViewport();

    pangolin::FinishFrame();
    img_with_points.pop();
    poses.pop();
    points_3d.pop();
    this_thread::sleep_for(chrono::milliseconds(30));
  }
  opt_flow_thread.join();

  const std::string result_json_path = log_path + "/calib_result.json";
  const std::string result_poses_json_path = log_path + "/poses.json";
  const std::string result_pts_json_path = log_path + "/pts.json";
  std::vector<std::vector<Sophus::SE3d>> cam_poses;

  return 0;
}