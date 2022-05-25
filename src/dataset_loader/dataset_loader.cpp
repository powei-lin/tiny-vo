#include <chrono>

#include <opencv2/imgcodecs.hpp>

#include <dataset_loader/dataset_loader.h>
#include <dataset_loader/dataset_loader_euroc.h>

using namespace std;
using namespace cv;

namespace argus {
std::shared_ptr<std::vector<cv::Mat>> DatasetLoaderBase::get_img_frame() {
  while (frameBuffer.empty()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  auto frame = frameBuffer.front();
  if (frame != nullptr) {
    frameBuffer.pop();
  }
  return frame;
}

size_t DatasetLoaderBase::total_frame() const {
  return min<int>((frame_cam_names.size() - start_idx) / step, max_frame_num);
}

uint16_t DatasetLoaderBase::total_cam_num() const { return cam_num; }

void DatasetLoaderBase::reset() {
  needTerminate = true;
  if (readingThread.joinable()) {
    readingThread.join();
  }
  while (!frameBuffer.empty()) {
    frameBuffer.pop();
  }

  needTerminate = false;
  currentFrameIdx = start_idx;
  count_used = 0;
  readingThread = std::thread([&]() {
    while (!needTerminate && currentFrameIdx < frame_cam_names.size() &&
           count_used < max_frame_num) {
      if (frameBuffer.size() < frame_buffer_size) {
        vector<cv::Mat> frame_i(cam_num);
        for (int cam = 0; cam < cam_num; cam++) {
          frame_i[cam] = cv::imread(frame_cam_names[currentFrameIdx][cam],
                                    cv::IMREAD_GRAYSCALE);
        }
        frameBuffer.push(make_shared<vector<cv::Mat>>(frame_i));
        currentFrameIdx += step;
        count_used += 1;
      } else {
        this_thread::sleep_for(chrono::milliseconds(5));
      }
    }
    frameBuffer.push(nullptr);
  });
}

DatasetLoaderBase::~DatasetLoaderBase() {
  if (readingThread.joinable())
    readingThread.join();
}

void DatasetLoaderBase::stop() {
  needTerminate = true;
  if (readingThread.joinable())
    readingThread.join();
}

DatasetLoaderBase::DatasetLoaderBase(const nlohmann::json &setting)
    : cam_num(setting.at("cam_num").get<uint16_t>()),
      start_idx(
          std::max<uint16_t>(0, setting.at("start_frame_idx").get<uint16_t>())),
      step(std::max<uint16_t>(1, setting.at("step").get<uint16_t>())),
      max_frame_num(setting.at("max_frame_num").get<uint16_t>()) {}

DatasetLoaderBase::Ptr
DatasetLoaderFactory::getDataLoader(const nlohmann::json &data_config,
                                    std::string folder_path) {
  DatasetLoaderBase::Ptr res;
  std::string ds_type = data_config.at("dataset_type").get<std::string>();
  if (ds_type == "euroc") {
    res.reset(new DatasetLoaderEuroc(data_config.at("setting"), folder_path));
  }
  return res;
}
} // namespace argus