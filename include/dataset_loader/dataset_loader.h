#pragma once
#include <memory>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <json.hpp>
#include <opencv2/core.hpp>

namespace argus {
class DatasetLoaderBase {
protected:
  /* data */
  std::vector<std::vector<std::string>> frame_cam_names;
  std::queue<std::shared_ptr<std::vector<cv::Mat>>> frameBuffer;

  constexpr static uint16_t frame_buffer_size = 500;

  const uint16_t cam_num;
  const uint16_t start_idx;
  const uint16_t step;
  const uint16_t max_frame_num;

  uint16_t currentFrameIdx = 0;
  uint16_t count_used = 0;

  std::thread readingThread;
  std::mutex bufferMutex;
  bool needTerminate = false;

  virtual void parse_dataset_folder(std::string folder_path) = 0;

public:
  typedef std::shared_ptr<DatasetLoaderBase> Ptr;
  DatasetLoaderBase(const nlohmann::json &setting);
  std::shared_ptr<std::vector<cv::Mat>> get_img_frame();
  size_t total_frame() const;
  uint16_t total_cam_num() const;
  void reset();
  void stop();
  ~DatasetLoaderBase();
};

class DatasetLoaderFactory {
public:
  static DatasetLoaderBase::Ptr getDataLoader(const nlohmann::json &data_config,
                                              std::string file_path);
};
} // namespace argus