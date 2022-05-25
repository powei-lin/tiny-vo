#include <dataset_loader/dataset_loader_euroc.h>

#include <iostream>
#include <map>
#include <opencv2/core/utility.hpp>
#include <unordered_map>
using namespace std;
using namespace cv;

namespace argus {
void DatasetLoaderEuroc::parse_dataset_folder(std::string folder_path) {
  string datasetDirPrefix = folder_path + "/mav0/cam";
  vector<vector<string>> cam_i_names(cam_num);

  map<unsigned long, int> synced_time;
  unordered_map<unsigned long, std::string> ts_name;
  for (int cam = 0; cam < cam_num; cam++) {
    string cam_folder_path = datasetDirPrefix + to_string(cam) + "/";
    vector<string> names;
    cv::glob(datasetDirPrefix + to_string(cam) + "/data/*.png", names);
    for (const auto &name : names) {
      const auto p0 = name.find_last_of('/');
      const auto p1 = name.find_last_of('.');
      const string ts_str = name.substr(p0 + 1, p1 - p0 - 1);
      unsigned long time_stamp = stoul(ts_str);
      if (synced_time.find(time_stamp) != synced_time.end()) {
        synced_time[time_stamp] += 1;
      } else {
        synced_time[time_stamp] = 1;
        ts_name[time_stamp] = ts_str;
      }
    }
  }

  // check the time_stamp exist in every cam
  for (const auto &ts : synced_time) {
    if (ts.second == cam_num) {
      for (int cam = 0; cam < cam_num; cam++) {
        const string image_path = datasetDirPrefix + to_string(cam) + "/data/" +
                                  ts_name.at(ts.first) + ".png";
        cam_i_names[cam].push_back(image_path);
      }
    }
  }

  frame_cam_names.resize(cam_i_names[0].size());
  for (size_t frame = 0; frame < frame_cam_names.size(); frame++) {
    vector<string> cam_name_in_frame(cam_i_names.size());
    for (size_t cam = 0; cam < cam_i_names.size(); cam++) {
      cam_name_in_frame[cam] = cam_i_names[cam][frame];
    }
    frame_cam_names[frame] = cam_name_in_frame;
  }
}

DatasetLoaderEuroc::DatasetLoaderEuroc(const nlohmann::json &setting,
                                       std::string folder_path)
    : DatasetLoaderBase(setting) {
  parse_dataset_folder(folder_path);
  reset();
}

DatasetLoaderEuroc::~DatasetLoaderEuroc() {}
} // namespace argus
