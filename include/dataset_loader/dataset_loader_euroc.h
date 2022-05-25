#pragma once

#include "dataset_loader.h"

namespace argus {
class DatasetLoaderEuroc : public DatasetLoaderBase {
private:
  /* data */
  void parse_dataset_folder(std::string folder_path) override;

public:
  DatasetLoaderEuroc(const nlohmann::json &setting, std::string folder_path);
  ~DatasetLoaderEuroc();
};
} // namespace argus