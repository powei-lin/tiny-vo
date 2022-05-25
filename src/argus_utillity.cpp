#include <argus_utillity.h>

#include <chrono>
#include <ctime>
#include <numeric>

namespace argus {
nlohmann::json load_json(const std::string &file_path) {
  std::ifstream i(file_path);
  nlohmann::json j;
  i >> j;
  i.close();
  return j;
}

void write_json(const std::string &file_path, const nlohmann::ordered_json &j) {
  std::ofstream o(file_path);
  o << std::setw(4) << j;
  o.close();
}

std::string date() {
  std::time_t t = std::time(nullptr);
  char mbstr[100];
  if (std::strftime(mbstr, sizeof(mbstr), "%Y%m%d_%H%M%S",
                    std::localtime(&t))) {
    return std::string(mbstr);
  }
  return std::string();
}

} // namespace argus