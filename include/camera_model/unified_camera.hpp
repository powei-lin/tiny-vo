#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace argus {
template <typename Scalar = double> struct UnifiedCamera {
  static constexpr int N = 5;

  using VecN = Eigen::Matrix<Scalar, N, 1>;

  static std::string model_name() { return "ucm"; }
  static std::vector<std::string> param_name_list() {
    return {"fx", "fy", "cx", "cy", "alpha"};
  }
  VecN param;
  Eigen::Vector2i img_col_row;
  UnifiedCamera() {
    param.setZero();
    img_col_row.setZero();
  }
  UnifiedCamera(const VecN &_param) : param(_param) { img_col_row.setZero(); }
  UnifiedCamera(const VecN &_param, const Eigen::Vector2i _img_col_row)
      : param(_param), img_col_row(_img_col_row) {}

  VecN get_param() const { return param; }
  void set_param(const VecN &_param) { param = _param; }
  Eigen::Vector2i get_img_col_row() const { return img_col_row; }
  void set_img_col_row(const Eigen::Vector2i _img_col_row) {
    img_col_row = _img_col_row;
  }

  bool project(const Eigen::Matrix<Scalar, 3, 1> &p3d,
               Eigen::Matrix<Scalar, 2, 1> &p2d) const {
    return UnifiedCamera::project(param.data(), p3d, p2d);
  }
  bool unproject(const Eigen::Matrix<Scalar, 2, 1> &p2d,
                 Eigen::Matrix<Scalar, 3, 1> &p3d) const {
    return UnifiedCamera::unproject(param.data(), p2d, p3d);
  }

  ~UnifiedCamera() {}

  template <typename T>
  static bool project(const T *param, const Eigen::Matrix<T, 3, 1> &p3d,
                      Eigen::Matrix<T, 2, 1> &p2d) {
    const T &fx = param[0];
    const T &fy = param[1];
    const T &cx = param[2];
    const T &cy = param[3];
    const T &alpha = param[4];

    const T &x = p3d[0];
    const T &y = p3d[1];
    const T &z = p3d[2];

    const T r2 = x * x + y * y;
    const T rho2 = r2 + z * z;
    const T rho = sqrt(rho2);

    const T norm = alpha * rho + (T(1) - alpha) * z;

    const T mx = x / norm;
    const T my = y / norm;

    p2d[0] = fx * mx + cx;
    p2d[1] = fy * my + cy;

    // Check if valid
    const T w =
        alpha > T(0.5) ? (T(1) - alpha) / alpha : alpha / (T(1) - alpha);
    const bool is_valid = (z > -w * rho);

    return is_valid;
  }

  template <typename T>
  static bool unproject(const T *param, const Eigen::Matrix<T, 2, 1> &p2d,
                        Eigen::Matrix<T, 3, 1> &p3d) {

    const T &fx = param[0];
    const T &fy = param[1];
    const T &cx = param[2];
    const T &cy = param[3];
    const T &alpha = param[4];

    const T &u = p2d[0];
    const T &v = p2d[1];

    const T xi = alpha / (T(1) - alpha);

    const T mxx = (u - cx) / fx;
    const T myy = (v - cy) / fy;

    const T mx = (T(1) - alpha) * mxx;
    const T my = (T(1) - alpha) * myy;

    const T r2 = mx * mx + my * my;

    // Check if valid
    const bool is_valid = !static_cast<bool>(
        (alpha > T(0.5)) && (r2 >= T(1) / ((T(2) * alpha - T(1)))));

    const T xi2 = xi * xi;

    const T n = sqrt(T(1) + (T(1) - xi2) * (r2));
    const T m = (T(1) + r2);

    const T k = (xi + n) / m;
    const T z = k - xi;
    p3d[0] = k * mx / z;
    p3d[1] = k * my / z;
    p3d[2] = T(1);
    return is_valid;
  }
};

} // namespace argus
