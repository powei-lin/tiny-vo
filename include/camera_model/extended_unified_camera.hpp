#pragma once
#include <Eigen/Core>
#include <string>
#include <vector>

namespace argus {
template <typename Scalar = double> struct ExtendedUnifiedCamera {
  static constexpr int N = 6;
  using VecN = Eigen::Matrix<Scalar, N, 1>;

  static std::string model_name() { return "eucm"; }
  static std::vector<std::string> param_name_list() {
    return {"fx", "fy", "cx", "cy", "alpha", "beta"};
  }

  VecN param;
  Eigen::Vector2i img_col_row;
  ExtendedUnifiedCamera() {
    param.setZero();
    img_col_row.setZero();
  }
  ExtendedUnifiedCamera(const VecN &_param) : param(_param) {
    img_col_row.setZero();
  }
  ExtendedUnifiedCamera(const VecN &_param, const Eigen::Vector2i _img_col_row)
      : param(_param), img_col_row(_img_col_row) {}

  VecN get_param() const { return param; }
  void set_param(const VecN &_param) { param = _param; }
  Eigen::Vector2i get_img_col_row() const { return img_col_row; }
  void set_img_col_row(const Eigen::Vector2i _img_col_row) {
    img_col_row = _img_col_row;
  }

  bool project(const Eigen::Matrix<Scalar, 3, 1> &p3d,
               Eigen::Matrix<Scalar, 2, 1> &p2d) const {
    return ExtendedUnifiedCamera::project(param.data(), p3d, p2d);
  }
  bool unproject(const Eigen::Matrix<Scalar, 2, 1> &p2d,
                 Eigen::Matrix<Scalar, 3, 1> &p3d) const {
    return ExtendedUnifiedCamera::unproject(param.data(), p2d, p3d);
  }

  ~ExtendedUnifiedCamera() {}

  template <typename T>
  static bool project(const T *param, const Eigen::Matrix<T, 3, 1> &p3d,
                      Eigen::Matrix<T, 2, 1> &p2d) {

    const T &fx = param[0];
    const T &fy = param[1];
    const T &cx = param[2];
    const T &cy = param[3];
    const T &alpha = param[4];
    const T &beta = param[5];

    const T &x = p3d[0];
    const T &y = p3d[1];
    const T &z = p3d[2];

    const T r2 = x * x + y * y;
    const T rho2 = beta * r2 + z * z;
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
    const T &beta = param[5];

    const T mx = (p2d[0] - cx) / fx;
    const T my = (p2d[1] - cy) / fy;

    const T r2 = mx * mx + my * my;
    const T gamma = T(1) - alpha;

    // Check if valid
    const bool is_valid = !static_cast<bool>(
        alpha > T(0.5) && (r2 >= T(1) / ((alpha - gamma) * beta)));

    const T tmp1 = (T(1) - alpha * alpha * beta * r2);
    const T tmp_sqrt = sqrt(T(1) - (alpha - gamma) * beta * r2);
    const T tmp2 = (alpha * tmp_sqrt + gamma);

    const T k = tmp1 / tmp2;

    p3d[0] = mx / k;
    p3d[1] = my / k;
    p3d[2] = T(1);
    return is_valid;
  }
};

using EUCMf = ExtendedUnifiedCamera<float>;
using EUCMd = ExtendedUnifiedCamera<double>;
} // namespace argus
