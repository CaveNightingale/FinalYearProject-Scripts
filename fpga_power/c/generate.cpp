#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <ostream>
#include <print>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

// clangd: -std=c++23

using std::abort;
using std::clamp;
using std::cos;
using std::fstream;
using std::ios;
using std::is_same_v;
using std::log;
using std::max;
using std::min;
using std::optional;
using std::ostream;
using std::pow;
using std::print;
using std::println;
using std::random_device;
using std::round;
using std::sin;
using std::sqrt;
using std::string;
using std::strncmp;
using std::tuple;
using std::vector;

struct PCG64 {
  using u64 = std::uint64_t;
  using u128 = unsigned __int128;

  static constexpr u128 multiplier =
      (static_cast<u128>(0x2360ed051fc65da4ULL) << 64) | 0x4385df649fccf645ULL;

  u128 state = 0;
  u128 increment = 0;

  explicit PCG64(u64 seed = 0x853c49e6748fea9bULL,
                 u64 seq = 0xda3e39cb94b95bdbULL) {
    seed_rng(seed, seq);
  }

  void seed_rng(u64 seed, u64 seq = 0xda3e39cb94b95bdbULL) {
    state = 0;
    increment = (static_cast<u128>(seq) << 1u) | 1u;
    next_u64();
    state += static_cast<u128>(seed);
    next_u64();
  }

  static u64 rotr64(u64 value, unsigned rot) {
    rot &= 63u;
    if (rot == 0u) {
      return value;
    }
    return (value >> rot) | (value << ((64u - rot) & 63u));
  }

  u64 next_u64() {
    const u128 old_state = state;
    state = old_state * multiplier + increment;

    const u64 xsl = static_cast<u64>((old_state >> 64u) ^ old_state);
    const unsigned int rot = static_cast<unsigned int>(old_state >> 122u);
    return rotr64(xsl, rot);
  }
};

template <int ExponentBit, int MantissaBit,
          int ExponentBias = (1 << (ExponentBit - 1)) - 1>
struct QFloatType {
  // The extended type is a minimal type that is capable of representing all
  // subnormal values of the original type as normal values
  using Extended = QFloatType<ExponentBit + 1, MantissaBit,
                              ExponentBias + (1 << ExponentBit)>;

  static uint32_t _max_abs() { return (1u << (ExponentBit + MantissaBit)) - 1; }

  static uint32_t quantize(double value) {
    uint64_t value_bits = std::bit_cast<uint64_t>(value);
    bool sign = ((value_bits >> 63) & 1) && value != 0; // Handle -0.0
    int64_t exponent = ((value_bits >> 52) & 0x7FF) - 1023 + ExponentBias;
    int64_t mantissa = value_bits & 0xF'FFFF'FFFF'FFFF;
    bool truncated = 0;
    if (exponent > (1 << ExponentBit) - 1) { // Saturate to max
      exponent = (1 << ExponentBit) - 1;
      mantissa = (1 << MantissaBit) - 1;
    } else if (exponent <= 0) { // Subnormal
      int shift = 1 - exponent + (52 - MantissaBit);
      if (shift <= 53) {
        truncated = ((mantissa | (1ULL << 52)) >> (shift - 1)) & 1;
        mantissa = (mantissa | (1ULL << 52)) >> shift;
      } else {
        truncated = 0;
        mantissa = 0;
      }
      exponent = 0;
    } else {
      int shift = 52 - MantissaBit;
      truncated = (mantissa >> (shift - 1)) & 1;
      mantissa = mantissa >> shift;
    }
    uint32_t quantized_abs = min<uint32_t>(
        (exponent << MantissaBit | mantissa) + truncated, _max_abs());
    uint32_t quantized = quantized_abs | (sign << (ExponentBit + MantissaBit));
    return quantized;
  }

  static double dequantize(uint32_t quantized) {
    bool sign = (quantized >> (ExponentBit + MantissaBit)) & 1;
    int32_t exponent = (quantized >> MantissaBit) & ((1 << ExponentBit) - 1);
    int32_t mantissa = quantized & ((1 << MantissaBit) - 1);
    if (exponent == 0) {
      return (sign ? -1.0 : 1.0) *
             (mantissa / static_cast<double>(1ULL << MantissaBit)) *
             pow(2.0, 1 - ExponentBias);
    } else {
      return (sign ? -1.0 : 1.0) *
             (1.0 + mantissa / static_cast<double>(1ULL << MantissaBit)) *
             pow(2.0, exponent - ExponentBias);
    }
  }

  static double max_value() { return dequantize(_max_abs()); }

  static double min_value() { return -max_value(); }

  static constexpr bool is_int() { return false; }

  static constexpr int bit_width() { return 1 + ExponentBit + MantissaBit; }

  static string name() {
    return "e" + std::to_string(ExponentBit) + "m" +
           std::to_string(MantissaBit);
  }
};

template <int Bit> struct QIntType {
  // Int type does not have subnormal values, so the extended type is itself
  using Extended = QIntType<Bit>;

  static uint32_t _max_abs() { return (1u << Bit) - 1; }

  static uint32_t quantize(double value) {
    int64_t quantized = round(value);
    return clamp<int64_t>(quantized, 0, _max_abs());
  }

  static double dequantize(uint32_t quantized) { return quantized; }

  static double max_value() { return _max_abs(); }

  static double min_value() { return 0.0; }

  static constexpr bool is_int() { return true; }

  static constexpr int bit_width() { return Bit; }

  static string name() { return "int" + std::to_string(Bit); }
};

using QE5M10 = QFloatType<5, 10>;
using QE4M3 = QFloatType<4, 3>;
using QE2M1 = QFloatType<2, 1>;
using QINT8 = QIntType<8>;
using QINT4 = QIntType<4>;
using QE6M10B47 = QFloatType<6, 10, 47>;
using QE5M3B23 = QFloatType<5, 3, 23>;

template <typename T> struct Matrix {
  int rows = 0;
  int cols = 0;

  vector<T> data;

  Matrix(int r, int c) : rows(r), cols(c), data(r * c) {}

  T &operator[](int r, int c) { return data[r * cols + c]; }
  T operator[](int r, int c) const { return data[r * cols + c]; }

  Matrix<T> mul(const Matrix<T> &other) const {
    assert(cols == other.rows);
    Matrix<T> result(rows, other.cols);
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < other.cols; j++) {
        T sum = 0;
        for (int k = 0; k < cols; k++) {
          sum += (*this)[i, k] * other[k, j];
        }
        result[i, j] = sum;
      }
    }
    return result;
  }

  static Matrix<T> zeros(int r, int c) { return Matrix<T>(r, c); }
  static Matrix<T> ones(int r, int c) {
    Matrix<T> mat(r, c);
    std::fill(mat.data.begin(), mat.data.end(), static_cast<T>(1));
    return mat;
  }

  static Matrix<double> random_normal(int r, int c, PCG64 &rng) {
    Matrix<double> mat(r, c);
    for (int i = 0; i < r * c; i += 2) {
      double u1 = rng.next_u64() * pow(2.0, -64);
      double u2 = rng.next_u64() * pow(2.0, -64);
      double z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
      double z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
      mat.data[i] = z0;
      if (i + 1 < r * c) {
        mat.data[i + 1] = z1;
      }
    }
    return mat;
  }

  static Matrix<double> random_laplace(int r, int c, PCG64 &rng) {
    Matrix<double> mat(r, c);
    for (int i = 0; i < r * c; i++) {
      double u = rng.next_u64() * pow(2.0, -64) - 0.5;
      mat.data[i] = -std::copysign(log(1 - 2 * std::abs(u)), u);
    }
    return mat;
  }

  tuple<double, double, double> error(const Matrix<double> &other) const {
    assert(rows == other.rows && cols == other.cols);
    double l1_error = 0, l2_error = 0, linf_error = 0;
    for (int i = 0; i < rows * cols; i++) {
      double error = std::abs(data[i] - other.data[i]);
      l1_error += error;
      l2_error += error * error;
      linf_error = max(linf_error, error);
    }
    return tuple(l1_error / (rows * cols), sqrt(l2_error / (rows * cols)),
                 linf_error);
  }
};

template <typename ActivationQType, typename WeightQType>
tuple<Matrix<uint16_t>, Matrix<uint16_t>,
      Matrix<uint16_t>> // quants, scales, zeros
quantize_weight(const Matrix<double> &mat) {
  static_assert(ActivationQType::bit_width() <= 16 &&
                !ActivationQType::is_int() &&
                (WeightQType::bit_width() < ActivationQType::bit_width() ||
                 is_same_v<WeightQType, ActivationQType>));
  int r = mat.rows;
  int c = mat.cols;
  assert(r % 64 == 0);
  auto quants = Matrix<uint16_t>::zeros(r, c);
  auto scales = Matrix<uint16_t>::zeros(r / 64, c);
  auto zeros = Matrix<uint16_t>::zeros(r / 64, c);
  if constexpr (is_same_v<WeightQType, ActivationQType>) {
    // No scale and zero point needed
    for (int i = 0; i < r; i++) {
      for (int j = 0; j < c; j++) {
        quants[i, j] = WeightQType::quantize(mat[i, j]);
      }
    }
    uint16_t one = ActivationQType::Extended::quantize(1.0);
    for (int i = 0; i < r / 64; i++) {
      for (int j = 0; j < c; j++) {
        scales[i, j] = one;
      }
    }
  } else if constexpr (!WeightQType::is_int()) {
    // Float quantization are symmetric
    for (int i = 0; i < r; i += 64) {
      for (int j = 0; j < c; j++) {
        double max_abs = 0.0;
        for (int k = 0; k < 64; k++) {
          max_abs = max(max_abs, std::abs(mat[i + k, j]));
        }
        uint16_t scale_raw = ActivationQType::quantize(
            max_abs / (WeightQType::max_value() - WeightQType::min_value()) *
            2);
        double scale = ActivationQType::dequantize(scale_raw);
        scales[i / 64, j] = ActivationQType::Extended::quantize(scale);
        for (int k = 0; k < 64; k++) {
          quants[i + k, j] = WeightQType::quantize(mat[i + k, j] / scale);
        }
      }
    }
  } else {
    // Int quantization are asymmetric
    for (int i = 0; i < r; i += 64) {
      for (int j = 0; j < c; j++) {
        double min_val = 0.0, max_val = 0.0;
        for (int k = 0; k < 64; k++) {
          min_val = min(min_val, mat[i + k, j]);
          max_val = max(max_val, mat[i + k, j]);
        }
        uint16_t scale_raw = ActivationQType::quantize(
            (max_val - min_val) /
            (WeightQType::max_value() - WeightQType::min_value()));
        double scale = ActivationQType::dequantize(scale_raw);
        scales[i / 64, j] = ActivationQType::Extended::quantize(scale);
        zeros[i / 64, j] = WeightQType::quantize(-min_val / scale);
        double zero = WeightQType::dequantize(zeros[i / 64, j]);
        for (int k = 0; k < 64; k++) {
          quants[i + k, j] =
              WeightQType::quantize(mat[i + k, j] / scale + zero);
        }
      }
    }
  }
  return tuple(quants, scales, zeros);
}

template <typename ActivationQType, typename WeightQType>
Matrix<double> dequantize_weight(const Matrix<uint16_t> &quants,
                                 const Matrix<uint16_t> &scales,
                                 const Matrix<uint16_t> &zeros) {
  int r = quants.rows;
  int c = quants.cols;
  assert(c % 64 == 0 && scales.rows == r / 64 && zeros.rows == r / 64 &&
         scales.cols == c && zeros.cols == c);
  Matrix<double> mat(r, c);
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < r; j += 64) {
      double scale = ActivationQType::Extended::dequantize(scales[j / 64, i]);
      double zero = WeightQType::dequantize(zeros[j / 64, i]);
      for (int k = 0; k < 64; k++) {
        mat[j + k, i] =
            (WeightQType::dequantize(quants[j + k, i]) - zero) * scale;
      }
    }
  }
  return mat;
}

template <typename ActivationQType>
Matrix<uint16_t> quantize_activation(const Matrix<double> &mat) {
  int r = mat.rows;
  int c = mat.cols;
  auto quants = Matrix<uint16_t>::zeros(r, c);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      quants[i, j] = ActivationQType::quantize(mat[i, j]);
    }
  }
  return quants;
}

template <typename ActivationQType>
Matrix<double> dequantize_activation(const Matrix<uint16_t> &quants) {
  int r = quants.rows;
  int c = quants.cols;
  auto mat = Matrix<double>::zeros(r, c);
  for (int i = 0; i < r; i++) {
    for (int j = 0; j < c; j++) {
      mat[i, j] = ActivationQType::dequantize(quants[i, j]);
    }
  }
  return mat;
}

template <typename ActivationQType, typename WeightQType> void generate_case() {
  random_device rd;
  uint64_t seed = rd();
  uint64_t seq = rd();
  PCG64 rng(seed, seq);
  string case_name =
      "test_w" + WeightQType::name() + "_a" + ActivationQType::name();
  println("Generating case {} with seed {} and seq {}", case_name, seed, seq);
  fstream file(case_name + ".txt", ios::out);
  println(file, "pcg64_seed {} {}", seed, seq);
  println(file, "batch_size {}", 24);
  println(file, "input_dtype {}", ActivationQType::name());
  println(file, "weight_dtype {}", WeightQType::name());
  println(file, "output_dtype {}", QE5M10::name());
  auto x = Matrix<double>::random_laplace(24, 4096, rng);
  auto w = Matrix<double>::random_normal(4096, 22016, rng);
  auto y = x.mul(w);
  auto x_quants = quantize_activation<ActivationQType>(x);
  auto [w_quants, w_scales, w_zeros] =
      quantize_weight<ActivationQType, WeightQType>(w);
  auto x_q = dequantize_activation<ActivationQType>(x_quants);
  auto w_q = dequantize_weight<ActivationQType, WeightQType>(w_quants, w_scales,
                                                             w_zeros);
  auto y_q = x_q.mul(w_q);
  auto y_quants = quantize_activation<QE5M10>(y_q);
  auto [x_mae, x_rmse, x_linf] = x.error(x_q);
  auto [w_mae, w_rmse, w_linf] = w.error(w_q);
  auto [y_mae, y_rmse, y_linf] =
      y.error(dequantize_activation<QE5M10>(y_quants));
  println(file, "input_activation_error mae {:.9f} rmse {:.9f} linf {:.9f}",
          x_mae, x_rmse, x_linf);
  println(file, "weight_error mae {:.9f} rmse {:.9f} linf {:.9f}", w_mae,
          w_rmse, w_linf);
  println(file, "output_activation_error mae {:.9f} rmse {:.9f} linf {:.9f}",
          y_mae, y_rmse, y_linf);
  auto act_width = ActivationQType::bit_width();
  auto weight_width = WeightQType::bit_width();
  println(file, "input_activation column_major {}x{}", x_quants.rows,
          x_quants.cols);
  for (int i = 0; i < x_quants.cols; i++) {
    for (int j = 0; j < x_quants.rows; j++) {
      if (j != x_quants.rows - 1) {
        print(file, "{:0{}x} ", x_quants[j, i], act_width / 4);
      } else {
        println(file, "{:0{}x}", x_quants[j, i], act_width / 4);
      }
    }
  }
  for (int i = 0; i < w_quants.cols / 16; i++) {
    if constexpr (is_same_v<WeightQType, ActivationQType>) {
      println(file, "weight_tile {} row_major {}x{}", i, w_quants.rows, 16);
    } else if constexpr (!WeightQType::is_int()) {
      println(file, "weight_tile {} row_major {}x{} {}x{}", i, w_scales.rows,
              16, w_quants.rows, 16);
    } else {
      println(file, "weight_tile {} row_major {}x{} {}x{} {}x{}", i,
              w_scales.rows, 16, w_zeros.rows, 16, w_quants.rows, 16);
    }
    if constexpr (!is_same_v<WeightQType, ActivationQType>) {
      for (int j = 0; j < w_scales.rows; j++) {
        for (int k = 0; k < 16; k++) {
          if (k != 15) {
            print(file, "{:0{}x} ", w_scales[j, k + i * 16], act_width / 4);
          } else {
            println(file, "{:0{}x}", w_scales[j, k + i * 16], act_width / 4);
          }
        }
      }
    }
    if constexpr (WeightQType::is_int()) {
      for (int j = 0; j < w_zeros.rows; j++) {
        for (int k = 0; k < 16; k++) {
          if (k != 15) {
            print(file, "{:0{}x} ", w_zeros[j, k + i * 16], weight_width / 4);
          } else {
            println(file, "{:0{}x}", w_zeros[j, k + i * 16], weight_width / 4);
          }
        }
      }
    }
    for (int j = 0; j < w_quants.rows; j++) {
      for (int k = 0; k < 16; k++) {
        if (k != 15) {
          print(file, "{:0{}x} ", w_quants[j, k + i * 16], weight_width / 4);
        } else {
          println(file, "{:0{}x}", w_quants[j, k + i * 16], weight_width / 4);
        }
      }
    }
  }
  println(file, "output_activation column_major {}x{}", y_quants.rows,
          y_quants.cols);
  for (int i = 0; i < y_quants.cols; i++) {
    for (int j = 0; j < y_quants.rows; j++) {
      if (j != y_quants.rows - 1) {
        print(file, "{:04x} ", y_quants[j, i]);
      } else {
        println(file, "{:04x}", y_quants[j, i]);
      }
    }
  }
}

void generate_tests() {
  generate_case<QE5M10, QE5M10>();
  generate_case<QE5M10, QINT8>();
  generate_case<QE5M10, QE4M3>();
  generate_case<QE5M10, QINT4>();
  generate_case<QE5M10, QE2M1>();
  generate_case<QE4M3, QE4M3>();
  generate_case<QE4M3, QINT4>();
  generate_case<QE4M3, QE2M1>();
}

int main() {
  generate_tests();
  return 0;
}