// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <deque> 
#include <iostream>
#include <cassert>
#include <sstream>
#include <string>
#include <sophus/so3.h>
#include <omp.h>
#include <Eigen/Core>
#include <cstdint> 
#include <thread>


std::string print_shape(const std::vector<std::int64_t> &v) {
  std::stringstream ss("");
  for (std::size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<std::int64_t> &v) {
  int total = 1;
  for (auto &i : v)
    total *= i;
  return total;
}

class OnnxModel {
private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_{nullptr};
    Ort::AllocatorWithDefaultOptions allocator_;
    Ort::MemoryInfo memory_info_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
    std::vector<std::int64_t> input_shapes_;
    std::vector<std::int64_t> output_shapes_;

    std::vector<const char *> input_names_char_;
    std::vector<const char *> output_names_char_;
    int gpu_id_;
    bool rotation_invariant_;
    int buffer_len_;
    std::vector<float> acc_flattened_;
    std::vector<float> gyro_flattened_;
    std::vector<float> ori_flattened_;
    std::vector<float> warmup_vector_;

    std::vector<float> velocities_;
    std::vector<float> covariances_;

    int num_threads_ = 1;

public:
  // Initialize OnnxModel
    OnnxModel(const std::string &model_file, bool use_cuda = false,
            int gpu_id = 0, bool rotation_invariant = false, int buffer_len=1000, int num_threads = -1)
        : env_(ORT_LOGGING_LEVEL_WARNING, "VelocityEstimator"),
        gpu_id_(gpu_id),
        rotation_invariant_(rotation_invariant),
        buffer_len_(buffer_len),
        memory_info_(Ort::MemoryInfo::CreateCpu(
                      OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault))
    {
        
        if (use_cuda) {
            std::cout<<"Use CUDA" << std::endl;
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(
                session_options_, gpu_id));
        }
        else {
            num_threads_ = std::thread::hardware_concurrency();
            if (num_threads > 0 && num_threads< num_threads_) num_threads_ = num_threads;
            session_options_.SetIntraOpNumThreads(num_threads_);
            std::cout<<"Use CPU, Num Threads : " << num_threads_ << std::endl;
        }

        session_ = Ort::Session(env_, model_file.c_str(), session_options_);
        setup();

        acc_flattened_.resize(buffer_len_*3);
        gyro_flattened_.resize(buffer_len_*3);
        ori_flattened_.resize(buffer_len_*3);
        warmup_vector_.resize(buffer_len_*3);
    }

    void setup() {
        for (std::size_t i = 0; i < session_.GetInputCount(); i++) {
            input_names_.emplace_back(
                session_.GetInputNameAllocated(i, allocator_).get());
            input_shapes_ =
                session_.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        }

        for (auto &s : input_shapes_) {
            if (s < 0) {
            s = 1;
            }
        }

        for (std::size_t i = 0; i < session_.GetOutputCount(); i++) {
            output_names_.emplace_back(
                session_.GetOutputNameAllocated(i, allocator_).get());
            output_shapes_ =
                session_.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        }

        for (auto &s : output_shapes_) {
            if (s < 0) {
            s = 1;
            }
        }

        assert(input_names_.size() == 3 && output_names_.size() == 2);

        // for (int i=0; i<input_names_.size(); i++){
        //     std::cout << "Input names: " << input_names_[i] << std::endl;
        //     std::cout << "Input shape: " << print_shape(input_shapes_) << std::endl;
        //     std::cout << "Total number of elements: " << calculate_product(input_shapes_)
        //                 << std::endl;
        // }
        // for (int i=0; i<output_names_.size(); i++){
        //     std::cout << "Output names: " << output_names_[i] << std::endl;
        //     std::cout << "Output shape: " << print_shape(output_shapes_) << std::endl;
        //     std::cout << "Total number of elements: "
        //                 << calculate_product(output_shapes_) << std::endl;
        // }

        input_names_char_.resize(input_names_.size());
        std::transform(std::begin(input_names_), std::end(input_names_),
                        std::begin(input_names_char_),
                        [&](const std::string &str) { return str.c_str(); });

        output_names_char_.resize(output_names_.size());
        std::transform(std::begin(output_names_), std::end(output_names_),
                        std::begin(output_names_char_),
                        [&](const std::string &str) { return str.c_str(); });
    }

    std::pair<std::vector<float>, std::vector<float>> get_all_outputs () {
        return std::make_pair(velocities_, covariances_);
    }

    void warmup() {
        Ort::Value warmup_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, warmup_vector_.data(), warmup_vector_.size(), input_shapes_.data(), input_shapes_.size());

        std::array<Ort::Value,3> input_tensors = {
            Ort::Value::CreateTensor<float>(memory_info_, warmup_vector_.data(), warmup_vector_.size(), input_shapes_.data(), input_shapes_.size()),
            Ort::Value::CreateTensor<float>(memory_info_, warmup_vector_.data(), warmup_vector_.size(), input_shapes_.data(), input_shapes_.size()),
            Ort::Value::CreateTensor<float>(memory_info_, warmup_vector_.data(), warmup_vector_.size(), input_shapes_.data(), input_shapes_.size()) };

        auto output_tensors =
            session_.Run(Ort::RunOptions{nullptr}, input_names_char_.data(),
                        input_tensors.data(), input_tensors.size(),
                        output_names_char_.data(), output_names_char_.size());
    }

    void preprocess(const FixedDeque<float> &acc_buffer,
                    const FixedDeque<float> &gyro_buffer,
                    const FixedDeque<float> &ori_buffer) {
        auto input_shape = input_shapes_;

        if (input_shape.size() != 3) {
            throw std::runtime_error("Invalid input shape. Expected format: [batch, "
                                    "sequence length, feature dimension]");
        }

        if (acc_buffer.data().size() != 3*buffer_len_) {
            throw std::runtime_error("Invalid Accereleration buffer seqence");
        }

        if (gyro_buffer.data().size() != 3*buffer_len_) {
            throw std::runtime_error("Invalid Gyro buffer seqence");
        }

        if (ori_buffer.data().size() != 3*buffer_len_) {
            throw std::runtime_error("Invalid Orientation buffer seqence");
        }

        std::copy(acc_buffer.data().begin(), acc_buffer.data().end(), acc_flattened_.begin());
        std::copy(gyro_buffer.data().begin(), gyro_buffer.data().end(), gyro_flattened_.begin());
        std::copy(ori_buffer.data().begin(), ori_buffer.data().end(), ori_flattened_.begin());

        if (rotation_invariant_) {
            const Eigen::Vector3d last_ori_vec(
                *(ori_flattened_.end() - 3),
                *(ori_flattened_.end() - 2),
                *(ori_flattened_.end() - 1)
            );
            Sophus::SO3 R_last = Sophus::SO3::exp(last_ori_vec);
            #ifdef MP_EN
            omp_set_num_threads(MP_PROC_NUM);
            #pragma omp parallel for
            for (int i=0; i < buffer_len_; i++) {
                const Eigen::Vector3d  so3_vec(
                    *(ori_flattened_.begin() + 3*i),
                    *(ori_flattened_.begin() + 3*i + 1),
                    *(ori_flattened_.begin() + 3*i + 2)
                );
                Sophus::SO3 R_global = Sophus::SO3::exp(so3_vec);
                Sophus::SO3 R_body = R_last.inverse() * R_global;
                const V3D so3 = R_body.log();
                ori_flattened_[3*i]   = so3.x();
                ori_flattened_[3*i+1] = so3.y();
                ori_flattened_[3*i+2] = so3.z();
            }
            #endif
        }
    }

    std::pair<Eigen::Vector3d, Eigen::Vector3d>
    operator()(const FixedDeque<float> &acc_buffer,
                const FixedDeque<float> &gyro_buffer,
                const FixedDeque<float> &ori_buffer) {
    try {
        preprocess(acc_buffer, gyro_buffer, ori_buffer);

        Ort::Value acc_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, acc_flattened_.data(), acc_flattened_.size(), input_shapes_.data(), input_shapes_.size());
        Ort::Value gyro_tensor = Ort::Value::CreateTensor<float>(
            memory_info_, gyro_flattened_.data(), gyro_flattened_.size(), input_shapes_.data(), input_shapes_.size());
        Ort::Value ori_tensor  = Ort::Value::CreateTensor<float>(
            memory_info_, ori_flattened_.data(), ori_flattened_.size(), input_shapes_.data(), input_shapes_.size());
        std::array<Ort::Value,3> input_tensors = { std::move(acc_tensor),
                                    std::move(gyro_tensor),
                                    std::move(ori_tensor) };

        auto output_tensors =
            session_.Run(Ort::RunOptions{nullptr}, input_names_char_.data(),
                        input_tensors.data(), input_tensors.size(),
                        output_names_char_.data(), output_names_char_.size());
        
        assert(output_tensors.size() == output_names_.size() &&
                output_tensors[0].IsTensor());

        auto *output_tensor =
            output_tensors.front().GetTensorMutableData<float>();
        
        auto& cov_tensor = output_tensors[0];
        auto& vel_tensor = output_tensors[1];

        auto cov_type_info = cov_tensor.GetTensorTypeAndShapeInfo();
        int64_t cov_size = cov_type_info.GetElementCount();
        float* cov_data = cov_tensor.GetTensorMutableData<float>();
        std::vector<float> cov(cov_data, cov_data + cov_size);

        auto vel_type_info = vel_tensor.GetTensorTypeAndShapeInfo();
        int64_t vel_size = vel_type_info.GetElementCount();
        
        float* vel_data = vel_tensor.GetTensorMutableData<float>();
        std::vector<float> vel(vel_data, vel_data + vel_size);

        auto data_shape = vel_type_info.GetShape();

        int rows = data_shape[1];
        int cols = 3;

        int start_idx = (rows - 1) * cols;  // 마지막 row 시작 index

        V3D last_vel(vel[start_idx + 0], vel[start_idx + 1], vel[start_idx + 2]);
        V3D last_cov(cov[start_idx + 0], cov[start_idx + 1], cov[start_idx + 2]);
        
        velocities_ = vel;
        covariances_ = cov;

        return std::make_pair(last_vel, last_cov);
    } catch (const Ort::Exception &exception) {
        std::cout << "ERROR running model inference: " << exception.what()
                << std::endl;
        exit(-1);
    }
    }
};