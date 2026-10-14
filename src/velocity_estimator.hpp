// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).

#include <vector>
#include <iostream>
#include <Eigen/Core>
#include <chrono> 
#include <sophus/so3.h> 


class VelocityEstimator {
private:
    int buffer_len_;
    double t_duration_ = 5.0;
    
    FixedDeque<float> acc_buffer_;
    FixedDeque<float> gyro_buffer_;
    FixedDeque<float> ori_buffer_;
    FixedDeque<V3D> preint_vel_buffer_;

    OnnxModel model_;

    bool remove_g_;
    bool remove_bias_;
    bool rotation_invariant_;

    int update_mode_;
    bool update_rotation_via_velocity_;
    double acc_unit_ = 1.0;

    state_ikfom state_last_;
    double t_last_ = 0.0;
    double t_last_imu_ = 0.0;

    Eigen::Vector3d net_vel_;
    Eigen::Vector3d net_cov_;

    int num_inf_frames_ = 0;
    int num_kf_frames_ =0;
    double inference_elapsed_ = 0.0;
    double kf_elapsed_ = 0.0;

    double scale_ = 0.01;

    int num_outs_;
    std::vector<int> time_indices_;
    
    

public:
    bool init_state_last = false;
    bool ready_to_infer = false;
    static constexpr double G_CONSTANT = 9.810665;

    static constexpr int ESKF = 0;
    static constexpr int TSVD = 1;
    static constexpr int WEIGH = 2;
    

    VelocityEstimator(int buffer_len, bool remove_g, bool remove_bias,
                    bool rotation_invariant, int acc_unit,
                    int update_mode, bool update_rotation_via_velocity,
                    string model_path, bool use_gpu, int gpu_id, int num_threads,
                    double scale = 0.01)
    : buffer_len_(buffer_len),
    remove_g_(remove_g), remove_bias_(remove_bias),
    rotation_invariant_(rotation_invariant),
    update_mode_(update_mode), update_rotation_via_velocity_(update_rotation_via_velocity),
    acc_buffer_(static_cast<size_t>(buffer_len*3)),
    gyro_buffer_(static_cast<size_t>(buffer_len*3)),
    ori_buffer_(static_cast<size_t>(buffer_len*3)),
    preint_vel_buffer_(static_cast<size_t>(buffer_len)),
    model_(model_path, use_gpu, gpu_id, rotation_invariant, buffer_len, num_threads),
    scale_(scale)
    {
        acc_unit_ = (acc_unit == 0) ? 1.0 : G_CONSTANT;
        reset_out_indices();
        std::cout<<"ACC Unit : " << acc_unit_ << " m/s^2" << std::endl;
    }
    void init_state(double t, state_ikfom state){
        t_last_ = t;
        state_last_ = state;
        init_state_last = true;
    }

    std::tuple<Eigen::Vector3d, Eigen::Vector3d> get_inference_result() {
        return std::make_tuple(net_vel_, net_cov_);
    }

    std::tuple<std::vector<float>, std::vector<float>> get_all_outputs() {
        return model_.get_all_outputs();
    }

    Eigen::Quaterniond slerp_quaternion (const Eigen::Quaterniond& qi, const Eigen::Quaterniond& qf, double t_ratio) {
        Eigen::Quaterniond q_interp = qi.slerp(t_ratio, qf);
        return q_interp.normalized();
    }

    void inference_velocity() {
        auto start = std::chrono::high_resolution_clock::now();
        const auto &[vel, cov] = model_(acc_buffer_, gyro_buffer_, ori_buffer_);
        net_vel_ = vel;
        net_cov_ = cov;

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        inference_elapsed_+=elapsed.count()*1000.0;
        num_inf_frames_++;
    }
    void warmup() {
        model_.warmup();
    }

    void print_avg_elapsed_time() {
        if (num_inf_frames_ == 0) num_inf_frames_ = 1;
        if (num_kf_frames_ == 0) num_kf_frames_ = 1;

        
        double avg_inference_elapsed = inference_elapsed_ / static_cast<double> (num_inf_frames_);
        double avg_kf_elapsed = kf_elapsed_ / static_cast<double> (num_kf_frames_);
        double avg_total_elapsed = avg_inference_elapsed + avg_kf_elapsed;

        std::cout << "====Avg. Processing Time====" << std::endl;
        std::cout << "Total time          : "  << avg_total_elapsed << " ms" << std::endl;
        std::cout << "Partial Update time : "  << avg_kf_elapsed << " ms" << std::endl;
        std::cout << "Inference Time      : "  << avg_inference_elapsed << " ms" << std::endl;
    }

    void accumulate_buffers(MeasureGroup &meas, double t_curr, const state_ikfom &state_curr) {
        auto v_imu = meas.imu;
        const V3D &grav = state_last_.grav;
        const V3D &ba   = state_last_.ba;
        const V3D &bg   = state_last_.bg;

        const Eigen::Quaterniond &q_last = state_last_.rot;
        const Eigen::Quaterniond &q_curr = state_curr.rot;

        const size_t num_seq = v_imu.size();
        const double td = t_curr - t_last_;

        for (int i=0; i < num_seq; i++) {
            auto imu_msg = v_imu[i];
            const double t_meas = imu_msg->header.stamp.toSec();

            double gyro_x = imu_msg->angular_velocity.x;
            double gyro_y = imu_msg->angular_velocity.y;
            double gyro_z = imu_msg->angular_velocity.z;

            double acc_x = imu_msg->linear_acceleration.x;
            double acc_y = imu_msg->linear_acceleration.y;
            double acc_z = imu_msg->linear_acceleration.z;


            acc_x *= acc_unit_;
            acc_y *= acc_unit_;
            acc_z *= acc_unit_;
            
            const double t_ratio = (t_meas-t_last_) / td;
            Eigen::Quaterniond q_interp = slerp_quaternion(q_last, q_curr, t_ratio);
            Sophus::SO3 R(q_interp);

            const V3D so3 = R.log();    // Logarithm mapping : SO3 -> so3

            if (remove_g_) {
                const V3D g_body = R.inverse() * grav;
                acc_x += g_body.x();
                acc_y += g_body.y();
                acc_z += g_body.z();
            }

            if (remove_bias_) {
                acc_x -= ba.x();
                acc_y -= ba.y();
                acc_z -= ba.z();

                gyro_x -= bg.x();
                gyro_y -= bg.y();
                gyro_z -= bg.z();
            }

            gyro_buffer_.push_back(gyro_x);
            gyro_buffer_.push_back(gyro_y);
            gyro_buffer_.push_back(gyro_z);

            acc_buffer_.push_back(acc_x);
            acc_buffer_.push_back(acc_y);
            acc_buffer_.push_back(acc_z);

            ori_buffer_.push_back(so3.x());
            ori_buffer_.push_back(so3.y());
            ori_buffer_.push_back(so3.z());

            const double dt = (t_last_imu_ == 0.0) ? t_duration_ / static_cast<double>(buffer_len_) : (t_meas - t_last_imu_);
            t_last_imu_ = t_meas;
            preint_velocities(V3D(acc_x, acc_y, acc_z), R, grav, dt);
        }

        const int num_accum = gyro_buffer_.data().size();

        if (!ready_to_infer && num_accum == buffer_len_*3)   ready_to_infer = true;

        t_last_ = t_curr;
        state_last_ = state_curr;

    }

    void update_velocity_partial(const state_ikfom &state_point,
                                DegeneracyDetector &degeneracy_detector,
                                esekfom::esekf<state_ikfom, 12, input_ikfom> &kf) {
        auto start = std::chrono::high_resolution_clock::now();
        Eigen::Matrix3d VVt = M3D::Zero();
        const auto R_trans = state_point.rot.conjugate();

        const auto &[num_degeneracy_trans, num_degeneracy_rot] = degeneracy_detector.get_degeneracy_num();
        const auto &[degeneracy_matrix, degeneracy_vector] = degeneracy_detector.get_degeneracy_block();

        for (int i=0; i<num_degeneracy_trans; i++) {
            const Eigen::Vector3d &evec = R_trans*degeneracy_matrix.block<1, 3>(i, 0);
            VVt += evec*evec.transpose();
        }
        const M3D VVt_orth = M3D::Identity() - VVt;

        if (update_mode_ == ESKF) {
            const Eigen::Vector3d vel_body = R_trans*state_point.vel;
            const Eigen::Vector3d z = VVt * (net_vel_ - vel_body);

            Eigen::Matrix3d R = M3D::Zero();
            R.diagonal() << scale_*net_cov_.x()*net_cov_.x(),
                            scale_*net_cov_.y()*net_cov_.y(),
                            scale_*net_cov_.z()*net_cov_.z();
            
            R = VVt * R * VVt + VVt_orth * (scale_) * VVt_orth;

            Sophus::SO3 rot(R_trans);
            Eigen::Matrix<double, 3, 23> H = Eigen::Matrix<double, 3, 23>::Zero();;
            
            H.block<3,3> (0, 12) = rot.matrix();
            if (update_rotation_via_velocity_) H.block<3,3> (0, 3) << SKEW_SYM_MATRX(vel_body);
            H = VVt * H;

            kf.update_velocity_share(R, z, H);
        }
        else if (update_mode_ == TSVD){
            const Eigen::Matrix3d VVt_w = R_trans.conjugate()*VVt*R_trans;
            const Eigen::Matrix3d VVt_orth_w = R_trans.conjugate()*VVt_orth*R_trans;

            state_ikfom new_state = state_point;
            Eigen::Vector3d new_vel = VVt_orth_w*state_point.vel + VVt_w*(R_trans.conjugate()*net_vel_);

            new_state.vel = new_vel;
            kf.change_x(new_state);

            const auto P_old = kf.get_P();
            auto P_new = P_old;
            // double scale = 1;

            Eigen::Matrix3d P_meas = Eigen::Matrix3d::Zero();
            P_meas.diagonal()<< scale_*net_cov_.x()*net_cov_.x(),
                                scale_*net_cov_.y()*net_cov_.y(),
                                scale_*net_cov_.z()*net_cov_.z();
            P_new.block<3,3>(12, 12) = VVt_orth_w*P_old.block<3,3>(12, 12)*VVt_orth_w.transpose()
                                        + VVt_w*(R_trans.conjugate()*P_meas*R_trans)*VVt_w;
            kf.change_P(P_new);
        }
        else {
            const double degeneracy_ratio = degeneracy_detector.get_degeneracy_ratio();

            const Eigen::Vector3d net_vel_world = state_point.rot * net_vel_;
            const Eigen::Vector3d new_vel = (1-degeneracy_ratio)*net_vel_world +
                                                degeneracy_ratio*state_point.vel;

            state_ikfom new_state = state_point;
            new_state.vel = new_vel;
            kf.change_x(new_state);

            const auto P_old = kf.get_P();
            auto P_new = P_old;
            // double scale = 0.01;
            
            Eigen::Matrix3d P_meas = Eigen::Matrix3d::Zero();
            P_meas.diagonal()<< scale_*net_cov_.x()*net_cov_.x(),
                                scale_*net_cov_.y()*net_cov_.y(),
                                scale_*net_cov_.z()*net_cov_.z();
            P_new.block<3,3>(12, 12) = degeneracy_ratio*degeneracy_ratio*P_old.block<3,3>(12, 12) + \
                                        (1-degeneracy_ratio)*(1-degeneracy_ratio)*(R_trans.conjugate()*P_meas*R_trans);

        }
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        kf_elapsed_+=elapsed.count()*1000.0;
        num_kf_frames_++;
    }

    void reset_out_indices() {
        const int kernel_size = 7;
        const int pad_size = 3;
        const int stride_size = 3;
        const int interval_size = stride_size * stride_size;

        int s_idx = (kernel_size - pad_size) + stride_size * (kernel_size - 1 - pad_size) + 1;
        
        time_indices_.clear();
        for (int i = s_idx; i < buffer_len_; i += interval_size)    time_indices_.push_back(i); // 0-based index for
        
        const int num_outs_ = (buffer_len_ -1) / interval_size + 1;

        const int num_lack = num_outs_ - time_indices_.size();
        // Do not append redundant indices
        if (time_indices_.back() != buffer_len_-1 && num_lack > 0)  time_indices_.push_back(buffer_len_-1);
    }

    // Preprocessed acc
    void preint_velocities(const V3D &acc, const Sophus::SO3 &rot, const V3D &grav, const double dt) {
        V3D last_vel = preint_vel_buffer_.size() > 0 ? preint_vel_buffer_.back() : V3D::Zero();
        last_vel += (rot * acc + grav) * dt;

        preint_vel_buffer_.push_back(last_vel);
    }

    std::pair<std::vector<V3D>, std::vector<V3D>> extract_velocities_at() {
        int num_indices = time_indices_.size();
        std::vector<V3D> preinte_vels(num_indices);
        std::vector<V3D> neural_vels(num_indices);
        const auto &[flattend_vels, flattend_covs] = model_.get_all_outputs();

        #ifdef MP_EN
        omp_set_num_threads(MP_PROC_NUM);
        #pragma omp parallel for
        for (int i=0; i < num_indices; i++) {
            // Preinte
            const int idx = time_indices_[i];
            const V3D &vel_preinte = preint_vel_buffer_.data()[idx];
            preinte_vels[i] = vel_preinte;

            // Neural Network
            const auto vx_nn_body = flattend_vels[3*i];
            const auto vy_nn_body = flattend_vels[3*i+1];
            const auto vz_nn_body = flattend_vels[3*i+2];

            const V3D vel_nn_body(vx_nn_body, vy_nn_body, vz_nn_body);
            const auto so3_x = ori_buffer_.data()[3*i];
            const auto so3_y = ori_buffer_.data()[3*i+1];
            const auto so3_z = ori_buffer_.data()[3*i+2];

            const V3D so3(so3_x, so3_y, so3_z);
            Sophus::SO3 R = Sophus::SO3::exp(so3);

            neural_vels[i] = R*vel_nn_body; 
        }
        #endif

        return std::make_pair(preinte_vels, neural_vels);
    }
};