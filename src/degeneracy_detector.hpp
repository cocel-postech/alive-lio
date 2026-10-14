// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).

#pragma once
#include <vector>
#include <iostream>
#include <Eigen/Core>


class DegeneracyDetector {
private:
    int degeneracy_detection_mode_;

    Eigen::Matrix<double, Eigen::Dynamic, 6> H_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> w_trans_;
    Eigen::Matrix<double, Eigen::Dynamic, 1> w_rots_;
    const double scaling_factor_ = 1000.0;
    const double min_weight_ = 1.0;
    const double max_weight_ = 10.0;

    Eigen::RowVector3d localizability_low_contrib_trans_;
    Eigen::RowVector3d localizability_high_contrib_trans_;
    Eigen::RowVector3d localizability_low_contrib_rots_;
    Eigen::RowVector3d localizability_high_contrib_rots_;

    int num_degeneracy_trans_ = 0;
    int num_degeneracy_rot_ = 0;

    std::array<int, 6> degeneracy_flags_ = {0, 0, 0, 0, 0, 0};

    Eigen::Matrix<double, 6, 6> degeneracy_matrix_;
    Eigen::Matrix<double, 6, 1> degeneracy_vector_;

    Eigen::Matrix3cd evecs_trans_;
    Eigen::Vector3cd evals_trans_;
    std::array<int,3> sorted_idx_trans_;

    Eigen::Matrix3cd evecs_rot_;
    Eigen::Vector3cd evals_rot_;
    std::array<int,3>  sorted_idx_rot_;

    double t_degeneracy_ = 0.0;
    double t_last_ = -1.0;


public:
    bool degeneracy_occured = false;

    static constexpr double K0 = 400.0;
    static constexpr double K1 = 250.0;
    static constexpr double K2 = 180.0;
    static constexpr double K3 = 35.0;

    static constexpr double TH_LOW_CONTRIB = 0.1736;
    static constexpr double TH_HIGH_CONTRIB = 0.707;

    static constexpr int FULL = 0;
    static constexpr int PARTIAL = 1;
    static constexpr int NONE = 2;

    static constexpr int XICP = 0;
    static constexpr int XICP_CORRELATION = 1;
    static constexpr int KALMAN_FILTER = 2;
    static constexpr int KALMAN_FILTER_CORRELATION = 3;

    DegeneracyDetector(){
    }

    std::tuple<std::array<int,3>, Eigen::Vector3cd, Eigen::Matrix3cd>
    compute_eigen(Eigen::Matrix3d &P){
        Eigen::EigenSolver<Eigen::Matrix3d> eigen_solver(P);
        Eigen::Vector3cd evals = eigen_solver.eigenvalues();
        Eigen::Matrix3cd evecs = eigen_solver.eigenvectors();

        Eigen::Vector3d evalsReal = evals.real();

        Eigen::Matrix3f::Index idxMin, idxMax;
        evalsReal.minCoeff(&idxMin);
        evalsReal.maxCoeff(&idxMax);
        int idxMid = 3 - idxMin - idxMax;

        std::array<int,3> indices = {idxMin, idxMid, idxMax};

        return std::make_tuple(indices, evals, evecs);

    }

    void init_information_matrix(int num_feature){
        H_.setZero(num_feature, 6);
        w_trans_.setZero(num_feature, 1);
        w_rots_.setZero(num_feature, 1);
    }

    void fill_H(int i, Eigen::Matrix<double, 1, 6> data) {
        H_.block<1,6>(i, 0) = data;
    }

    void fill_weight_trans(int i, double weight) {
        w_trans_(i) = std::clamp(weight/scaling_factor_, min_weight_, max_weight_);
    }
    void fill_weight_rot(int i, double weight) {
        w_rots_(i) = std::clamp(weight/scaling_factor_, min_weight_, max_weight_)*std::log(H_.block<1,3>(i, 3).norm()+1.0);
    }

    int categorize_localizability(const double low_contrib_value,
                                const double high_contrib_value,
                                const bool is_rot) {
        if ((high_contrib_value > K2 && low_contrib_value > K1) || (is_rot && low_contrib_value > K0))  return FULL;
        else if (high_contrib_value >= K3 || low_contrib_value >= K2) return PARTIAL;
        else    return NONE;
    }

    std::array<int, 6> get_degeneracy_flags() {
        return degeneracy_flags_;
    }

    std::tuple<Eigen::Matrix<double, 6, 6>, Eigen::Matrix<double, 6, 1>>  get_degeneracy_block() {
        return std::make_tuple(degeneracy_matrix_, degeneracy_vector_);
    }

    std::tuple<int, int> get_degeneracy_num(){
        return std::make_tuple(num_degeneracy_trans_, num_degeneracy_rot_);
    }

    std::tuple<Eigen::RowVector3d, Eigen::RowVector3d> get_cond_num_trans() {
        return std::make_tuple(localizability_low_contrib_trans_, localizability_high_contrib_trans_);
    }

    std::tuple<Eigen::RowVector3d, Eigen::RowVector3d> get_cond_num_rot() {
        return std::make_tuple(localizability_low_contrib_rots_, localizability_high_contrib_rots_);
    }

    void init_t_degeneracy() {
        t_degeneracy_ = 0.0;
        t_last_ = -1.0;
    }

    void update_t_degeneracy(const double t_curr) {
        double td = t_curr - t_last_;
        if (td < 0 || t_last_ == -1.0) td = 0.1;
        t_degeneracy_ += td;
        t_last_ = t_curr;
    }

    double get_t_degeneracy(){
        return t_degeneracy_;
    }

    double get_sorted_eval_tran(const int i) {
        return evals_trans_(sorted_idx_trans_[i]).real();
    }
    double get_sorted_eval_rot(const int i) {
        return evals_rot_(sorted_idx_rot_[i]).real();
    }

    // Unsorted accessors: index i matches degeneracy_flags_[i] and the
    // localizability vectors, i.e. column i of the eigen decomposition.
    double get_eval_trans(const int i) {
        return evals_trans_(i).real();
    }
    Eigen::Vector3d get_evec_trans(const int i) {
        return evecs_trans_.col(i).real();
    }

    double get_degeneracy_ratio() {
        return get_sorted_eval_tran(0)/get_sorted_eval_tran(2);
    }

    void print_degeneracy_trans_inform() {
        std::cout << "Translation degeneracy" << std::endl;
        for (int i = 0; i<3; i++) {
            if (degeneracy_flags_[i] == FULL) std:: cout <<"Full ";
            else if (degeneracy_flags_[i] == PARTIAL) std::cout <<"Partial ";
            else std::cout <<"None ";
        }
        std::cout<<std::endl;
    }

    void print_degeneracy_rot_inform() {
        std::cout << "Rotation degeneracy" << std::endl;
        for (int i = 0; i<3; i++) {
            if (degeneracy_flags_[i+3] == FULL) std:: cout <<"Full ";
            else if (degeneracy_flags_[i+3] == PARTIAL) std::cout <<"Partial ";
            else std::cout <<"None ";
        }
        std::cout<<std::endl;
    }

    void init_degeneracy_mode(int degeneracy_detection_mode) {
        degeneracy_detection_mode_ = degeneracy_detection_mode;
    }

    Eigen::Matrix<double, Eigen::Dynamic, 6> get_H() {
        return H_;
    }

    void compute_degeneracy_direction() {
        Eigen::Matrix<double, 6, 6> P = H_.transpose()*H_;
        Eigen::Matrix3d Ptrans;
        Eigen::Matrix3d Prot;

        if (degeneracy_detection_mode_ == XICP || degeneracy_detection_mode_ == KALMAN_FILTER) {
            Ptrans = P.block<3,3>(0,0);
            Prot = P.block<3,3>(3,3);
        }
        else if (degeneracy_detection_mode_ == XICP_CORRELATION || degeneracy_detection_mode_ == KALMAN_FILTER_CORRELATION) {
            const auto Prt = P.block<3,3>(3, 0);
            const auto Ptr = P.block<3,3>(0, 3);

            const auto Ptt = P.block<3,3>(0,0);
            const auto Prr = P.block<3,3>(3,3);
            
            Ptrans = Ptt - Ptr*Prr.inverse()*Prt;
            Prot = Prr - Prt*Ptt.inverse()*Ptr;
        }
        const auto &[inds_pos, evals_pos, evecs_pos] = compute_eigen(Ptrans);
        const auto &[inds_rot, evals_rot, evecs_rot] = compute_eigen(Prot);

        sorted_idx_trans_ = inds_pos;
        evecs_trans_ = evecs_pos;
        evals_trans_ = evals_pos;

        sorted_idx_rot_ = inds_rot;
        evecs_rot_ = evecs_rot;
        evals_rot_ = evals_rot;


        Eigen::MatrixXd H_trans = H_.block(0, 0, H_.rows(), 3);
        Eigen::MatrixXd H_rot   = H_.block(0, 3, H_.rows(), 3);

        Eigen::VectorXd norms = H_rot.rowwise().norm();
        H_rot = H_rot.array().colwise() / norms.array();

        const auto inform_mat_trans = (H_trans*evecs_pos).cwiseAbs().eval();
        const auto inform_mat_rot   = (H_rot*evecs_rot).cwiseAbs().eval();

        auto low_contrib_mat_trans = (inform_mat_trans.array() > TH_LOW_CONTRIB)
                    .select(inform_mat_trans.array(), 0.0)
                    .matrix().eval();
        auto low_contrib_mat_rot  = (inform_mat_rot.array() > TH_LOW_CONTRIB)
                    .select(inform_mat_rot.array(), 0.0)
                    .matrix().eval();
        auto high_contrib_mat_trans = (inform_mat_trans.array() > TH_HIGH_CONTRIB)
                    .select(inform_mat_trans.array(), 0.0)
                    .matrix().eval();
        auto high_contrib_mat_rot = (inform_mat_rot.array() > TH_HIGH_CONTRIB)
                    .select(inform_mat_rot.array(), 0.0)
                    .matrix().eval();

        if (degeneracy_detection_mode_ == KALMAN_FILTER || degeneracy_detection_mode_ == KALMAN_FILTER_CORRELATION) {
            low_contrib_mat_trans.array().colwise() *= w_trans_.array();
            low_contrib_mat_rot.array().colwise() *= w_trans_.array();
            high_contrib_mat_trans.array().colwise() *= w_rots_.array();
            high_contrib_mat_rot.array().colwise() *= w_rots_.array();
        }

        localizability_low_contrib_trans_ = low_contrib_mat_trans.colwise().sum();
        localizability_low_contrib_rots_ = low_contrib_mat_rot.colwise().sum();
        localizability_high_contrib_trans_ = high_contrib_mat_trans.colwise().sum();
        localizability_high_contrib_rots_ = high_contrib_mat_rot.colwise().sum();

        num_degeneracy_trans_ = 0;
        num_degeneracy_rot_ = 0;
        degeneracy_occured =false;
        
        for (int i = 0; i<3; i++) {
            degeneracy_flags_[i] = categorize_localizability(localizability_low_contrib_trans_[i],
                                                            localizability_high_contrib_trans_[i], false);
            degeneracy_flags_[i+3] = categorize_localizability(localizability_low_contrib_rots_[i],
                                                            localizability_high_contrib_rots_[i], true);

            if (degeneracy_flags_[i] == NONE) num_degeneracy_trans_++;
            if (degeneracy_flags_[i+3] == NONE) num_degeneracy_rot_++;
        }

        if (num_degeneracy_trans_ > 0) degeneracy_occured = true;

        degeneracy_matrix_.setZero();
        degeneracy_vector_.setZero();
        int j = 0;

        for (int i = 0; i<3; i++) {
            if (degeneracy_flags_[i] == NONE) {
                degeneracy_matrix_.block<1, 3>(j, 0) = evecs_pos.col(i).real().transpose();
                j++;
            }
        }
        for (int i = 0; i<3; i++) {
            if (degeneracy_flags_[i+3] == NONE) {
                degeneracy_matrix_.block<1, 3>(j, 3) = evecs_rot.col(i).real().transpose();
                j++;
            }
        }

    }
};