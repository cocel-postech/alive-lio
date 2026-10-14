// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).

#pragma once

#include <cstdio>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <array>
#include <unistd.h>

#include <Eigen/Core>

#include "degeneracy_detector.hpp"

/*
 * Per-frame console dashboard.
 *
 * The whole frame is rendered into a single string and written with one
 * std::cout call, then flushed. When stdout is a terminal the screen is
 * cleared first so the dashboard stays in place; when stdout is redirected
 * (e.g. to a log file) the frame is appended as plain text without ANSI codes.
 */
class ConsoleStatus {
public:
    // ---- frame info -------------------------------------------------------
    int    frame_idx        = 0;
    double lidar_end_time   = 0.0;
    double loop_ms          = 0.0;
    int    num_raw_points   = 0;
    int    num_down_points  = 0;
    int    num_effective    = 0;

    // ---- velocities (body frame, m/s) -------------------------------------
    Eigen::Vector3d rot_body_from_world_vel_lio   = Eigen::Vector3d::Zero(); // LiDAR-only ESKF
    Eigen::Vector3d vel_neural                    = Eigen::Vector3d::Zero(); // network output
    Eigen::Vector3d sigma_neural                  = Eigen::Vector3d::Zero(); // network std
    Eigen::Vector3d vel_fused                     = Eigen::Vector3d::Zero(); // after partial update
    Eigen::Matrix3d R_body_from_world             = Eigen::Matrix3d::Identity();

    bool network_enabled   = false;
    bool network_ready     = false;   // false while buffers are filling / warming up
    bool network_applied   = false;   // partial update actually ran this frame
    double t_degeneracy    = 0.0;     // accumulated degenerate time [s]

    ConsoleStatus() : is_tty_(isatty(fileno(stdout)) != 0) {}

    void set_lio_velocity(const Eigen::Vector3d &vel_world, const Eigen::Matrix3d &R_body_from_world_) {
        R_body_from_world = R_body_from_world_;
        rot_body_from_world_vel_lio = R_body_from_world * vel_world;
    }
    void set_fused_velocity(const Eigen::Vector3d &vel_world) {
        vel_fused = R_body_from_world * vel_world;
    }

    void print(DegeneracyDetector &dd) const {
        std::ostringstream os;
        os.setf(std::ios::fixed);

        if (is_tty_) os << "\033[H\033[2J";   // cursor home + clear screen

        const std::string rule(78, '-');

        // ---- header -------------------------------------------------------
        os << bold("ALIVE-LIO") << "  frame " << frame_idx
           << "  |  t = " << std::setprecision(3) << lidar_end_time << " s"
           << "  |  " << std::setprecision(1) << loop_ms << " ms/frame\n";
        os << rule << '\n';

        // ---- scan ---------------------------------------------------------
        const double eff_ratio = num_down_points > 0
            ? 100.0 * num_effective / static_cast<double>(num_down_points) : 0.0;
        os << "Scan        raw " << std::setw(6) << num_raw_points
           << "  ->  down " << std::setw(6) << num_down_points
           << "  ->  effective " << std::setw(6) << num_effective
           << " (" << std::setprecision(1) << eff_ratio << "%)\n";
        os << rule << '\n';

        // ---- degeneracy (translation) ------------------------------------
        const auto  flags = dd.get_degeneracy_flags();
        const auto &[low_c, high_c] = dd.get_cond_num_trans();
        const auto &[num_deg_trans, num_deg_rot] = dd.get_degeneracy_num();

        os << bold("Degeneracy (translation)") << "   ";
        if (num_deg_trans > 0) os << color_red("DEGENERATE  x" + std::to_string(num_deg_trans));
        else                   os << color_green("well-constrained");
        os << "    accumulated " << std::setprecision(2) << t_degeneracy << " s\n";

        os << std::setw(18) << std::left << "  axis" << std::right
           << std::setw(12) << "axis-1" << std::setw(12) << "axis-2" << std::setw(12) << "axis-3" << '\n';

        os << std::setw(18) << std::left << "  localizability" << std::right;
        for (int i = 0; i < 3; i++) os << flag_label(flags[i], 12);
        os << '\n';

        os << std::setw(18) << std::left << "  low-contrib" << std::right << std::setprecision(1);
        for (int i = 0; i < 3; i++) os << std::setw(12) << low_c[i];
        os << "   (K1=" << std::setprecision(0) << DegeneracyDetector::K1
           << ", K2=" << DegeneracyDetector::K2 << ")\n";

        os << std::setw(18) << std::left << "  high-contrib" << std::right << std::setprecision(1);
        for (int i = 0; i < 3; i++) os << std::setw(12) << high_c[i];
        os << "   (K2=" << std::setprecision(0) << DegeneracyDetector::K2
           << ", K3=" << DegeneracyDetector::K3 << ")\n";

        os << std::setw(18) << std::left << "  eigenvalue" << std::right << std::setprecision(1);
        for (int i = 0; i < 3; i++) os << std::setw(12) << dd.get_eval_trans(i);
        os << '\n';

        // Direction of each axis expressed in the body frame, so that it can be
        // compared with the body-frame velocities below.
        os << std::setw(18) << std::left << "  dir (body)" << std::right << std::setprecision(2);
        for (int i = 0; i < 3; i++) {
            const Eigen::Vector3d d = R_body_from_world * dd.get_evec_trans(i);
            os << ' ' << vec3_compact(d);
        }
        os << '\n';
        os << rule << '\n';

        // ---- velocity comparison -----------------------------------------
        os << bold("Velocity (body frame, m/s)") << "         "
           << std::setw(9) << "x" << std::setw(9) << "y" << std::setw(9) << "z" << std::setw(9) << "|v|" << '\n';

        print_vel_row(os, "  LIO   (LiDAR-only ESKF)", rot_body_from_world_vel_lio);

        if (!network_enabled) {
            os << "  Neural                      (disabled: localizability/en_network = false)\n";
        } else if (!network_ready) {
            os << "  Neural                      (warming up / filling IMU buffer)\n";
        } else {
            print_vel_row(os, "  Neural (network)", vel_neural);
            print_vel_row(os, "  Neural sigma", sigma_neural, /*norm=*/false);
            print_vel_row(os, "  Delta  (LIO - Neural)", rot_body_from_world_vel_lio - vel_neural);
        }

        print_vel_row(os, "  Fused  (published)", vel_fused);

        // Projection onto each eigen-axis: the component the network corrects.
        if (network_enabled && network_ready) {
            os << '\n';
            os << "  projection on axis    ";
            for (int i = 0; i < 3; i++) os << std::setw(16) << ("axis-" + std::to_string(i + 1));
            os << '\n';
            os << std::setprecision(3);
            os << "    LIO                 ";
            for (int i = 0; i < 3; i++) {
                const Eigen::Vector3d d = R_body_from_world * dd.get_evec_trans(i);
                os << proj_label(rot_body_from_world_vel_lio.dot(d), flags[i]);
            }
            os << '\n';
            os << "    Neural              ";
            for (int i = 0; i < 3; i++) {
                const Eigen::Vector3d d = R_body_from_world * dd.get_evec_trans(i);
                os << proj_label(vel_neural.dot(d), flags[i]);
            }
            os << '\n';
            os << "    Fused               ";
            for (int i = 0; i < 3; i++) {
                const Eigen::Vector3d d = R_body_from_world * dd.get_evec_trans(i);
                os << proj_label(vel_fused.dot(d), flags[i]);
            }
            os << '\n';
        }

        os << rule << '\n';
        if (network_enabled && network_ready) {
            os << "Network update: "
               << (network_applied ? color_yellow("APPLIED on degenerate axes") : "not applied (no degenerate axis)")
               << '\n';
        }

        std::cout << os.str() << std::flush;
    }

private:
    bool is_tty_;

    static std::string vec3_compact(const Eigen::Vector3d &v) {
        char buf[48];
        std::snprintf(buf, sizeof(buf), "[%5.2f,%5.2f,%5.2f]", v.x(), v.y(), v.z());
        return buf;
    }

    void print_vel_row(std::ostringstream &os, const std::string &label,
                       const Eigen::Vector3d &v, bool norm = true) const {
        os << std::setw(36) << std::left << label << std::right << std::setprecision(3);
        for (int i = 0; i < 3; i++) os << std::setw(9) << v[i];
        if (norm) os << std::setw(9) << v.norm();
        os << '\n';
    }

    // A projection value (16-char column), colored like the axis it belongs to.
    std::string proj_label(double value, int flag) const {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%16.3f", value);
        return colorize(buf, flag);
    }

    // Right-aligned flag text in a `width` column. Padding is done before the
    // ANSI codes are added because std::setw would count the escape bytes.
    std::string flag_label(int flag, int width) const {
        const char *txt = flag == DegeneracyDetector::FULL    ? "FULL"
                        : flag == DegeneracyDetector::PARTIAL ? "PARTIAL"
                        :                                       "NONE";
        std::string s = txt;
        if (static_cast<int>(s.size()) < width) s = std::string(width - s.size(), ' ') + s;
        return colorize(s, flag);
    }

    std::string colorize(const std::string &s, int flag) const {
        if (!is_tty_) return s;
        if (flag == DegeneracyDetector::FULL)    return color_green(s);
        if (flag == DegeneracyDetector::PARTIAL) return color_yellow(s);
        return color_red(s);
    }

    std::string bold(const std::string &s)         const { return is_tty_ ? "\033[1m"  + s + "\033[0m" : s; }
    std::string color_green(const std::string &s)  const { return is_tty_ ? "\033[32m" + s + "\033[0m" : s; }
    std::string color_yellow(const std::string &s) const { return is_tty_ ? "\033[33m" + s + "\033[0m" : s; }
    std::string color_red(const std::string &s)    const { return is_tty_ ? "\033[31m" + s + "\033[0m" : s; }
};
