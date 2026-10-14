// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).
//
// This file is derived from PV-LIO (https://github.com/HViktorTsoi/PV-LIO),
// FAST-LIO2 (https://github.com/hku-mars/FAST_LIO) and VoxelMap
// (https://github.com/hku-mars/VoxelMap), all GPL-2.0. The original
// copyright notices below are retained.

#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP
#include "common_lib.h"
#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
//#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <tsl/robin_map.h>
#include <cmath>
#include <array>

static int plane_id = 0;
const int EUCLID = 0;
const int PROB = 1;
const int FUSING = 2;

// a point to plane matching structure
typedef struct ptpl {
  V3D point;
  V3D point_world;
  V3D normal;
  V3D center;
  Eigen::Matrix<double, 6, 6> plane_cov;
  double d;
  int layer;
  Eigen::Matrix3d cov_lidar;
} ptpl;

typedef struct ptpo {
  V3D point;
  V3D point_world;
  V3D point_center;
  M3D point_cov;
  M3D cov_lidar;
  size_t num_neighbors;
  size_t num_candidate_voxels;
} ptpo;

// 3D point with covariance
typedef struct pointWithCov {
  V3D point;
  V3D point_world;
  M3D cov;
  M3D cov_lidar;
} pointWithCov;

typedef struct Plane {
  V3D center = V3D::Zero();
  V3D normal;
  V3D y_normal;
  V3D x_normal;
  M3D covariance = M3D::Zero();
  Eigen::Matrix<double, 6, 6> plane_cov;
  float radius = 0;
  float min_eigen_value = 1;
  float mid_eigen_value = 1;
  float max_eigen_value = 1;
  float d = 0;
  int points_size = 0;

  int num_out_of_plane = 0;

  bool is_plane = false;
  bool is_init = false;
  int id;
  // is_update and last_update_points_size are only for publish plane
  bool is_update = false;
  int last_update_points_size = 0;
  bool update_enable = true;
} Plane;

class VOXEL_LOC {
public:
  int64_t x, y, z;

  VOXEL_LOC(const V3D &point) {
    x = static_cast<int>(std::floor(point.x()));
    y = static_cast<int>(std::floor(point.y()));
    z = static_cast<int>(std::floor(point.z()));
  }

  bool operator==(const VOXEL_LOC &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

struct VoxelHash {
    size_t operator()(const VOXEL_LOC &voxel) const {
        const uint32_t *vec = reinterpret_cast<const uint32_t *>(&voxel);
        return (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
    }
};


using DataAssociationFunction = void (*)(const pointWithCov&, const std::vector<pointWithCov>&, pointWithCov&, double&, double&, bool&, const double);
class OctoTree {
public:
  std::vector<pointWithCov> temp_points_; // all points in an octo tree
  std::vector<pointWithCov> new_points_;  // new points in an octo tree
  Plane *plane_ptr_;
  int max_layer_;
  bool indoor_mode_;
  int layer_;
  int octo_state_; // 0 is end of tree, 1 is not
  OctoTree *leaves_[8];
  double voxel_center_[3]; // x, y, z
  pointWithCov pv_center;
  std::vector<int> layer_point_size_;
  float quater_length_;
  float planer_threshold_;
  int max_plane_update_threshold_;
  int update_size_threshold_;
  int all_points_num_;
  int new_points_num_;
  int max_points_size_;
  
  int update_points_size_;
  int max_mature_points_size_;
  // By SJKim
  double octo_size_;
  int voxel_size_;            

  bool init_octo_;
  bool update_enable_;
  OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
           int max_point_size,
           int max_mature_points_size,
           float planer_threshold,
           int voxel_size)
      : max_layer_(max_layer), layer_(layer),
        layer_point_size_(layer_point_size), max_points_size_(max_point_size),
        max_mature_points_size_(max_mature_points_size),
        update_points_size_(max_point_size),
        planer_threshold_(planer_threshold),
        voxel_size_(voxel_size) {
    temp_points_.clear();
    octo_state_ = 0;
    new_points_num_ = 0;
    all_points_num_ = 0;
    // when new points num > 5, do a update
    update_size_threshold_ = 5;
    init_octo_ = false;
    update_enable_ = true;
    max_plane_update_threshold_ = layer_point_size_[layer_];

  
    // By SJKim
    octo_size_ = voxel_size_/std::pow(2.0, layer_);

    for (int i = 0; i < 8; i++) {
      leaves_[i] = nullptr;
    }
    plane_ptr_ = new Plane;
  }
  ~OctoTree() {
    std::vector<pointWithCov>().swap(temp_points_);
    std::vector<pointWithCov>().swap(new_points_);
    if (plane_ptr_) {
        delete plane_ptr_;
        plane_ptr_ = nullptr;
    }
    for (int i = 0; i < 8; i++) {
        if (leaves_[i]) {
            delete leaves_[i];
            leaves_[i] = nullptr;
        }
    }
  }

  /**
   * 주어진 점 집합(points)을 기반으로 평면 모델(Plane)의 파라미터를 초기화한다.
   * 평면의 중심(center), 공분산(covariance), 고유값 및 고유벡터를 계산하여,
   * 평면의 법선(normal)과 기타 평면 관련 파라미터(예: d, radius 등)를 구한다.
   *
   * @param points  각 점과 해당 공분산 정보를 담은 벡터 (pointWithCov 타입)
   * @param plane   초기화할 평면 구조체(Plane 포인터)
   */
  void init_plane(const std::vector<pointWithCov> &points, Plane *plane, const int new_points_num) {
    plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();

    if (plane->is_init) {
      plane->covariance += plane->center * plane->center.transpose();
      plane->covariance *= plane->points_size;

      plane->center *= plane->points_size;
    }

    plane->normal = V3D::Zero();
    plane->points_size = points.size();
    plane->radius = 0;

    for (int i = points.size() - new_points_num; i < points.size(); ++i) {
      const auto &point = points[i].point;
      plane->center += point;
      plane->covariance += point * point.transpose();
    }
    // 평균(centroid) 계산
    plane->center = plane->center / plane->points_size;
    // 샘플 공분산 행렬 계산: E[xx^T] - (center)(center)^T
    plane->covariance = plane->covariance / plane->points_size -
                        plane->center * plane->center.transpose();

    // 공분산 행렬에 대한 고유값 분해 수행
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
    Eigen::Matrix3cd evecs = es.eigenvectors();
    Eigen::Vector3cd evals = es.eigenvalues();
    Eigen::Vector3d evalsReal;
    evalsReal = evals.real();

    // 최소, 최대 고유값의 인덱스를 구하고 나머지 중간 인덱스를 계산 (0,1,2의 합이 3)
    Eigen::Matrix3f::Index evalsMin, evalsMax;
    evalsReal.rowwise().sum().minCoeff(&evalsMin);
    evalsReal.rowwise().sum().maxCoeff(&evalsMax);
    int evalsMid = 3 - evalsMin - evalsMax;

    // 각 인덱스에 해당하는 고유벡터 추출 (실수부)
    Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
    Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
    Eigen::Vector3d evecMax = evecs.real().col(evalsMax);

    

    // plane covariance calculation
    // 평면의 오차 공분산 계산을 위한 Jacobian 행렬의 일부
    // VoxelMap 논문의 수식 (7)
    Eigen::Matrix3d J_Q;
    J_Q << 1.0 / plane->points_size, 0, 0, 
           0, 1.0 / plane->points_size, 0, 
           0, 0, 1.0 / plane->points_size;

    bool estimate_plane = (evalsReal(evalsMin) < planer_threshold_);
    // 최소 고유값이 임계치보다 작으면 평면성이 충분하다고 판단하여 추가 계산 수행
    if (estimate_plane) {
      // (추후 활용할 수 있는 인덱스 및 임시 행렬 벡터 선언)
      std::vector<int> index(points.size());
      std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());

      for (int i = 0; i < points.size(); i++) {
        Eigen::Matrix<double, 6, 3> J;
        Eigen::Matrix3d F;
        for (int m = 0; m < 3; m++) {
          if (m != (int)evalsMin) {
            Eigen::Matrix<double, 1, 3> F_m =
                (points[i].point - plane->center).transpose() /
                ((plane->points_size) * (evalsReal[evalsMin] - evalsReal[m])) *
                (evecs.real().col(m) * evecs.real().col(evalsMin).transpose() +
                 evecs.real().col(evalsMin) * evecs.real().col(m).transpose());
            F.row(m) = F_m;
          } else {
            Eigen::Matrix<double, 1, 3> F_m;
            F_m << 0, 0, 0;
            F.row(m) = F_m;
          }
        }
        J.block<3, 3>(0, 0) = evecs.real() * F;
        J.block<3, 3>(3, 0) = J_Q;
        // 각 점의 공분산(points[i].cov)을 Jacobian을 통해 전파하여 평면 오차 공분산에 누적
        plane->plane_cov += J * points[i].cov * J.transpose();
      }

      // 평면의 주요 파라미터 설정
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      // 평면 방정식 n*x + d = 0에서 d 계산 (중심을 이용)
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
      plane->is_plane = true;

      // 업데이트 조건: 초기 업데이트이거나, 이전 업데이트 이후 100개 이상의 점이 추가된 경우
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }

    } else { // 평면성이 부족한 경우: is_plane을 false로 설정하고, 나머지 평면 파라미터는 동일하게 기록
      if (!plane->is_init) {
        plane->id = plane_id;
        plane_id++;
        plane->is_init = true;
      }
      if (plane->last_update_points_size == 0) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      } else if (plane->points_size - plane->last_update_points_size > 100) {
        plane->last_update_points_size = plane->points_size;
        plane->is_update = true;
      }
      plane->is_plane = false;
      plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
          evecs.real()(2, evalsMin);
      plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
          evecs.real()(2, evalsMid);
      plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
          evecs.real()(2, evalsMax);
      plane->min_eigen_value = evalsReal(evalsMin);
      plane->mid_eigen_value = evalsReal(evalsMid);
      plane->max_eigen_value = evalsReal(evalsMax);
      plane->radius = sqrt(evalsReal(evalsMax));
      plane->d = -(plane->normal(0) * plane->center(0) +
                   plane->normal(1) * plane->center(1) +
                   plane->normal(2) * plane->center(2));
    }
  }

  void init_octo_tree() {
    // 임시 점의 수가 업데이트 임계값을 초과할 때만 평면 초기화 수행 (temp_points_.size() > 5)
    if (temp_points_.size() > max_plane_update_threshold_) {
      // 현재 노드의 점들로 평면 모델 초기화
      init_plane(temp_points_, plane_ptr_, new_points_num_);
      if (plane_ptr_->is_plane == true) {
        // 평면성이 인정되면, 노드 상태를 0으로 설정
        octo_state_ = 0;
        // 점의 수가 너무 많으면 공분산 업데이트나 일반 업데이트를 비활성화 (계산 효율 고려)
        if (temp_points_.size() > max_points_size_) {
          update_enable_ = false;
        }

      } else {
        // 평면성이 부족하면, 노드 상태를 1로 설정하고 재귀적으로 노드 분할 수행
        octo_state_ = 1;
        cut_octo_tree();
      }
      // 노드 초기화 완료 표시 및 새 점 카운터 초기화
      init_octo_ = true;
      new_points_num_ = 0;
      //      temp_points_.clear();
    }
  }

  void cut_octo_tree() {
    // 현재 노드의 layer가 최대 layer에 도달했다면 더 이상 분할하지 않음.
    if (layer_ >= max_layer_ - 1) { // max_layer_ -> max_layer_-1
      octo_state_ = 0;
      return;
    }
    // 현재 노드에 저장된 모든 임시 점(temp_points_)에 대해
    for (size_t i = 0; i < temp_points_.size(); i++) {
      int xyz[3] = {
        temp_points_[i].point[0] > voxel_center_[0] ? 1 : 0,
        temp_points_[i].point[1] > voxel_center_[1] ? 1 : 0,
        temp_points_[i].point[2] > voxel_center_[2] ? 1 : 0
      };
      int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
      if (leaves_[leafnum] == nullptr) {
        leaves_[leafnum] = new OctoTree(
            max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
            max_mature_points_size_,
            planer_threshold_,
            voxel_size_);
        // 리프의 voxel 중심은 현재 voxel 중심에서 quater_length_의 방향에 따라 offset됨
        leaves_[leafnum]->voxel_center_[0] =
            voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[1] =
            voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
        leaves_[leafnum]->voxel_center_[2] =
            voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;

        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
        leaves_[leafnum]->pv_center = temp_points_[i];
      }
      // 해당 리프에 현재 점을 추가하고, 새로운 점 카운터 증가
      leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
      leaves_[leafnum]->new_points_num_++;
    }
    for (uint i = 0; i < 8; i++) {
      if (leaves_[i] != nullptr) {
        if (leaves_[i]->temp_points_.size() >
            leaves_[i]->max_plane_update_threshold_) { // temp_points_.size() > 5

          init_plane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_, leaves_[i]->new_points_num_);
          if (leaves_[i]->plane_ptr_->is_plane) {
            leaves_[i]->octo_state_ = 0;
          } else {
            leaves_[i]->octo_state_ = 1;
            leaves_[i]->cut_octo_tree();
          }
          leaves_[i]->init_octo_ = true;
          leaves_[i]->new_points_num_ = 0;
        }
      }
    }
  }

  bool DeterminePointOnPlane(const Plane *plane, const pointWithCov &pv) {

    bool is_on_plane = false;
    const auto &point = pv.point;
    const double radius_k = 3.0;
    const double sigma_num = 3.0;
    

    float dis_to_plane =
        fabs(plane->normal(0) * point(0) + plane->normal(1) * point(1) +
             plane->normal(2) * point(2) + plane->d);

    float dis_to_center =
        (plane->center(0) - point(0)) * (plane->center(0) - point(0)) +
        (plane->center(1) - point(1)) * (plane->center(1) - point(1)) +
        (plane->center(2) - point(2)) * (plane->center(2) - point(2));

    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane->radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = point - plane->center;
      J_nq.block<1, 3>(0, 3) = -plane->normal;
      double sigma_l = J_nq * plane->plane_cov * J_nq.transpose();
      sigma_l += plane->normal.transpose() * pv.cov * plane->normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) is_on_plane = true;
    }

    return is_on_plane;
  }

  void UpdatePlaneWithOutOfPlane(const pointWithCov &pv) {
    if (!update_enable_) {
      if (all_points_num_ < max_mature_points_size_) {
        if (!DeterminePointOnPlane(plane_ptr_, pv)) {

          plane_ptr_->num_out_of_plane++;
          temp_points_.push_back(pv);
          all_points_num_++;

          if (plane_ptr_->num_out_of_plane > max_plane_update_threshold_) {
            init_plane(temp_points_, plane_ptr_, new_points_num_);

            plane_ptr_->num_out_of_plane = 0;
          }
        }
        }
        else {
          if (temp_points_.size() != 0) std::vector<pointWithCov>().swap(temp_points_);
        }
    }
    return;
  }

  void UpdateOctoTree(const pointWithCov &pv) {
    // save the most closest point to the center of the voxel
    Eigen::Vector3d center(voxel_center_[0], voxel_center_[1], voxel_center_[2]);
    if ((pv.point - center).squaredNorm() < (pv_center.point - center).squaredNorm()) {
      pv_center = pv;
    }

    if (!init_octo_) {
      new_points_num_++;
      all_points_num_++;
      temp_points_.push_back(pv);
      if (temp_points_.size() > max_plane_update_threshold_) {
        init_octo_tree();
      }
    } else {
      if (plane_ptr_->is_plane) {
        if (update_enable_) {
          new_points_num_++;
          all_points_num_++;
          if (update_enable_) temp_points_.push_back(pv);

          if (new_points_num_ > update_size_threshold_) {
            if (update_enable_) init_plane(temp_points_, plane_ptr_, new_points_num_);
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
          }
        } else {
          if (max_mature_points_size_ != update_points_size_)  UpdatePlaneWithOutOfPlane(pv);
          else std::vector<pointWithCov>().swap(temp_points_);
          return;
        }
      } else {
        if (layer_ < max_layer_-1) {
          if (temp_points_.size() != 0) {
            std::vector<pointWithCov>().swap(temp_points_);
          }
          int xyz[3] = {
            pv.point[0] > voxel_center_[0] ? 1 : 0,
            pv.point[1] > voxel_center_[1] ? 1 : 0,
            pv.point[2] > voxel_center_[2] ? 1 : 0
          };
          int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
          if (leaves_[leafnum] != nullptr) {
            leaves_[leafnum]->UpdateOctoTree(pv);
          } else {
            leaves_[leafnum] = new OctoTree(
                max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                max_mature_points_size_,
                planer_threshold_,
                voxel_size_);
            leaves_[leafnum]->layer_point_size_ = layer_point_size_;
            leaves_[leafnum]->voxel_center_[0] =
                voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[1] =
                voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
            leaves_[leafnum]->voxel_center_[2] =
                voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
            leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            leaves_[leafnum]->UpdateOctoTree(pv);
          }
        } else {
          if (update_enable_) {
            new_points_num_++;
            all_points_num_++;
            if (update_enable_) temp_points_.push_back(pv);

            if (new_points_num_ > update_size_threshold_) {
              if (update_enable_) init_plane(temp_points_, plane_ptr_, new_points_num_);
              new_points_num_ = 0;
            }
            if (all_points_num_ >= max_points_size_) update_enable_ = false;
          }
        }
      }
    }
  }
};

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
            uint8_t &b) {
  r = 255;
  g = 255;
  b = 255;

  if (v < vmin) {
    v = vmin;
  }

  if (v > vmax) {
    v = vmax;
  }

  double dr, dg, db;

  if (v < 0.1242) {
    db = 0.504 + ((1. - 0.504) / 0.1242) * v;
    dg = dr = 0.;
  } else if (v < 0.3747) {
    db = 1.;
    dr = 0.;
    dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
  } else if (v < 0.6253) {
    db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
    dg = 1.;
    dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
  } else if (v < 0.8758) {
    db = 0.;
    dr = 1.;
    dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
  } else {
    db = 0.;
    dg = 0.;
    dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
  }

  r = (uint8_t)(255 * dr);
  g = (uint8_t)(255 * dg);
  b = (uint8_t)(255 * db);
}


/**
 * buildVoxelMap 함수는 입력된 점 리스트(input_points)를 이용하여 voxel map을 구축합니다.
 * 각 voxel은 OctoTree 객체로 관리되며, voxel map은 feat_map에 저장됩니다.
 *
 * @param input_points        점과 공분산 정보를 담은 벡터 (예: pv_list)
 * @param voxel_size          voxel의 크기 (예: max_voxel_size)
 * @param max_layer           각 voxel의 계층적 구조에서 최대 레벨 (일반적으로 2~4)
 * @param layer_point_size    각 계층별로 필요한 점의 수를 담은 벡터 (예: layer_size, ex. [5, 5, 5, 5, 5])
 * @param max_points_size     한 voxel에 저장 가능한 최대 점의 개수 (예: 1000)
 * @param planer_threshold    평면 여부 판단에 사용되는 임계값, 예를 들어 최소 고유값 (ex: 0.01)
 * @param feat_map            생성된 voxel map을 저장하는 unordered_map, 키는 VOXEL_LOC, 값은 OctoTree 포인터
 */

void buildVoxelMap(const std::vector<pointWithCov> &input_points, // pv_list
                   const float voxel_size, // max_voxel_size
                   const int max_layer, // max_layer (2~4)
                   const std::vector<int> &layer_point_size, // layer_size ([5, 5, 5, 5, 5])
                   const int max_points_size, // max_points_size (1000)
                   const int max_mature_points_size, // max_mature_points_size (1000)                
                   const float planer_threshold, // min_eigen_value (0.01)
                   tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map) {
  
  uint plsize = input_points.size();
  for (uint i = 0; i < plsize; i++) {
    // 현재 처리할 점 정보 (pointWithCov 타입)
    const pointWithCov p_v = input_points[i];
    // Allign a float array
    const V3D& loc_xyz = p_v.point;

    // Align voxel loc id
    VOXEL_LOC position(loc_xyz / voxel_size);

              
    // Find the pointer of query voxel
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_mature_points_size,
                       planer_threshold,
                       voxel_size);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;

      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
    Eigen::Vector3d center(feat_map[position]->voxel_center_[0], 
                           feat_map[position]->voxel_center_[1], 
                           feat_map[position]->voxel_center_[2]);
    if ((p_v.point - center).squaredNorm() < (feat_map[position]->pv_center.point - center).squaredNorm()) {
      feat_map[position]->pv_center = p_v;
    }
  }

  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->init_octo_tree();
  }
}

void updateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size,
                    const int max_mature_points_size,
                    const float planer_threshold,
                    tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; i++) {
        const pointWithCov p_v = input_points[i];
        const V3D& loc_xyz = p_v.point;
        VOXEL_LOC position(loc_xyz / voxel_size);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
            feat_map[position]->UpdateOctoTree(p_v);
        } else {
            OctoTree *octo_tree =
                    new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                                 max_mature_points_size,
                                 planer_threshold,
                                 voxel_size);
            feat_map[position] = octo_tree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
            feat_map[position]->UpdateOctoTree(p_v);
        }
    }
}

void updateVoxelMapOMP(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, 
                    const int max_mature_points_size,
                    const float planer_threshold,
                    tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map) {
  tsl::robin_map<VOXEL_LOC, vector<pointWithCov>, VoxelHash> position_index_map;
  int insert_count = 0, update_count = 0;
  uint plsize = input_points.size();


  double t_update_start = omp_get_wtime();
  for (uint i = 0; i < plsize; i++) {
    const pointWithCov p_v = input_points[i];
    const V3D& loc_xyz = p_v.point;
    VOXEL_LOC position(loc_xyz/voxel_size);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      update_count++;
      position_index_map[position].push_back(p_v);
    } else {
      insert_count++;
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_mature_points_size,
                       planer_threshold,
                       voxel_size);
      feat_map[position] = octo_tree;
      feat_map[position]->quater_length_ = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;

      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
  double t_update_end = omp_get_wtime();
  std::vector<std::pair<VOXEL_LOC, std::vector<pointWithCov>>> position_vec;
  position_vec.reserve(position_index_map.size());
  for (const auto &kv : position_index_map) {
    position_vec.emplace_back(kv);
  }

  t_update_start = omp_get_wtime();
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for default(none) shared(position_vec, feat_map)
#endif
  for (size_t i = 0; i < position_vec.size(); ++i) {
    const VOXEL_LOC &position = position_vec[i].first;
    const std::vector<pointWithCov> &points = position_vec[i].second;
    for (const pointWithCov &p_v : points) {
      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
  t_update_end = omp_get_wtime();
}


void RemovePointsFarFromLocation(const V3D &pos,
                                 tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map,
                                 const double max_distance) {
    const auto squared_max_distance = max_distance * max_distance;
    for (auto it = feat_map.begin(); it != feat_map.end();) {
        const auto &[voxel, current_octo] = *it;
        const auto pt = V3D(current_octo->voxel_center_);
        if ((pt - pos).squaredNorm() >= (squared_max_distance)) {
            delete  current_octo;
            it = feat_map.erase(it);
        } else {
            ++it;
        }
    }
}

// source의 sigma만을 사용해서, correspondence_searching을 하는 함수

void build_single_ptpl_residual(const pointWithCov &pv, const OctoTree *current_octo,
                                const int current_layer, const int max_layer,
                                const double sigma_num, bool &is_sucess,
                                double &prob, ptpl &single_ptpl) {
  double radius_k = 3;
  Eigen::Vector3d p_w = pv.point_world;
  if (current_octo->plane_ptr_->is_plane) {
    Plane &plane = *current_octo->plane_ptr_;
    float dis_to_plane =
        fabs(plane.normal(0) * p_w(0) + plane.normal(1) * p_w(1) +
             plane.normal(2) * p_w(2) + plane.d);
    float dis_to_center =
        (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
        (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
        (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
    float range_dis = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_w - plane.center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_sucess = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob = this_prob;
          single_ptpl.point = pv.point;
          single_ptpl.point_world = pv.point_world;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal = plane.normal;
          single_ptpl.center = plane.center;
          single_ptpl.d = plane.d;
          single_ptpl.layer = current_layer;
          single_ptpl.cov_lidar = pv.cov_lidar;
        }
        return;
      } else {
        return;
      }
    } else {
      return;
    }
  } else {
    if (!(current_octo->init_octo_))  return;
    if (current_layer < max_layer - 1) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {

          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          build_single_ptpl_residual(pv, leaf_octo, current_layer + 1, max_layer,
                                    sigma_num, is_sucess, prob, single_ptpl);
        }
      }
      return;
    } else {
      return;
    }
  }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
                    std::vector<Plane> &plane_list) {
  if (current_octo->layer_ > pub_max_voxel_layer) {
    return;
  }
  if (current_octo->plane_ptr_->is_update) {
    plane_list.push_back(*current_octo->plane_ptr_);
  }
  if (current_octo->layer_ < current_octo->max_layer_) {
    if (!current_octo->plane_ptr_->is_plane) {
      for (size_t i = 0; i < 8; i++) {
        if (current_octo->leaves_[i] != nullptr) {
          GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
                         plane_list);
        }
      }
    }
  }
  return;
}



void BuildResidualListOMP(const tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  std::mutex mylock;
  ptpl_list.clear();
  non_match.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<V3D> non_match_list(pv_list.size());
  std::vector<bool> useful_ptpl(pv_list.size());
  std::vector<size_t> index(pv_list.size());

  size_t ptpl_num = 0;
  size_t non_match_num = 0;
  for (size_t i = 0; i < index.size(); ++i) {
    index[i] = i;
    useful_ptpl[i] = false;
  }
#ifdef MP_EN
  omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
  for (int i = 0; i < index.size(); i++) {
    pointWithCov pv = pv_list[i];
    const V3D& loc_xyz = pv.point_world;
    VOXEL_LOC position(loc_xyz / voxel_size);
    // 查找当前点所属的voxel
    auto iter = voxel_map.find(position);

    if (iter != voxel_map.end()) {
      OctoTree *current_octo = iter->second;
      ptpl single_ptpl;
      bool is_sucess = false;
      double prob = 0;

      build_single_ptpl_residual(pv, current_octo, 0, max_layer, sigma_num,
                                is_sucess, prob, single_ptpl);
      if (!is_sucess) {
        VOXEL_LOC near_position = position;
        V3I dLoc = V3I::Zero();

        for (int j=0; j<3 ; j++) {
          if (loc_xyz(j) >
            (current_octo->voxel_center_[j] + current_octo->quater_length_)) {
            dLoc(j) = 1;
          }
          else if (loc_xyz(j) < (current_octo->voxel_center_[j] -
                                  current_octo->quater_length_)) {
            dLoc(j) = -1;
          }
        }

        double closest_distance_to_axis = voxel_size;
        int near_axis = -1;
        for (int j =0; j<3; j++) {
          if (dLoc(j) == 0) continue;
          double axis_distance = std::abs(current_octo->voxel_center_[j] + dLoc(j)*voxel_size - loc_xyz(j));

          if (axis_distance < closest_distance_to_axis) {
            closest_distance_to_axis = axis_distance;
            near_axis = j;
          }
        }

        if (near_axis == 0)       near_position.x += dLoc.x();
        else if (near_axis == 1)  near_position.y += dLoc.y();
        else if (near_axis == 2)  near_position.z += dLoc.z();

        // if (near_position == position) continue;
        if (near_axis != -1) {
          auto iter_near = voxel_map.find(near_position);
          if (iter_near != voxel_map.end()) {
            build_single_ptpl_residual(pv, iter_near->second, 0, max_layer, sigma_num,
                                      is_sucess, prob, single_ptpl);
          }
        }

      }
      if (is_sucess) {
        mylock.lock();
        ptpl_num++;
        useful_ptpl[i] = true;
        all_ptpl_list[i] = single_ptpl;
        mylock.unlock();
      } else {
        mylock.lock();
        non_match_num++;
        useful_ptpl[i] = false;
        non_match_list[i] = pv.point;
        mylock.unlock();
      }
    }
    else {
      mylock.lock();
      non_match_num++;
      useful_ptpl[i] = false;
      non_match_list[i] = pv.point;
      mylock.unlock();
    }
  }
  ptpl_list.reserve(ptpl_num);
  non_match.reserve(non_match_num);
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.emplace_back(all_ptpl_list[i]);
    }
    else{
      non_match.emplace_back(non_match_list[i]);
    }
  }
}

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec,
                     const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q) {

  Eigen::Matrix3d rot;
  rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0),
      z_vec(1), z_vec(2);
  Eigen::Matrix3d rotation = rot.transpose();
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void CalcQuation(const Eigen::Vector3d &vec, const int axis,
                 geometry_msgs::Quaternion &q) {
  Eigen::Vector3d x_body = vec;
  Eigen::Vector3d y_body(1, 1, 0);
  if (x_body(2) != 0) {
    y_body(2) = -(y_body(0) * x_body(0) + y_body(1) * x_body(1)) / x_body(2);
  } else {
    if (x_body(1) != 0) {
      y_body(1) = -(y_body(0) * x_body(0)) / x_body(1);
    } else {
      y_body(0) = 0;
    }
  }
  y_body.normalize();
  Eigen::Vector3d z_body = x_body.cross(y_body);
  Eigen::Matrix3d rot;

  rot << x_body(0), x_body(1), x_body(2), y_body(0), y_body(1), y_body(2),
      z_body(0), z_body(1), z_body(2);
  Eigen::Matrix3d rotation = rot.transpose();
  if (axis == 2) {
    Eigen::Matrix3d rot_inc;
    rot_inc << 0, 0, 1, 0, 1, 0, -1, 0, 0;
    rotation = rotation * rot_inc;
  }
  Eigen::Quaterniond eq(rotation);
  q.w = eq.w();
  q.x = eq.x();
  q.y = eq.y();
  q.z = eq.z();
}

void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub,
                    const std::string plane_ns, const Plane &single_plane,
                    const float alpha, const Eigen::Vector3d rgb) {
  visualization_msgs::Marker plane;
  plane.header.frame_id = "camera_init";
  plane.header.stamp = ros::Time();
  plane.ns = plane_ns;
  plane.id = single_plane.id;
  plane.type = visualization_msgs::Marker::CYLINDER;
  plane.action = visualization_msgs::Marker::ADD;
  plane.pose.position.x = single_plane.center[0];
  plane.pose.position.y = single_plane.center[1];
  plane.pose.position.z = single_plane.center[2];
  geometry_msgs::Quaternion q;
  CalcVectQuation(single_plane.x_normal, single_plane.y_normal,
                  single_plane.normal, q);
  plane.pose.orientation = q;
  plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
  plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
  plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
  plane.color.a = alpha;
  plane.color.r = rgb(0);
  plane.color.g = rgb(1);
  plane.color.b = rgb(2);
  plane.lifetime = ros::Duration();
  plane_pub.markers.push_back(plane);
}

void pubNoPlaneMap(const tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map,
                   const ros::Publisher &plane_map_pub) {
  int id = 0;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (!iter->second->plane_ptr_->is_plane) {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          OctoTree *temp_octo_tree = iter->second->leaves_[i];
          if (!temp_octo_tree->plane_ptr_->is_plane) {
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (!temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                  Eigen::Vector3d plane_rgb(1, 1, 1);
                  pubSinglePlane(voxel_plane, "no_plane",
                                 *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                 use_alpha, plane_rgb);
                }
              }
            }
          }
        }
      }
    }
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}



void pubVoxelMap(const tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &voxel_map,
                 const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {
  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 0.8;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);
  std::vector<Plane> pub_plane_list;
  for (auto iter = voxel_map.begin(); iter != voxel_map.end(); iter++) {
    GetUpdatePlane(iter->second, pub_max_voxel_layer, pub_plane_list);
  }
  for (size_t i = 0; i < pub_plane_list.size(); i++) {
    V3D plane_cov = pub_plane_list[i].plane_cov.block<3, 3>(0, 0).diagonal();
    double trace = plane_cov.sum();
    if (trace >= max_trace) {
      trace = max_trace;
    }
    trace = trace * (1.0 / max_trace);
    trace = pow(trace, pow_num);
    uint8_t r, g, b;
    mapJet(trace, 0, 1, r, g, b);
    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
    double alpha;
    if (pub_plane_list[i].is_plane) {
      alpha = use_alpha;
    } else {
      alpha = 0;
    }
    pubSinglePlane(voxel_plane, "plane", pub_plane_list[i], alpha, plane_rgb);
  }
  plane_map_pub.publish(voxel_plane);
  loop.sleep();
}

void pubPlaneMap(const tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> &feat_map,
                 const ros::Publisher &plane_map_pub) {
  OctoTree *current_octo = nullptr;

  double max_trace = 0.25;
  double pow_num = 0.2;
  ros::Rate loop(500);
  float use_alpha = 1.0;
  visualization_msgs::MarkerArray voxel_plane;
  voxel_plane.markers.reserve(1000000);

  for (auto iter = feat_map.begin(); iter != feat_map.end(); iter++) {
    if (iter->second->plane_ptr_->is_update) {
      Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

      V3D plane_cov =
          iter->second->plane_ptr_->plane_cov.block<3, 3>(0, 0).diagonal();
      double trace = plane_cov.sum();
      if (trace >= max_trace) {
        trace = max_trace;
      }
      trace = trace * (1.0 / max_trace);
      trace = pow(trace, pow_num);
      uint8_t r, g, b;
      mapJet(trace, 0, 1, r, g, b);
      Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
      float alpha = 0.0;
      if (iter->second->plane_ptr_->is_plane) {
        alpha = use_alpha;
      } else {

      }

      pubSinglePlane(voxel_plane, "plane", *(iter->second->plane_ptr_), alpha,
                     plane_rgb);

      iter->second->plane_ptr_->is_update = false;
    } else {
      for (uint i = 0; i < 8; i++) {
        if (iter->second->leaves_[i] != nullptr) {
          if (iter->second->leaves_[i]->plane_ptr_->is_update) {
            Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);

            V3D plane_cov = iter->second->leaves_[i]
                                ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                                .diagonal();
            double trace = plane_cov.sum();
            if (trace >= max_trace) {
              trace = max_trace;
            }
            trace = trace * (1.0 / max_trace);
            trace = pow(trace, pow_num);
            uint8_t r, g, b;
            mapJet(trace, 0, 1, r, g, b);
            Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
            plane_rgb << 0, 1, 0;

            float alpha = 0.0;
            if (iter->second->leaves_[i]->plane_ptr_->is_plane) {
              alpha = use_alpha;
            } else {
            }
            pubSinglePlane(voxel_plane, "plane",
                           *(iter->second->leaves_[i]->plane_ptr_), alpha,
                           plane_rgb);
            iter->second->leaves_[i]->plane_ptr_->is_update = false;
          } else {
            OctoTree *temp_octo_tree = iter->second->leaves_[i];
            for (uint j = 0; j < 8; j++) {
              if (temp_octo_tree->leaves_[j] != nullptr) {
                if (temp_octo_tree->leaves_[j]->octo_state_ == 0 &&
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update) {
                  if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                    // std::cout << "subsubplane" << std::endl;
                    Eigen::Vector3d normal_rgb(0.0, 1.0, 0.0);
                    V3D plane_cov =
                        temp_octo_tree->leaves_[j]
                            ->plane_ptr_->plane_cov.block<3, 3>(0, 0)
                            .diagonal();
                    double trace = plane_cov.sum();
                    if (trace >= max_trace) {
                      trace = max_trace;
                    }
                    trace = trace * (1.0 / max_trace);
                    trace = pow(trace, pow_num);
                    uint8_t r, g, b;
                    mapJet(trace, 0, 1, r, g, b);
                    Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
                    plane_rgb << 0, 0, 1;
                    float alpha = 0.0;
                    if (temp_octo_tree->leaves_[j]->plane_ptr_->is_plane) {
                      alpha = use_alpha;
                    }

                    pubSinglePlane(voxel_plane, "plane",
                                   *(temp_octo_tree->leaves_[j]->plane_ptr_),
                                   alpha, plane_rgb);
                    // loop.sleep();
                    temp_octo_tree->leaves_[j]->plane_ptr_->is_update = false;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  plane_map_pub.publish(voxel_plane);
  // plane_map_pub.publish(voxel_norm);
  loop.sleep();
}

M3D calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc)
{
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  return direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
};

#endif