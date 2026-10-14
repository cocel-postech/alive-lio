// ALIVE-LIO: Degeneracy-Aware Learning of Inertial Velocity for Enhancing
// ESKF-Based LiDAR-Inertial Odometry (IEEE T-ITS 2026)
// Copyright (c) 2026 Seongjun Kim, POSTECH.
// Licensed under the GNU General Public License v2.0 (see LICENSE).
//
// This file is derived from PV-LIO (https://github.com/HViktorTsoi/PV-LIO),
// FAST-LIO2 (https://github.com/hku-mars/FAST_LIO) and VoxelMap
// (https://github.com/hku-mars/VoxelMap), all GPL-2.0. The original
// copyright notices below are retained.

// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <Python.h>
#include <so3_math.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <Eigen/Core>
#include "IMU_Processing.hpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/Vector3.h>
#include <livox_ros_driver/CustomMsg.h>
#include "preprocess.h"
#include "voxel_map_util.hpp"
#include <chrono>   // 연산 시간 체크를 위한 인클루딩
#include <deque>    // Motion window를 위한 덱 자료구조 인클루딩
#include <cmath>

// Save cov
#include "std_msgs/Float64MultiArray.h"

// Onnx
#include <onnxruntime_cxx_api.h>
#include <sophus/so3.h> 
#include "fixed_deque.hpp"
#include "onnx_handler.hpp"
#include "degeneracy_detector.hpp"
#include "velocity_estimator.hpp"
#include "console_status.hpp"



#define INIT_TIME           (0.1)
#define LASER_POINT_COV     (0.001)
#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

/*** Time Log Variables ***/
double kdtree_incremental_time = 0.0, kdtree_search_time = 0.0, kdtree_delete_time = 0.0;
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot4[MAXN], s_plot5[MAXN], s_plot6[MAXN], s_plot7[MAXN], s_plot8[MAXN], s_plot9[MAXN], s_plot10[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
int    kdtree_size_st = 0, kdtree_size_end = 0, add_point_size = 0, kdtree_delete_counter = 0;
bool   time_sync_en = false, extrinsic_est_en = true, path_en = true;
double lidar_time_offset = 0.0;
/**************************/

float res_last[100000] = {0.0};
float map_range = 300.0f;
const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

string root_dir = ROOT_DIR;
string  lid_topic, imu_topic;

double res_mean_last = 0.05, total_residual = 0.0;
double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double total_distance = 0, lidar_end_time = 0, first_lidar_time = 0.0;
int    effct_feat_num = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;
int    iterCount = 0, feats_down_size = 0, NUM_MAX_ITERATIONS = 0, laserCloudValidNum = 0, pcd_index = 0;
int    final_planar_num = 0, final_non_planar_num = 0;
bool   point_selected_surf[100000] = {0};
bool   lidar_pushed, flg_first_scan = true, flg_exit = false, flg_EKF_inited;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;


vector<vector<int>>  pointSearchInd_surf;
vector<PointVector>  Nearest_Points;
vector<double>       extrinT(3, 0.0);
vector<double>       extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr normvec(new PointCloudXYZI(100000, 1));
PointCloudXYZI::Ptr _featsArray;
std::vector<M3D> var_down_body;

pcl::VoxelGrid<PointType> downSizeFilterSurf;

std::vector<float> nn_dist_in_feats;
std::vector<float> nn_plane_std;
PointCloudXYZI::Ptr feats_with_correspondence(new PointCloudXYZI());

V3F XAxisPoint_body(LIDAR_SP_LEN, 0.0, 0.0);
V3F XAxisPoint_world(LIDAR_SP_LEN, 0.0, 0.0);
V3D euler_cur;
V3D position_last(Zero3d);
V3D Lidar_T_wrt_IMU(Zero3d);
M3D Lidar_R_wrt_IMU(Eye3d);

// params for voxel mapping algorithm
double min_eigen_value = 0.003;
int max_layer = 0;

int max_points_size = 50;
int max_mature_points_size = 1000;
double sigma_num = 3.0;
double max_voxel_size = 1.0;
std::vector<int> layer_size;

double ranging_cov = 0.0;
double angle_cov = 0.0;
std::vector<double> layer_point_size;




bool publish_voxel_map = false;
int publish_max_voxel_layer = 0;
bool scan_semantic_publish_en = false;

tsl::robin_map<VOXEL_LOC, OctoTree *, VoxelHash> voxel_map;

/*** EKF inputs and output ***/
MeasureGroup Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom state_point;
state_ikfom predicted_state_point;
vect3 pos_lid;

nav_msgs::Path path;
nav_msgs::Odometry odomAftMapped;
geometry_msgs::Quaternion geoQuat;
geometry_msgs::PoseStamped msg_body_pose;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

// planar와 non-planar 시각화를 위한 저장 변수
std::vector<ptpl> planar_points;
std::vector<ptpo> non_planar_points;

int degeneracy_detection_mode = 0;
int num_meas = 0;

bool en_network = false;
bool remove_g = false;
bool remove_bias = false;
bool rotation_invariant = false;
bool isReadyInfer = false;
int buffer_len = 1000;
double t_th = 0.0;
string model_name = "model";
bool use_gpu = false;
int num_threads = -1;
int acc_unit = 0;
double net_vel_scale = 0.01;   // measurement-noise scale for network velocity update


DegeneracyDetector degeneracy_detector;
bool optimize_degeneracy_only = false;
bool update_rotation_via_velocity = false;
int update_mode = 0;
bool en_constrained_optimize = false;

bool publish_estimated_value = false;
bool publish_state_value = false;
bool console_status_en = true;      // per-frame console dashboard (degeneracy + velocity comparison)
ConsoleStatus console_status;



FixedDeque<float> acc_buffer(static_cast<size_t>(buffer_len*3));
FixedDeque<float> gyro_buffer(static_cast<size_t>(buffer_len*3));
FixedDeque<float> ori_buffer(static_cast<size_t>(buffer_len*3));


void SigHandle(int sig)
{
    flg_exit = true;
    ROS_WARN("catch sig %d", sig);
    sig_buffer.notify_all();
}

const bool var_contrast(pointWithCov &x, pointWithCov &y) {
    return (x.cov.diagonal().squaredNorm() < y.cov.diagonal().squaredNorm());
};

void pointBodyToWorld_ikfom(PointType const * const pi, PointType * const po, state_ikfom &s)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(s.rot * (s.offset_R_L_I*p_body + s.offset_T_L_I) + s.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}


void pointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
//    po->intensity = pi->intensity;
}

template<typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po)
{
    V3D p_body(pi[0], pi[1], pi[2]);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po[0] = p_global(0);
    po[1] = p_global(1);
    po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const * const pi, PointType * const po)
{
    V3D p_body(pi->x, pi->y, pi->z);
    V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);

    po->x = p_global(0);
    po->y = p_global(1);
    po->z = p_global(2);
    po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const * const pi, PointType * const po)
{
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu(state_point.offset_R_L_I*p_body_lidar + state_point.offset_T_L_I);

    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;

    po->curvature = pi->curvature;
    po->normal_x = pi->normal_x;
}

void standard_pcl_cbk(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    auto time_offset = lidar_time_offset;
    mtx_buffer.lock();
    scan_count ++;
    double preprocess_start_time = omp_get_wtime();
    if (msg->header.stamp.toSec() + time_offset < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    // time_buffer.push_back(ros::Time::now().toSec() + time_offset);
    time_buffer.push_back(msg->header.stamp.toSec() + time_offset);
    last_timestamp_lidar = msg->header.stamp.toSec() + time_offset;
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

double timediff_lidar_wrt_imu = 0.0;
bool   timediff_set_flg = false;
void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg)
{
    mtx_buffer.lock();
    double preprocess_start_time = omp_get_wtime();
    scan_count ++;
    if (msg->header.stamp.toSec() < last_timestamp_lidar)
    {
        ROS_ERROR("lidar loop back, clear buffer");
        lidar_buffer.clear();
    }
    last_timestamp_lidar = msg->header.stamp.toSec();

    if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() && !lidar_buffer.empty() )
    {
        printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n",last_timestamp_imu, last_timestamp_lidar);
    }

    if (time_sync_en && !timediff_set_flg && abs(last_timestamp_lidar - last_timestamp_imu) > 1 && !imu_buffer.empty())
    {
        timediff_set_flg = true;
        timediff_lidar_wrt_imu = last_timestamp_lidar + 0.1 - last_timestamp_imu;
        printf("Self sync IMU and LiDAR, time diff is %.10lf \n", timediff_lidar_wrt_imu);
    }

    PointCloudXYZI::Ptr  ptr(new PointCloudXYZI());
    p_pre->process(msg, ptr);
    lidar_buffer.push_back(ptr);
    time_buffer.push_back(last_timestamp_lidar);

    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in)
{
    publish_count ++;
    sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

    if (abs(timediff_lidar_wrt_imu) > 0.1 && time_sync_en)
    {
        msg->header.stamp = \
        ros::Time().fromSec(timediff_lidar_wrt_imu + msg_in->header.stamp.toSec());
    }

    double timestamp = msg->header.stamp.toSec();

    if (timestamp < last_timestamp_imu)
    {
        ROS_WARN("imu loop back, ignoring!!!");
        ROS_WARN("current T: %f, last T: %f", timestamp, last_timestamp_imu);
        return;
    }
    if (std::abs(msg->angular_velocity.x) > 10
        || std::abs(msg->angular_velocity.y) > 10
        || std::abs(msg->angular_velocity.z) > 10) {
        ROS_WARN("Large IMU measurement!!! Drop Data!!! %.3f  %.3f  %.3f",
                 msg->angular_velocity.x,
                 msg->angular_velocity.y,
                 msg->angular_velocity.z
        );
        return;
    }

    

    mtx_buffer.lock();
    last_timestamp_imu = timestamp;
    imu_buffer.push_back(msg);
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}


double lidar_mean_scantime = 0.0;
int    scan_num = 0;
bool sync_packages(MeasureGroup &meas)
{
    if (lidar_buffer.empty() || imu_buffer.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if(!lidar_pushed)
    {
        meas.lidar = lidar_buffer.front();
        meas.lidar_beg_time = time_buffer.front();
        if (meas.lidar->points.size() <= 1) // time too little
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
            ROS_WARN("Too few input point cloud!\n");
        }
        else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime)
        {
            lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
        }
        else
        {
            scan_num ++;
            if (meas.lidar->points.back().curvature / double(1000) > 0.05*2){
                lidar_end_time = meas.lidar_beg_time;
            }
            else {
                lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
            }
            
            lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
        }

        meas.lidar_end_time = lidar_end_time;

        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time)
    {
        return false;
    }

    /*** push imu data, and pop from imu buffer ***/
    double imu_time = imu_buffer.front()->header.stamp.toSec();
    meas.imu.clear();
    mtx_buffer.lock();
    size_t num_imu = imu_buffer.size();
    while ((!imu_buffer.empty()) && (imu_time < lidar_end_time))
    {
        imu_time = imu_buffer.front()->header.stamp.toSec();
        if(imu_time > lidar_end_time) break;
        meas.imu.push_back(imu_buffer.front());
        imu_buffer.pop_front();
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();

    mtx_buffer.unlock();
    sig_buffer.notify_all();

    lidar_pushed = false;
    return true;
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());


void publish_frame_world(const ros::Publisher & pubLaserCloudFull)
{
    if(scan_pub_en)
    {
        PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
        int size = laserCloudFullRes->points.size();
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size; i++)
        {
            PointType const * const p = &laserCloudFullRes->points[i];
            if(p->intensity < 5){
                continue;
            }
            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);

            laserCloudWorld.push_back(p_world);

        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFull.publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

}

void publish_frame_body(const ros::Publisher & pubLaserCloudFull_body)
{
    PointCloudXYZI::Ptr laserCloudFullRes(dense_pub_en ? feats_undistort : feats_down_body);
    int size = laserCloudFullRes->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));
    for (int i = 0; i < size; i++)
    {
        RGBpointBodyLidarToIMU(&laserCloudFullRes->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudmsg.header.frame_id = "body";
    pubLaserCloudFull_body.publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

void publish_map(const ros::Publisher & pubLaserCloudMap)
{
    sensor_msgs::PointCloud2 laserCloudMap;
    pcl::toROSMsg(*featsFromMap, laserCloudMap);
    laserCloudMap.header.stamp = ros::Time().fromSec(lidar_end_time);
    laserCloudMap.header.frame_id = "camera_init";
    pubLaserCloudMap.publish(laserCloudMap);
}

template<typename T>
void set_posestamp(T & out)
{
    out.pose.position.x = state_point.pos(0);
    out.pose.position.y = state_point.pos(1);
    out.pose.position.z = state_point.pos(2);
    out.pose.orientation.x = geoQuat.x;
    out.pose.orientation.y = geoQuat.y;
    out.pose.orientation.z = geoQuat.z;
    out.pose.orientation.w = geoQuat.w;

}

void publish_odometry(const ros::Publisher & pubOdomAftMapped)
{
    odomAftMapped.header.frame_id = "camera_init";
    odomAftMapped.child_frame_id = "body";
    odomAftMapped.header.stamp = ros::Time().fromSec(Measures.lidar_beg_time);// ros::Time().fromSec(lidar_end_time);
    set_posestamp(odomAftMapped.pose);
    pubOdomAftMapped.publish(odomAftMapped);
    auto P = kf.get_P();
    for (int i = 0; i < 6; i ++)
    {
        int k = i < 3 ? i + 3 : i - 3;
        odomAftMapped.pose.covariance[i*6 + 0] = P(k, 3);
        odomAftMapped.pose.covariance[i*6 + 1] = P(k, 4);
        odomAftMapped.pose.covariance[i*6 + 2] = P(k, 5);
        odomAftMapped.pose.covariance[i*6 + 3] = P(k, 0);
        odomAftMapped.pose.covariance[i*6 + 4] = P(k, 1);
        odomAftMapped.pose.covariance[i*6 + 5] = P(k, 2);
    }

    static tf::TransformBroadcaster br;
    tf::Transform                   transform;
    tf::Quaternion                  q;
    transform.setOrigin(tf::Vector3(odomAftMapped.pose.pose.position.x, \
                                    odomAftMapped.pose.pose.position.y, \
                                    odomAftMapped.pose.pose.position.z));
    q.setW(odomAftMapped.pose.pose.orientation.w);
    q.setX(odomAftMapped.pose.pose.orientation.x);
    q.setY(odomAftMapped.pose.pose.orientation.y);
    q.setZ(odomAftMapped.pose.pose.orientation.z);
    transform.setRotation( q );
    br.sendTransform( tf::StampedTransform( transform, odomAftMapped.header.stamp, "camera_init", "body" ) );

    static tf::TransformBroadcaster br_world;
    transform.setOrigin(tf::Vector3(0, 0, 0));
    q.setValue(p_imu->Initial_R_wrt_G.x(), p_imu->Initial_R_wrt_G.y(), p_imu->Initial_R_wrt_G.z(), p_imu->Initial_R_wrt_G.w());
    transform.setRotation(q);
    br_world.sendTransform(tf::StampedTransform(transform, odomAftMapped.header.stamp, "world", "camera_init"));
}

void publish_path(const ros::Publisher pubPath)
{
    set_posestamp(msg_body_pose);
    msg_body_pose.header.stamp = ros::Time().fromSec(lidar_end_time);
    msg_body_pose.header.frame_id = "camera_init";

    /*** if path is too large, the rvis will crash ***/
    static int jjj = 0;
    jjj++;
    if (jjj % 1 == 0)
    {
        path.header.stamp = msg_body_pose.header.stamp;
        path.poses.push_back(msg_body_pose);
        pubPath.publish(path);
    }
}

void publish_semantic_pointcloud(const state_ikfom &state_point, const ros::Publisher & pubPlanar,
                                 const ros::Publisher & pubNonPlanar,
                                 const std::vector<ptpl> &planar_points,
                                 const std::vector<ptpo> &non_planar_points)
{
    const auto publishPTPL = [&](const size_t size_ptpl){
        PointCloudXYZI::Ptr laserCloudPTPL(new PointCloudXYZI);
        laserCloudPTPL->points.reserve(size_ptpl);
        
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size_ptpl; i++)
        {
            pcl::PointXYZINormal pt;
            
            pt.x = planar_points[i].point.x();
            pt.y = planar_points[i].point.y();
            pt.z = planar_points[i].point.z();

            PointType const * const p = &pt;

            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
            laserCloudWorld.push_back(p_world);

        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";

        pubPlanar.publish(laserCloudmsg);
    };

    const auto publishPTPO = [&](const size_t size_ptpo){
        PointCloudXYZI::Ptr laserCloudPTPL(new PointCloudXYZI);
        laserCloudPTPL->points.reserve(size_ptpo);
        
        PointCloudXYZI laserCloudWorld;
        for (int i = 0; i < size_ptpo; i++)
        {
            pcl::PointXYZINormal pt;
            
            pt.x = non_planar_points[i].point.x();
            pt.y = non_planar_points[i].point.y();
            pt.z = non_planar_points[i].point.z();

            PointType const * const p = &pt;// = &(planar_points[i].point);

            PointType p_world;

            RGBpointBodyToWorld(p, &p_world);
            laserCloudWorld.push_back(p_world);

        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time);
        laserCloudmsg.header.frame_id = "camera_init";

        pubNonPlanar.publish(laserCloudmsg);
    };
    const size_t size_ptpl = planar_points.size();

    if (size_ptpl == 0){
        ROS_WARN("No point-to-planar to publish.");
    }
    else {
        publishPTPL(size_ptpl);
    }
    const size_t size_ptpo = non_planar_points.size();
    if (size_ptpo > 0) publishPTPO(size_ptpo);

    return;
}

void publish_covariance(const ros::Publisher &pubCovariance)
{
    std_msgs::Float64MultiArray msg;

    const auto P = kf.get_P();
    
    const auto get_flattened_matrix = [](const Eigen::Matrix<double, 23, 23> &P, const int index) {
        Eigen::Matrix<double, 3, 3> block = P.block(index, index, 3, 3);
        return std::vector<float>(block.data(), block.data() + 9);

    };
    ros::Time t = ros::Time().fromSec(lidar_end_time);

    msg.data.push_back(t.sec);
    msg.data.push_back(t.nsec);


    std::vector<int> indices = {0, 3, 12, 15, 18};

    for (const auto &index : indices) {
        std::vector<float> P_flattened = get_flattened_matrix(P, index);
        msg.data.insert(msg.data.end(), P_flattened.begin(), P_flattened.end());
    }

    pubCovariance.publish(msg);
}

void publish_state(const ros::Publisher &pubState, const V3D &gyro_bias, const V3D &acc_bias, const V3D &velocity, const V3D &gravity)
{
    std_msgs::Float64MultiArray msg;

    ros::Time t = ros::Time().fromSec(lidar_end_time);

    msg.data.push_back(t.sec);
    msg.data.push_back(t.nsec);

    msg.data.push_back(gyro_bias(0));
    msg.data.push_back(gyro_bias(1));
    msg.data.push_back(gyro_bias(2));

    msg.data.push_back(acc_bias(0));
    msg.data.push_back(acc_bias(1));
    msg.data.push_back(acc_bias(2));

    msg.data.push_back(velocity(0));
    msg.data.push_back(velocity(1));
    msg.data.push_back(velocity(2));

    msg.data.push_back(gravity(0));
    msg.data.push_back(gravity(1));
    msg.data.push_back(gravity(2));

    pubState.publish(msg);
}

void publish_condition_number(const ros::Publisher &pubConditionNumber) {
    std_msgs::Float64MultiArray msg;
    ros::Time t = ros::Time().fromSec(lidar_end_time);
    msg.data.push_back(t.sec);
    msg.data.push_back(t.nsec);

    const auto &[low_cond, high_cond] = degeneracy_detector.get_cond_num_trans();
    const auto & [degeneracy_matrix, degeneracy_vector] = degeneracy_detector.get_degeneracy_block();
    const auto & degeneracy_flags = degeneracy_detector.get_degeneracy_flags();
    const auto & [num_degeneracy_trans, num_degeneracy_rot] = degeneracy_detector.get_degeneracy_num();
    for (const auto &num: low_cond) msg.data.push_back(num);
    for (const auto &num: high_cond) msg.data.push_back(num);
    for (int i=0; i<3 ; i++) msg.data.push_back(degeneracy_flags[i]);


    for (int i = 0; i < 3; ++i) {
        if (i < num_degeneracy_trans) {
            for (int j = 0; j < 3; ++j) {
                msg.data.push_back(degeneracy_matrix(i, j));
            }
        }
        else {
            for (int j = 0; j < 3; ++j) {
                msg.data.push_back(0);
            }
        }
    }
    pubConditionNumber.publish(msg);
}

void publish_collected_data(const ros::Publisher &pubCovariance,
                            const ros::Publisher &pubFinalState, const ros::Publisher &pubPredictedState,
                            const ros::Publisher &pubFinalError, const ros::Publisher &pubPredictedError,
                            const ros::Publisher &pubKFMeasuredError, const ros::Publisher &pubKFPredictedError,
                            const state_ikfom &state_point,
                            const state_ikfom &predicted_state_point,
                            const Eigen::Matrix<double, 23, 1> &dx,
                            const Eigen::Matrix<double, 23, 1> &predicted_dx,
                            const Eigen::Matrix<double, 23, 1> &kf_measured_dx,
                            const Eigen::Matrix<double, 23, 1> &kf_predicted_dx) {
    publish_covariance(pubCovariance);
    publish_state(pubFinalState, state_point.bg, state_point.ba, state_point.vel, state_point.grav);
    publish_state(pubPredictedState, predicted_state_point.bg, predicted_state_point.ba, predicted_state_point.vel, state_point.grav);

    publish_state(pubFinalError, V3D(dx.block<3, 1> (15, 0)), V3D(dx.block<3, 1> (18, 0)), V3D(dx.block<3, 1> (12, 0)), V3D(dx.block<3, 1> (15, 0)));
    publish_state(pubPredictedError, V3D(predicted_dx.block<3, 1> (15, 0)), V3D(predicted_dx.block<3, 1> (18, 0)), V3D(predicted_dx.block<3, 1> (12, 0)), V3D(predicted_dx.block<3, 1> (15, 0)));

    publish_state(pubKFMeasuredError, V3D(kf_measured_dx.block<3, 1> (15, 0)), V3D(kf_measured_dx.block<3, 1> (18, 0)), V3D(kf_measured_dx.block<3, 1> (12, 0)), V3D(kf_measured_dx.block<3, 1> (15, 0)));
    publish_state(pubKFPredictedError, V3D(kf_predicted_dx.block<3, 1> (15, 0)), V3D(kf_predicted_dx.block<3, 1> (18, 0)), V3D(kf_predicted_dx.block<3, 1> (12, 0)), V3D(kf_predicted_dx.block<3, 1> (15, 0)));
}

void publish_network_output(const ros::Publisher &pubNetworkOutput, const V3D & v_global, const V3D & cov){
    std_msgs::Float64MultiArray msg;

    ros::Time t = ros::Time().fromSec(lidar_end_time);
    // std::cout<<

    msg.data.push_back(t.sec);
    msg.data.push_back(t.nsec);

    msg.data.push_back(v_global(0));
    msg.data.push_back(v_global(1));
    msg.data.push_back(v_global(2));

    msg.data.push_back(cov(0));
    msg.data.push_back(cov(1));
    msg.data.push_back(cov(2));

    pubNetworkOutput.publish(msg);
}

void publish_velocity(const ros::Publisher &pubVelocity, const std::vector<V3D> &vels){
    std_msgs::Float64MultiArray msg;

    ros::Time t = ros::Time().fromSec(lidar_end_time);
    // std::cout<<

    msg.data.push_back(t.sec);
    msg.data.push_back(t.nsec);

    int num_data = vels.size();

    for (int i=0; i< num_data; i++) {
        const V3D &vel = vels[i];
        msg.data.push_back(vel.x());
        msg.data.push_back(vel.y());
        msg.data.push_back(vel.z());
    }

    pubVelocity.publish(msg);
}

void transformLidar(const state_ikfom &state_point, const PointCloudXYZI::Ptr &input_cloud, PointCloudXYZI::Ptr &trans_cloud)
{
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        pcl::PointXYZINormal p_c = input_cloud->points[i];
        Eigen::Vector3d p_lidar(p_c.x, p_c.y, p_c.z);

        V3D p_body = state_point.rot * (state_point.offset_R_L_I * p_lidar + state_point.offset_T_L_I) + state_point.pos;
        PointType pi;
        pi.x = p_body(0);
        pi.y = p_body(1);
        pi.z = p_body(2);
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

M3D transformLiDARCovToWorld(Eigen::Vector3d &p_lidar, const esekfom::esekf<state_ikfom, 12, input_ikfom>& kf, const Eigen::Matrix3d& COV_lidar)
{
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(p_lidar);
    auto state = kf.get_x();

    M3D il_rot_var = kf.get_P().block<3, 3>(6, 6);
    M3D il_t_var = kf.get_P().block<3, 3>(9, 9);

    M3D COV_body =
            state.offset_R_L_I * COV_lidar * state.offset_R_L_I.conjugate()
            + state.offset_R_L_I * (-point_crossmat) * il_rot_var * (-point_crossmat).transpose() * state.offset_R_L_I.conjugate()
            + il_t_var;

    V3D p_body = state.offset_R_L_I * p_lidar + state.offset_T_L_I;

    point_crossmat << SKEW_SYM_MATRX(p_body);
    M3D rot_var = kf.get_P().block<3, 3>(3, 3);
    M3D t_var = kf.get_P().block<3, 3>(0, 0);

    // Eq. (3)
    M3D COV_world =
        state.rot * COV_body * state.rot.conjugate()
        + state.rot * (-point_crossmat) * rot_var * (-point_crossmat).transpose()  * state.rot.conjugate()
        + t_var;

    return COV_world;
}

void observation_model_share(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data)
{
    feats_with_correspondence->clear();
    total_residual = 0.0;

    vector<pointWithCov> pv_list;
    PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);

    transformLidar(s, feats_down_body, world_lidar);
    pv_list.resize(feats_down_body->size());
    for (size_t i = 0; i < feats_down_body->size(); i++) {
        pointWithCov pv;
        pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
        pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
        M3D cov_lidar = var_down_body[i];
        M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);
        pv.cov = cov_world;
        pv.cov_lidar = cov_lidar;
        pv_list[i] = pv;
    }

    double match_start = omp_get_wtime();
    std::vector<ptpl> ptpl_list;
    std::vector<V3D> non_match_list;
    BuildResidualListOMP(voxel_map, max_voxel_size, sigma_num, max_layer, pv_list, ptpl_list, non_match_list); // 3.0 means 3 sigma
    double match_end = omp_get_wtime();

    // non-match points도 rviz로 시각화할 때를 위한 용도
    if (scan_semantic_publish_en) {
        planar_points = ptpl_list;

        // ptpo(custom)와 non_match_list(vector)가 자료 구조가 달라서 이렇게 선언했는데, 필요 시 non_match_list를 처음부터 ptpo형태로 바꿀 수 있음
        non_planar_points.clear();
        non_planar_points.resize(non_match_list.size());

        for (int i=0; i<non_match_list.size(); i++){
            non_planar_points[i].point = non_match_list[i];
        }
    }

    /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
    effct_feat_num = ptpl_list.size();
    if (effct_feat_num < 1){
        ekfom_data.valid = false;
        ROS_WARN("No Effective Points! \n");
        return;
    }
    degeneracy_detector.init_information_matrix(effct_feat_num);

    num_meas = effct_feat_num;

    ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);
    ekfom_data.h.resize(effct_feat_num);
    ekfom_data.R.resize(effct_feat_num, 1);


    double max_Rinv = 0.0;

#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
    for (int i = 0; i < effct_feat_num; i++)
    {

//        const PointType &laser_p  = laserCloudOri->points[i];
        V3D point_this_be(ptpl_list[i].point);
        M3D point_be_crossmat;
        point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
        V3D point_this = s.offset_R_L_I * point_this_be + s.offset_T_L_I;
        M3D point_crossmat;
        point_crossmat<<SKEW_SYM_MATRX(point_this);

        /*** get the normal vector of closest surface/corner ***/
//        const PointType &norm_p = corr_normvect->points[i];
//        V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
        V3D norm_vec(ptpl_list[i].normal);

        /*** calculate the Measuremnt Jacobian matrix H ***/
        V3D C(s.rot.conjugate() *norm_vec);
        V3D A(point_crossmat * C);
        if (extrinsic_est_en)
        {
            V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C); //s.rot.conjugate()*norm_vec);
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B), VEC_FROM_ARRAY(C);
        }
        else
        {
            ekfom_data.h_x.block<1, 12>(i,0) << norm_vec.x(), norm_vec.y(), norm_vec.z(), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
        }


        degeneracy_detector.fill_H(i, ekfom_data.h_x.block<1, 6>(i, 0));

        /*** Measuremnt: distance to the closest surface/corner ***/
        float pd2 = norm_vec.x() * ptpl_list[i].point_world.x()
                + norm_vec.y() * ptpl_list[i].point_world.y()
                + norm_vec.z() * ptpl_list[i].point_world.z()
                + ptpl_list[i].d;
        ekfom_data.h(i) = -pd2;

        V3D point_world = ptpl_list[i].point_world;
        // /*** get the normal vector of closest surface/corner ***/
        Eigen::Matrix<double, 1, 6> J_nq;
        J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center;
        J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
        double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();

        M3D cov_lidar = ptpl_list[i].cov_lidar;
        M3D R_cov_Rt = s.rot * s.offset_R_L_I * cov_lidar * s.offset_R_L_I.conjugate() * s.rot.conjugate();
        double R_inv = 1.0 / (sigma_l + norm_vec.transpose() * R_cov_Rt * norm_vec);

        ekfom_data.R(i) = std::clamp(R_inv, 1e-5, 1e5);

        degeneracy_detector.fill_weight_trans(i, ekfom_data.R(i));
        degeneracy_detector.fill_weight_rot(i, ekfom_data.R(i));
        if (R_inv > max_Rinv) max_Rinv = R_inv;
    }

    res_mean_last = total_residual / effct_feat_num;

    degeneracy_detector.compute_degeneracy_direction();
    const auto &[num_degeneracy_trans, num_degeneracy_rot] = degeneracy_detector.get_degeneracy_num();
    ekfom_data.degeneracy_flag = en_constrained_optimize & degeneracy_detector.degeneracy_occured;

    const auto &[C, d] = degeneracy_detector.get_degeneracy_block();
    ekfom_data.C = C.block(0, 0, num_degeneracy_trans, 6);
    ekfom_data.d = d.block(0, 0, num_degeneracy_trans, 1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "laserMapping");
    ros::NodeHandle nh;

    nh.param<double>("time_offset", lidar_time_offset, 0.0);

    nh.param<bool>("publish/path_en",path_en, true);
    nh.param<bool>("publish/scan_publish_en",scan_pub_en, true);
    nh.param<bool>("publish/dense_publish_en",dense_pub_en, true);
    nh.param<bool>("publish/scan_bodyframe_pub_en",scan_body_pub_en, true);
    nh.param<string>("common/lid_topic",lid_topic,"/livox/lidar");
    nh.param<string>("common/imu_topic", imu_topic,"/livox/imu");
    nh.param<bool>("common/time_sync_en", time_sync_en, false);

    // mapping algorithm params
    nh.param<float>("mapping/map_range",map_range,300.f);
    nh.param<int>("mapping/max_iteration", NUM_MAX_ITERATIONS, 4);
    nh.param<int>("mapping/max_points_size", max_points_size, 100);
    nh.param<int>("mapping/max_mature_points_size", max_mature_points_size, 1000);
    nh.param<vector<double>>("mapping/layer_point_size", layer_point_size,vector<double>());
    nh.param<int>("mapping/max_layer", max_layer, 2);
    nh.param<double>("mapping/voxel_size", max_voxel_size, 1.0);
    nh.param<double>("mapping/down_sample_size", filter_size_surf_min, 0.5);
    nh.param<double>("mapping/plannar_threshold", min_eigen_value, 0.01);
    nh.param<double>("mapping/sigma_num", sigma_num, 3.0);
    nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en, true);
    nh.param<vector<double>>("mapping/extrinsic_T", extrinT, vector<double>());
    nh.param<vector<double>>("mapping/extrinsic_R", extrinR, vector<double>());

    // localizability
    nh.param<bool>("localizability/en_network", en_network, false);
    nh.param<int>("localizability/degeneracy_detection_mode", degeneracy_detection_mode, 0);
    nh.param<bool>("localizability/remove_g", remove_g, false);
    nh.param<bool>("localizability/remove_bias", remove_bias, false);
    nh.param<bool>("localizability/rotation_invariant", rotation_invariant, false);
    nh.param<int>("localizability/buffer_len", buffer_len, 1000);
    nh.param<bool>("localizability/publish_estimated_value", publish_estimated_value, false);
    nh.param<bool>("localizability/optimize_degeneracy_only", optimize_degeneracy_only, false);
    nh.param<int>("localizability/update_mode", update_mode, 0);
    nh.param<string>("localizability/model_name", model_name, "model");
    nh.param<bool>("localizability/use_gpu", use_gpu, false);
    nh.param<int>("localizability/num_threads", num_threads, -1);
    nh.param<bool>("localizability/en_constrained_optimize", en_constrained_optimize, false);
    nh.param<bool>("localizability/update_rotation_via_velocity", update_rotation_via_velocity, false);
    nh.param<double>("localizability/t_th", t_th, 0.0);
    nh.param<int>("localizability/acc_unit", acc_unit, 0);
    nh.param<double>("localizability/net_vel_scale", net_vel_scale, 0.01);
    // nh.param<bool>("localizability/en_degeneracy_weight", en_degeneracy_weight, false);
    

    
    // noise model params
    nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
    nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);
    nh.param<double>("noise_model/gyr_cov",gyr_cov,0.1);
    nh.param<double>("noise_model/acc_cov",acc_cov,0.1);
    nh.param<double>("noise_model/b_gyr_cov",b_gyr_cov,0.0001);
    nh.param<double>("noise_model/b_acc_cov",b_acc_cov,0.0001);

    // visualization params
    nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
    nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
    nh.param<bool>("publish/scan_semantic_publish_en", scan_semantic_publish_en, true);
    nh.param<bool>("publish/publish_state_value", publish_state_value, true);
    nh.param<bool>("publish/console_status_en", console_status_en, true);
    

    nh.param<double>("preprocess/blind_min", p_pre->blind_min, 0.01);
    nh.param<double>("preprocess/blind_max", p_pre->blind_max, 100.0);
    nh.param<int>("preprocess/lidar_type", p_pre->lidar_type, AVIA);
    nh.param<int>("preprocess/scan_line", p_pre->N_SCANS, 16);
    nh.param<int>("preprocess/scan_rate", p_pre->SCAN_RATE, 10);
    nh.param<int>("preprocess/point_filter_num", p_pre->point_filter_num, 1);
    nh.param<bool>("preprocess/feature_extract_enable", p_pre->feature_enabled, false);
    cout<<"p_pre->lidar_type "<<p_pre->lidar_type<<endl;
    for (int i = 0; i < layer_point_size.size(); i++) {
        layer_size.push_back(layer_point_size[i]);
    }

    path.header.stamp    = ros::Time::now();
    path.header.frame_id ="camera_init";

    /*** variables definition ***/
    int effect_feat_num = 0, frame_num = 0;
    bool flg_EKF_converged, EKF_stop_flg = 0;

    _featsArray.reset(new PointCloudXYZI());

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));
    memset(point_selected_surf, true, sizeof(point_selected_surf));
    memset(res_last, -1000.0f, sizeof(res_last));

    // XXX 暂时现在lidar callback中固定转换到IMU系下
    Lidar_T_wrt_IMU<<VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU<<MAT_FROM_ARRAY(extrinR);
    p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
    p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
    p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
    p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
    p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

    double epsi[23] = {0.001};
    fill(epsi, epsi+23, 0.001);
    
    kf.init_dyn_share(get_f, df_dx, df_dw, observation_model_share, NUM_MAX_ITERATIONS, epsi);


    std::string pkg_path = ros::package::getPath("alive_lio");
    std::string model_path = pkg_path + "/onnx/model/" + model_name + ".onnx";

    degeneracy_detector.init_degeneracy_mode(degeneracy_detection_mode);
    VelocityEstimator velocity_estimator(buffer_len, remove_g, remove_bias,
                                        rotation_invariant, acc_unit,
                                        update_mode, update_rotation_via_velocity,
                                        model_path, use_gpu, 0, num_threads, net_vel_scale);
    velocity_estimator.warmup();

    
    /*** ROS subscribe initialization ***/
    ros::Subscriber sub_pcl = p_pre->lidar_type == AVIA ? \
        nh.subscribe(lid_topic, 200000, livox_pcl_cbk) : \
        nh.subscribe(lid_topic, 200000, standard_pcl_cbk);
    ros::Subscriber sub_imu = nh.subscribe(imu_topic, 200000, imu_cbk);
    ros::Publisher pubLaserCloudFull = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered", 100000);
    ros::Publisher pubLaserCloudFull_body = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_registered_body", 100000);
    ros::Publisher pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>
            ("/cloud_effected", 100000);
    ros::Publisher pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>
            ("/Laser_map", 100000);
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>
            ("/Odometry", 100000);
    ros::Publisher pubExtrinsic = nh.advertise<nav_msgs::Odometry>
            ("/Extrinsic", 100000);
    ros::Publisher pubPath          = nh.advertise<nav_msgs::Path>
            ("/path", 100000);
    ros::Publisher voxel_map_pub =
            nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);

    // Planar vs Non-planar 시각화 publisher
    ros::Publisher pubPlanar = nh.advertise<sensor_msgs::PointCloud2>("/planar_points", 100000);
    ros::Publisher pubNonPlanar = nh.advertise<sensor_msgs::PointCloud2>("/non_planar_points", 100000);
//------------------------------------------------------------------------------------------------------
    // Covariance
    ros::Publisher pubCovariance = nh.advertise<std_msgs::Float64MultiArray>("/Covariance", 100000);
    ros::Publisher pubFinalState = nh.advertise<std_msgs::Float64MultiArray>("/Final_state", 100000);
    ros::Publisher pubPredictedState = nh.advertise<std_msgs::Float64MultiArray>("/Predicted_state", 100000);
    ros::Publisher pubFinalError = nh.advertise<std_msgs::Float64MultiArray>("/Final_error", 100000);
    ros::Publisher pubPredictedError = nh.advertise<std_msgs::Float64MultiArray>("/Predicted_error", 100000);
    ros::Publisher pubKFMeasuredError = nh.advertise<std_msgs::Float64MultiArray>("/KFMearued_error", 100000);
    ros::Publisher pubKFPredictedError = nh.advertise<std_msgs::Float64MultiArray>("/KFPredicted_error", 100000);
    ros::Publisher pubNetworkOutput = nh.advertise<std_msgs::Float64MultiArray>("/NetworkOuput", 100000);
    ros::Publisher pubConditionNumber = nh.advertise<std_msgs::Float64MultiArray>("/ConditionNumber", 100000);
    
    ros::Publisher pubNeuralVelocity = nh.advertise<std_msgs::Float64MultiArray>("/NeuralVelocities", 100000);
    ros::Publisher pubPreinteVelocity = nh.advertise<std_msgs::Float64MultiArray>("/PreinteVelocities", 100000);
    // for Plane Map
    bool init_map = false;

    double sum_optimize_time = 0, sum_update_time = 0;
    int scan_index = 0;
    signal(SIGINT, SigHandle);
    ros::Rate rate(5000);
    bool status = ros::ok();
    while (status)
    {
        if (flg_exit) break;
        ros::spinOnce();
        if(sync_packages(Measures))
        {
            if (flg_first_scan)
            {
                first_lidar_time = Measures.lidar_beg_time;
                p_imu->first_lidar_time = first_lidar_time;
                flg_first_scan = false;
                continue;
            }
            const auto state_last = kf.get_x();

            /* Need to Init for interpolating SO3 */
            if (!velocity_estimator.init_state_last)   velocity_estimator.init_state(lidar_end_time-0.1, state_last); 

            p_imu->Process(Measures, kf, feats_undistort);
            state_point = kf.get_x();
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;

            predicted_state_point = state_point; 
            static auto predicted_cov = kf.get_P();

            if (feats_undistort->empty() || (feats_undistort == NULL))
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }

            flg_EKF_inited = (Measures.lidar_beg_time - first_lidar_time) < INIT_TIME ? \
                            false : true;
            // ===============================================================================================================
            // if ekf is initialized, then initialize the voxel map
            if (flg_EKF_inited && !init_map) {
                PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
                transformLidar(state_point, feats_undistort, world_lidar); // transform the point cloud to the world coordinate
                std::vector<pointWithCov> pv_list;

                // calculate the covariance of all points in the first frame and use it to build the initial map
                for (size_t i = 0; i < world_lidar->size(); i++) {
                    pointWithCov pv;
                    pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                    V3D point_this(feats_undistort->points[i].x,
                                   feats_undistort->points[i].y,
                                   feats_undistort->points[i].z);
                    // if z=0, error will occur in calcBodyCov. To be solved
                    if (point_this[2] == 0) {
                        point_this[2] = 0.001;
                    }
                    M3D cov_lidar = calcBodyCov(point_this, ranging_cov, angle_cov);
                    // transform to world coordinate
                    M3D cov_world = transformLiDARCovToWorld(point_this, kf, cov_lidar);

                    pv.cov = cov_world;
                    pv_list.push_back(pv);
                    Eigen::Vector3d sigma_pv = pv.cov.diagonal();
                    sigma_pv[0] = sqrt(sigma_pv[0]);
                    sigma_pv[1] = sqrt(sigma_pv[1]);
                    sigma_pv[2] = sqrt(sigma_pv[2]);
                }

                buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                              max_points_size,
                              max_mature_points_size,
                              min_eigen_value,
                              voxel_map);

                if (publish_voxel_map) {
                    pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
                    publish_frame_world(pubLaserCloudFull);
                    publish_frame_body(pubLaserCloudFull_body);
                }
                init_map = true;
                continue;
            }
            auto start = std::chrono::high_resolution_clock::now(); // Frame 연산 시작 시간
            /*** Scan Downsampling ***/
            downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
            downSizeFilterSurf.setInputCloud(feats_undistort);
            downSizeFilterSurf.filter(*feats_down_body);
            sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            feats_down_size = feats_down_body->points.size();

            // Since the body var of the point cloud is always constant, it is calculated in advance and can be reused during iteration
            var_down_body.clear();
            for (auto & pt:feats_down_body->points) {
                V3D point_this(pt.x, pt.y, pt.z);
                var_down_body.push_back(calcBodyCov(point_this, ranging_cov, angle_cov));
            }

            /*** ICP and iterated Kalman filter update ***/
            if (feats_down_size < 5)
            {
                ROS_WARN("No point, skip this scan!\n");
                continue;
            }
            
            // ===============================================================================================================
            double t_update_start = omp_get_wtime();
            double solve_H_time = 0;
            kf.update_iterated_dyn_share_diagonal();
            double t_update_end = omp_get_wtime();
            sum_optimize_time += t_update_end - t_update_start;

            /* LiDAR-only estimate, kept for the console comparison against the network */
            {
                const state_ikfom lio_state = kf.get_x();
                console_status.set_lio_velocity(lio_state.vel, lio_state.rot.conjugate().toRotationMatrix());
                console_status.network_enabled = en_network;
                console_status.network_ready   = false;
                console_status.network_applied = false;
            }

            
            /* Velocity Prediction via Deep Learning */ 
            if (en_network) {
                velocity_estimator.accumulate_buffers(Measures, lidar_end_time, state_point);
                if (velocity_estimator.ready_to_infer)   {
                    velocity_estimator.inference_velocity();
                    const auto &[estim_vel, cov] = velocity_estimator.get_inference_result();
                    console_status.network_ready = true;
                    console_status.vel_neural    = estim_vel;
                    console_status.sigma_neural  = cov;

                    /* Publish Global Velocity */
                    V3D v_global = state_point.rot * estim_vel;
                    if (publish_estimated_value)    publish_network_output(pubNetworkOutput, v_global, cov);
                    // std::cout << "Body velocity : " << (state_point.rot.conjugate()*state_point.vel).transpose() << std::endl;

                    if (publish_estimated_value) {
                        const auto &[preinte_vels, neural_vels] = velocity_estimator.extract_velocities_at();
                        publish_velocity(pubPreinteVelocity, preinte_vels);
                        publish_velocity(pubNeuralVelocity, neural_vels);
                    }

                    if (degeneracy_detector.degeneracy_occured){
                        degeneracy_detector.update_t_degeneracy(lidar_end_time);
                        velocity_estimator.update_velocity_partial(state_point, degeneracy_detector, kf);
                        console_status.network_applied = true;
                    }
                }
                else {
                    /* If using GPU, the warmup process is needed */
                    velocity_estimator.warmup();
                }
            }


            state_point = kf.get_x();
            console_status.set_fused_velocity(state_point.vel);

            euler_cur = SO3ToEuler(state_point.rot);
            pos_lid = state_point.pos + state_point.rot * state_point.offset_T_L_I;
            geoQuat.x = state_point.rot.coeffs()[0];
            geoQuat.y = state_point.rot.coeffs()[1];
            geoQuat.z = state_point.rot.coeffs()[2];
            geoQuat.w = state_point.rot.coeffs()[3];
            
            // ===============================================================================================================
            // update the map
            /*** add the points to the voxel map ***/
            // use the latest state estimate to transform the points and the covariance of the points to the world coordinate
            std::vector<pointWithCov> pv_list;
            PointCloudXYZI::Ptr world_lidar(new PointCloudXYZI);
            transformLidar(state_point, feats_down_body, world_lidar);
            for (size_t i = 0; i < feats_down_body->size(); i++) {
                pointWithCov pv;
                pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
                M3D cov_lidar = var_down_body[i];
                M3D cov_world = transformLiDARCovToWorld(pv.point, kf, cov_lidar);

                pv.cov = cov_world;
                pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
                pv_list.push_back(pv);
            }
            t_update_start = omp_get_wtime();
            std::sort(pv_list.begin(), pv_list.end(), var_contrast);
            updateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size,
                            max_points_size,
                            max_mature_points_size,
                            min_eigen_value,
                            voxel_map);
            t_update_end = omp_get_wtime();
            sum_update_time += t_update_end - t_update_start;

            if (map_range > 0)  RemovePointsFarFromLocation(state_point.pos, voxel_map, map_range);

            scan_index++;


            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = end - start;

            if (console_status_en) {
                console_status.frame_idx       = scan_index;
                console_status.lidar_end_time  = lidar_end_time;
                console_status.loop_ms         = elapsed.count() * 1000.0;
                console_status.num_raw_points  = static_cast<int>(feats_undistort->size());
                console_status.num_down_points = feats_down_size;
                console_status.num_effective   = effct_feat_num;
                console_status.t_degeneracy    = degeneracy_detector.get_t_degeneracy();
                console_status.print(degeneracy_detector);
            }
            // ===============================================================================================================
            /******* Publish odometry *******/
            publish_odometry(pubOdomAftMapped);
            publish_condition_number(pubConditionNumber);

            
            if (publish_state_value)   publish_collected_data(pubCovariance,
                                                        pubFinalState, pubPredictedState,
                                                        pubFinalError, pubPredictedError,
                                                        pubKFMeasuredError, pubKFPredictedError,
                                                        state_point, predicted_state_point,
                                                        kf.get_final_error(), kf.get_predicted_error(),
                                                        kf.get_KFMeasured_error(), kf.get_KFPredicted_error());            
//
//            /*** add the feature points to map kdtree ***/
//            map_incremental();
//
            /******* Publish points *******/
            if (path_en)                         publish_path(pubPath);
            if (scan_pub_en)      publish_frame_world(pubLaserCloudFull);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFull_body);
            if (publish_voxel_map) {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            // Planar vs non_planar publish
            if (scan_semantic_publish_en)     publish_semantic_pointcloud(state_point, pubPlanar, pubNonPlanar,
                                                                            planar_points, non_planar_points);
        }
        status = ros::ok();
        rate.sleep();
    }
    velocity_estimator.print_avg_elapsed_time();
    return 0;
}
