#include <thread>
#include <csignal>
#include <chrono>
#include <atomic>
#include <ros/ros.h>
#include <std_msgs/Float32MultiArray.h>

#include "localizer/icp_localizer.h"
#include "lio_builder/lio_builder.h"
#include <tf2_ros/transform_broadcaster.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>

#include "fastlio/SlamReLoc.h"
#include "fastlio/MapConvert.h"
#include "fastlio/SlamHold.h"
#include "fastlio/SlamStart.h"
#include "fastlio/SlamRelocCheck.h"
// ===== LATENCY DIAG HELPERS =====
static inline double wall_sec()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
// ================================
bool terminate_flag = false;

void signalHandler(int signum)
{
    std::cout << "SHUTTING DOWN LOCALIZER NODE!" << std::endl;
    terminate_flag = true;
}

struct SharedData
{
    std::mutex service_mutex;
    std::mutex main_mutex;
    bool pose_updated = false;
    bool localizer_activate = false;
    bool service_called = false;
    bool service_success = false;

    std::string map_path;
    Eigen::Matrix3d offset_rot = Eigen::Matrix3d::Identity();
    Eigen::Vector3d offset_pos = Eigen::Vector3d::Zero();
    Eigen::Matrix3d local_rot;
    Eigen::Vector3d local_pos;
    Eigen::Matrix4d initial_guess;
    fastlio::PointCloudXYZI::Ptr cloud;

    bool reset_flag = false;
    bool halt_flag = false;

    // ===== ICP 诊断字段 =====
    double icp_last_score    = -1.0;
    double icp_last_align_ms = 0.0;
    int    icp_success_count = 0;
    int    icp_fail_count    = 0;
    double sensor_time_at_icp  = 0.0;  // ICP最近一次使用的传感器时间戳
    double sensor_time_latest  = 0.0;  // 主线程最新传感器时间戳
    // ========================
};

class LocalizerThread
{
public:
    LocalizerThread() {}

    void setSharedDate(std::shared_ptr<SharedData> shared_data)
    {
        shared_data_ = shared_data;
    }

    void setRate(double rate)
    {
        rate_ = std::make_shared<ros::Rate>(rate);
    }
    void setRate(std::shared_ptr<ros::Rate> rate)
    {
        rate_ = rate;
    }
    void setLocalizer(std::shared_ptr<fastlio::IcpLocalizer> localizer)
    {
        icp_localizer_ = localizer;
    }

    void operator()()
    {
        current_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);

        // ===== ICP线程诊断变量 =====
        double icp_rate_window_start = wall_sec();
        int    icp_call_in_window    = 0;
        int    icp_call_total        = 0;
        double sensor_time_snap      = 0.0;
        // ===========================

        while (ros::ok())
        {
            rate_->sleep();
            if (terminate_flag)
                break;
            if (shared_data_->halt_flag)
                continue;
            if (!shared_data_->localizer_activate)
                continue;
            if (!shared_data_->pose_updated)
                continue;
            gloabl_pose_.setIdentity();
            bool rectify = false;
            Eigen::Matrix4d init_guess;
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->pose_updated = false;
                init_guess.setIdentity();
                local_rot_ = shared_data_->local_rot;
                local_pos_ = shared_data_->local_pos;
                init_guess.block<3, 3>(0, 0) = shared_data_->offset_rot * local_rot_;
                init_guess.block<3, 1>(0, 3) = shared_data_->offset_rot * local_pos_ + shared_data_->offset_pos;
                pcl::copyPointCloud(*shared_data_->cloud, *current_cloud_);
                sensor_time_snap = shared_data_->sensor_time_latest; // 记录使用的传感器时间
            }

            // ===== ICP 开始计时 =====
            double icp_t0 = wall_sec();
            // ========================

            if (shared_data_->service_called)

            {
                std::lock_guard<std::mutex> lock(shared_data_->service_mutex);
                shared_data_->service_called = false;
                icp_localizer_->init(shared_data_->map_path, false);
                gloabl_pose_ = icp_localizer_->multi_align_sync(current_cloud_, shared_data_->initial_guess);
                if (icp_localizer_->isSuccess())
                {
                    rectify = true;
                    shared_data_->localizer_activate = true;
                    shared_data_->service_success = true;
                }

                else
                {
                    rectify = false;
                    shared_data_->localizer_activate = false;
                    shared_data_->service_success = false;
                }
            }
            else
            {
                gloabl_pose_ = icp_localizer_->align(current_cloud_, init_guess);
                if (icp_localizer_->isSuccess())
                    rectify = true;
                else
                    rectify = false;
            }

            // ===== ICP完成，记录诊断 =====
            double icp_elapsed_ms = (wall_sec() - icp_t0) * 1000.0;
            icp_call_total++;
            icp_call_in_window++;
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->icp_last_align_ms  = icp_elapsed_ms;
                shared_data_->icp_last_score     = icp_localizer_->getScore();
                shared_data_->sensor_time_at_icp = sensor_time_snap;
                if (rectify) shared_data_->icp_success_count++;
                else         shared_data_->icp_fail_count++;
            }
            // ICP线程每5秒打印汇总
            {
                double now_w = wall_sec();
                double window = now_w - icp_rate_window_start;
                if (window >= 5.0)
                {
                    ROS_INFO("[ICP_DIAG] ICP线程: 实际频率=%.2fHz  本次耗时=%.1fms  "
                             "得分=%.4f  成功=%d  失败=%d  总调用=%d",
                             icp_call_in_window / window,
                             icp_elapsed_ms,
                             icp_localizer_->getScore(),
                             shared_data_->icp_success_count,
                             shared_data_->icp_fail_count,
                             icp_call_total);
                    if (!rectify)
                        ROS_WARN("[ICP_DIAG] ICP对齐失败！得分=%.4f  耗时=%.1fms",
                                 icp_localizer_->getScore(), icp_elapsed_ms);
                    icp_rate_window_start = now_w;
                    icp_call_in_window    = 0;
                }
            }
            // =============================

            if (rectify)
            {
                std::lock_guard<std::mutex> lock(shared_data_->main_mutex);
                shared_data_->offset_rot = gloabl_pose_.block<3, 3>(0, 0) * local_rot_.transpose();
                shared_data_->offset_pos = -gloabl_pose_.block<3, 3>(0, 0) * local_rot_.transpose() * local_pos_ + gloabl_pose_.block<3, 1>(0, 3);
            }
        }
    }

private:
    std::shared_ptr<SharedData> shared_data_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;
    std::shared_ptr<ros::Rate> rate_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr current_cloud_;
    Eigen::Matrix4d gloabl_pose_;
    Eigen::Matrix3d local_rot_;
    Eigen::Vector3d local_pos_;
};

class LocalizerROS
{
public:
    LocalizerROS(tf2_ros::TransformBroadcaster &br, std::shared_ptr<SharedData> shared_data) : shared_date_(shared_data), br_(br)
    {
        initParams();
        initSubscribers();
        initPublishers();
        initServices();
        lio_builder_ = std::make_shared<fastlio::LIOBuilder>(lio_params_);
        icp_localizer_ = std::make_shared<fastlio::IcpLocalizer>(localizer_params_.refine_resolution,
                                                                 localizer_params_.rough_resolution,
                                                                 localizer_params_.refine_iter,
                                                                 localizer_params_.rough_iter,
                                                                 localizer_params_.thresh);
        icp_localizer_->setSearchParams(localizer_params_.xy_offset, localizer_params_.yaw_offset, localizer_params_.yaw_resolution);
        localizer_loop_.setRate(loop_rate_);
        localizer_loop_.setSharedDate(shared_data);
        localizer_loop_.setLocalizer(icp_localizer_);
        localizer_thread_ = std::make_shared<std::thread>(std::ref(localizer_loop_));
    }

    void initParams()
    {
        nh_.param<std::string>("map_frame", global_frame_, "map");
        nh_.param<std::string>("local_frame", local_frame_, "local");
        nh_.param<std::string>("body_frame", body_frame_, "body");
        nh_.param<std::string>("imu_topic", imu_data_.topic, "/livox/imu");
        nh_.param<std::string>("livox_topic", livox_data_.topic, "/livox/lidar");
        nh_.param<bool>("publish_map_cloud", publish_map_cloud_, false);
        double local_rate, loop_rate;
        nh_.param<double>("local_rate", local_rate, 20.0);
        nh_.param<double>("loop_rate", loop_rate, 1.0);
        local_rate_ = std::make_shared<ros::Rate>(local_rate);
        loop_rate_ = std::make_shared<ros::Rate>(loop_rate);

        nh_.param<double>("lio_builder/det_range", lio_params_.det_range, 100.0);
        nh_.param<double>("lio_builder/cube_len", lio_params_.cube_len, 500.0);
        nh_.param<double>("lio_builder/resolution", lio_params_.resolution, 0.1);
        nh_.param<double>("lio_builder/move_thresh", lio_params_.move_thresh, 1.5);
        nh_.param<bool>("lio_builder/align_gravity", lio_params_.align_gravity, true);
        nh_.param<std::vector<double>>("lio_builder/imu_ext_rot", lio_params_.imu_ext_rot, std::vector<double>());
        nh_.param<std::vector<double>>("lio_builder/imu_ext_pos", lio_params_.imu_ext_pos, std::vector<double>());

        nh_.param<double>("localizer/refine_resolution", localizer_params_.refine_resolution, 0.2);
        nh_.param<double>("localizer/rough_resolution", localizer_params_.rough_resolution, 0.5);
        nh_.param<double>("localizer/refine_iter", localizer_params_.refine_iter, 5);
        nh_.param<double>("localizer/rough_iter", localizer_params_.rough_iter, 10);
        nh_.param<double>("localizer/thresh", localizer_params_.thresh, 0.15);

        nh_.param<double>("localizer/xy_offset", localizer_params_.xy_offset, 2.0);
        nh_.param<double>("localizer/yaw_resolution", localizer_params_.yaw_resolution, 0.5);
        nh_.param<int>("localizer/yaw_offset", localizer_params_.yaw_offset, 1);
    }

    void initSubscribers()
    {
        imu_sub_ = nh_.subscribe(imu_data_.topic, 1000, &ImuData::callback, &imu_data_);
        livox_sub_ = nh_.subscribe(livox_data_.topic, 1000, &LivoxData::callback, &livox_data_);
    }

    void initPublishers()
    {
        local_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("local_cloud", 1000);
        body_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("body_cloud", 1000);
        odom_pub_ = nh_.advertise<nav_msgs::Odometry>("Odometry", 1000);
        global_odom_pub_ = nh_.advertise<nav_msgs::Odometry>("Odometry_global", 1000);  // 纠正后的全局Odometry
        map_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("map_cloud", 1000);
        // ===== 诊断话题 =====
        latency_diag_pub_ = nh_.advertise<std_msgs::Float32MultiArray>("latency_diag", 10);
        // ====================
    }
    
    bool relocCallback(fastlio::SlamReLoc::Request &req, fastlio::SlamReLoc::Response &res)
    {
        std::string map_path = req.pcd_path;
        float x = req.x;
        float y = req.y;
        float z = req.z;
        float roll = req.roll;
        float pitch = req.pitch;
        float yaw = req.yaw;
        Eigen::AngleAxisf rollAngle(roll, Eigen::Vector3f::UnitX());
        Eigen::AngleAxisf pitchAngle(pitch, Eigen::Vector3f::UnitY());
        Eigen::AngleAxisf yawAngle(yaw, Eigen::Vector3f::UnitZ());
        Eigen::Quaternionf q = rollAngle * pitchAngle * yawAngle;
        {
            std::lock_guard<std::mutex> lock(shared_date_->service_mutex);
            shared_date_->halt_flag = false;
            shared_date_->service_called = true;
            shared_date_->localizer_activate = true;
            shared_date_->map_path = map_path;
            shared_date_->initial_guess.block<3, 3>(0, 0) = q.toRotationMatrix().cast<double>();
            shared_date_->initial_guess.block<3, 1>(0, 3) = Eigen::Vector3d(x, y, z);
        }
        res.status = 1;
        res.message = "RELOCALIZE CALLED!";

        return true;
    }

    bool mapConvertCallback(fastlio::MapConvert::Request &req, fastlio::MapConvert::Response &res)
    {
        pcl::PCDReader reader;
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        reader.read(req.map_path, *cloud);
        pcl::VoxelGrid<pcl::PointXYZI> down_sample_filter;
        down_sample_filter.setLeafSize(req.resolution, req.resolution, req.resolution);
        down_sample_filter.setInputCloud(cloud);
        down_sample_filter.filter(*cloud);

        fastlio::PointCloudXYZI::Ptr cloud_with_norm = fastlio::IcpLocalizer::addNorm(cloud);
        pcl::PCDWriter writer;
        writer.writeBinaryCompressed(req.save_path, *cloud_with_norm);
        res.message = "CONVERT SUCCESS!";
        res.status = 1;

        return true;
    }

    bool slamHoldCallback(fastlio::SlamHold::Request &req, fastlio::SlamHold::Response &res)
    {
        shared_date_->service_mutex.lock();
        shared_date_->halt_flag = true;
        shared_date_->reset_flag = true;
        shared_date_->service_mutex.unlock();
        res.message = "SLAM HALT!";
        res.status = 1;
        return true;
    }

    bool slamStartCallback(fastlio::SlamStart::Request &req, fastlio::SlamStart::Response &res)
    {
        shared_date_->service_mutex.lock();
        shared_date_->halt_flag = false;
        shared_date_->service_mutex.unlock();
        res.message = "SLAM START!";
        res.status = 1;
        return true;
    }

    bool slamRelocCheckCallback(fastlio::SlamRelocCheck::Request &req, fastlio::SlamRelocCheck::Response &res)
    {
        res.status = shared_date_->service_success;
        return true;
    }

    void initServices()
    {
        reloc_server_ = nh_.advertiseService("slam_reloc", &LocalizerROS::relocCallback, this);
        map_convert_server_ = nh_.advertiseService("map_convert", &LocalizerROS::mapConvertCallback, this);
        hold_server_ = nh_.advertiseService("slam_hold", &LocalizerROS::slamHoldCallback, this);
        start_server_ = nh_.advertiseService("slam_start", &LocalizerROS::slamStartCallback, this);
        reloc_check_server_ = nh_.advertiseService("slam_reloc_check", &LocalizerROS::slamRelocCheckCallback, this);
    }

    void publishCloud(ros::Publisher &publisher, const sensor_msgs::PointCloud2 &cloud_to_pub)
    {
        if (publisher.getNumSubscribers() == 0)
            return;
        publisher.publish(cloud_to_pub);
    }

    void publishOdom(const nav_msgs::Odometry &odom_to_pub)
    {
        if (odom_pub_.getNumSubscribers() == 0)
            return;
        odom_pub_.publish(odom_to_pub);
    }

    void systemReset()
    {
        offset_rot_ = Eigen::Matrix3d::Identity();
        offset_pos_ = Eigen::Vector3d::Zero();
        {
            std::lock_guard<std::mutex> lock(shared_date_->main_mutex);
            shared_date_->offset_rot = Eigen::Matrix3d::Identity();
            shared_date_->offset_pos = Eigen::Vector3d::Zero();
            shared_date_->service_success = false;
        }
        lio_builder_->reset();
    }

    void run()
    {
        // ===== 主循环延迟诊断变量 =====
        double last_diag_wall  = wall_sec();
        int    frame_count     = 0;
        double total_lag_ms    = 0.0;
        double max_lag_ms      = 0.0;
        double total_iter_ms   = 0.0;
        int    slow_frame_count= 0;
        // ==============================

        while (ros::ok())
        {
            double iter_t0 = wall_sec();  // 本帧开始时间

            local_rate_->sleep();
            ros::spinOnce();

            if (terminate_flag)
                break;
            if (!measure_group_.syncPackage(imu_data_, livox_data_))
                continue;
            if (shared_date_->halt_flag)
                continue;

            if (shared_date_->reset_flag)
            {
                // ROS_INFO("SLAM RESET!");
                systemReset();
                shared_date_->service_mutex.lock();
                shared_date_->reset_flag = false;
                shared_date_->service_mutex.unlock();
            }

            lio_builder_->mapping(measure_group_);
            if (lio_builder_->currentStatus() == fastlio::Status::INITIALIZE)
                continue;

            current_time_ = measure_group_.lidar_time_end;
            current_state_ = lio_builder_->currentState();
            current_cloud_body_ = lio_builder_->cloudUndistortedBody();

            // ===== 传感器时间滞后计算 =====
            {
                double wall_now      = wall_sec();
                double ros_now_sec   = ros::Time::now().toSec();
                double lag_ms        = (ros_now_sec - current_time_) * 1000.0;
                double iter_ms       = (wall_now - iter_t0) * 1000.0;

                frame_count++;
                total_lag_ms  += lag_ms;
                total_iter_ms += iter_ms;
                if (lag_ms  > max_lag_ms) max_lag_ms = lag_ms;
                if (iter_ms > 40.0)       slow_frame_count++;

                // 更新最新传感器时间给ICP线程参考
                {
                    std::lock_guard<std::mutex> lock(shared_date_->main_mutex);
                    shared_date_->sensor_time_latest = current_time_;
                }

                // 每5秒打印一次汇总
                if (wall_now - last_diag_wall >= 5.0)
                {
                    double avg_lag   = total_lag_ms  / frame_count;
                    double avg_iter  = total_iter_ms / frame_count;
                    double actual_hz = frame_count / (wall_now - last_diag_wall);

                    double icp_score_s, icp_ms_s, icp_st_s;
                    int    icp_succ_s, icp_fail_s;
                    {
                        std::lock_guard<std::mutex> lk(shared_date_->main_mutex);
                        icp_score_s = shared_date_->icp_last_score;
                        icp_ms_s    = shared_date_->icp_last_align_ms;
                        icp_st_s    = shared_date_->sensor_time_at_icp;
                        icp_succ_s  = shared_date_->icp_success_count;
                        icp_fail_s  = shared_date_->icp_fail_count;
                    }
                    double icp_stale_ms = (current_time_ - icp_st_s) * 1000.0;

                    ROS_INFO("========== [延迟诊断 5s汇总] ==========");
                    ROS_INFO("[主线程] 实际Hz=%.1f  均帧耗时=%.1fms  慢帧=%d/%d",
                             actual_hz, avg_iter, slow_frame_count, frame_count);
                    ROS_INFO("[时间滞后] 传感器落后ROS时间: 均值=%.1fms  最大=%.1fms",
                             avg_lag, max_lag_ms);
                    if (avg_lag > 100.0)
                        ROS_WARN("[时间滞后] !! 均值>100ms，队列严重积压，定位有明显延迟!!");
                    else if (avg_lag > 33.0)
                        ROS_WARN("[时间滞后] 均值>33ms（超过1帧），轻度积压");
                    ROS_INFO("[ICP线程] 得分=%.4f  耗时=%.1fms  成功=%d  失败=%d",
                             icp_score_s, icp_ms_s, icp_succ_s, icp_fail_s);
                    ROS_INFO("[ICP线程] 数据陈旧度=%.1fms（ICP点云距现在多久）", icp_stale_ms);
                    if (icp_stale_ms > 2000.0)
                        ROS_WARN("[ICP线程] !! ICP修正用的点云已陈旧%.1fms !!", icp_stale_ms);
                    ROS_INFO("[缓冲区] IMU=%zu  LiDAR=%zu",
                             imu_data_.buffer.size(), livox_data_.buffer.size());
                    if (livox_data_.buffer.size() > 2)
                        ROS_WARN("[缓冲区] LiDAR积压%zu帧！主循环严重跟不上！",
                                 livox_data_.buffer.size());
                    ROS_INFO("=========================================");

                    // 重置窗口
                    last_diag_wall   = wall_now;
                    frame_count      = 0;
                    total_lag_ms     = max_lag_ms = total_iter_ms = 0.0;
                    slow_frame_count = 0;
                }

                // 每帧发布到 /latency_diag 供 rqt_plot 使用
                if (latency_diag_pub_.getNumSubscribers() > 0)
                {
                    double icp_stale_pub = (current_time_ - shared_date_->sensor_time_at_icp) * 1000.0;
                    std_msgs::Float32MultiArray dmsg;
                    // [0]=传感器lag_ms [1]=LiDAR缓冲帧数 [2]=IMU缓冲帧数
                    // [3]=ICP得分      [4]=ICP耗时ms     [5]=ICP陈旧度ms
                    // [6]=ICP成功次数  [7]=ICP失败次数
                    dmsg.data = {
                        (float)lag_ms,
                        (float)livox_data_.buffer.size(),
                        (float)imu_data_.buffer.size(),
                        (float)shared_date_->icp_last_score,
                        (float)shared_date_->icp_last_align_ms,
                        (float)icp_stale_pub,
                        (float)shared_date_->icp_success_count,
                        (float)shared_date_->icp_fail_count
                    };
                    latency_diag_pub_.publish(dmsg);
                }
            }
            // ==============================

            {
                std::lock_guard<std::mutex> lock(shared_date_->main_mutex);
                shared_date_->local_rot = current_state_.rot.toRotationMatrix();
                shared_date_->local_pos = current_state_.pos;
                shared_date_->cloud = current_cloud_body_;
                offset_rot_ = shared_date_->offset_rot;
                offset_pos_ = shared_date_->offset_pos;
                shared_date_->pose_updated = true;
            }
            br_.sendTransform(eigen2Transform(
                current_state_.rot.toRotationMatrix(),
                current_state_.pos,
                local_frame_,
                body_frame_,
                current_time_));
            br_.sendTransform(eigen2Transform(
                offset_rot_,
                offset_pos_,
                global_frame_,
                local_frame_,
                current_time_));
            publishOdom(eigen2Odometry(current_state_.rot.toRotationMatrix(),
                                       current_state_.pos,
                                       local_frame_,
                                       body_frame_,
                                       current_time_));

            // 发布全局纠正后的Odometry
            {
                // 计算全局位姿: global = offset × local
                Eigen::Matrix3d global_rot = offset_rot_ * current_state_.rot.toRotationMatrix();
                Eigen::Vector3d global_pos = offset_rot_ * current_state_.pos + offset_pos_;

                nav_msgs::Odometry global_odom = eigen2Odometry(global_rot,
                                                                 global_pos,
                                                                 global_frame_,
                                                                 body_frame_,
                                                                 current_time_);

                // 只有激活全局定位时才发布（避免发布错误数据）
                if (shared_date_->localizer_activate && global_odom_pub_.getNumSubscribers() > 0)
                {
                    global_odom_pub_.publish(global_odom);
                }
            }

            publishCloud(body_cloud_pub_,
                         pcl2msg(current_cloud_body_,
                                 body_frame_,
                                 current_time_));
            publishCloud(local_cloud_pub_,
                         pcl2msg(lio_builder_->cloudWorld(),
                                 local_frame_,
                                 current_time_));
            if (publish_map_cloud_)
            {
                if (icp_localizer_->isInitialized())
                {
                    publishCloud(map_cloud_pub_,
                                 pcl2msg(icp_localizer_->getRoughMap(),
                                         global_frame_,
                                         current_time_));
                }
            }
        }

        localizer_thread_->join();
        std::cout << "LOCALIZER NODE IS DOWN!" << std::endl;
    }

private:
    ros::NodeHandle nh_;
    std::string body_frame_;
    std::string local_frame_;
    std::string global_frame_;

    double current_time_;
    bool publish_map_cloud_;
    fastlio::state_ikfom current_state_;

    ImuData imu_data_;
    LivoxData livox_data_;
    MeasureGroup measure_group_;
    std::shared_ptr<SharedData> shared_date_;
    std::shared_ptr<ros::Rate> local_rate_;
    std::shared_ptr<ros::Rate> loop_rate_;
    tf2_ros::TransformBroadcaster &br_;
    fastlio::LioParams lio_params_;
    fastlio::LocalizerParams localizer_params_;
    std::shared_ptr<fastlio::LIOBuilder> lio_builder_;
    std::shared_ptr<fastlio::IcpLocalizer> icp_localizer_;
    LocalizerThread localizer_loop_;
    std::shared_ptr<std::thread> localizer_thread_;

    ros::Subscriber imu_sub_;

    ros::Subscriber livox_sub_;

    ros::Publisher odom_pub_;

    ros::Publisher global_odom_pub_;  // 纠正后的全局Odometry发布器

    ros::Publisher body_cloud_pub_;

    ros::Publisher local_cloud_pub_;

    ros::Publisher map_cloud_pub_;

    ros::Publisher latency_diag_pub_;  // /latency_diag

    ros::ServiceServer reloc_server_;


    ros::ServiceServer map_convert_server_;

    ros::ServiceServer reloc_check_server_;

    ros::ServiceServer hold_server_;

    ros::ServiceServer start_server_;

    Eigen::Matrix3d offset_rot_ = Eigen::Matrix3d::Identity();

    Eigen::Vector3d offset_pos_ = Eigen::Vector3d::Zero();

    fastlio::PointCloudXYZI::Ptr current_cloud_body_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "localizer_node");
    tf2_ros::TransformBroadcaster br;
    signal(SIGINT, signalHandler);
    std::shared_ptr<SharedData> shared_date = std::make_shared<SharedData>();
    LocalizerROS localizer_ros(br, shared_date);
    localizer_ros.run();
    return 0;
}
