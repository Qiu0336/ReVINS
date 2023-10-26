#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <bitset>
#include "DBow3/src/DBoW3.h"
#include "DBow3/src/DescManip.h"

#include "orb.h"
#include "camera/camera.h"
#include "utils/io.h"
#include "utils/lie_group.h"
#include "utils/timer.h"
#include "settings.h"


class KeyFrame
{
    public:
    KeyFrame(io::timestamp_t _time_stamp, int _index, int _sequence_id, Eigen::Isometry3d &_Twb, cv::Mat &_image);
    void ComputeBRIEFPoint();
    const std::vector<cv::KeyPoint>& GetKeypoints() { return keypoints; }
    const std::vector<cv::Point2f>& GetKeypoints_norm() { return keypoints_norm; }
    const std::vector<int>& GetMapPoint_ids() { return mappoint_ids; }
    const cv::Mat& GetDescriptors() { return descriptors; }
    const DBoW3::BowVector& GetBowVector() { return dbow_vector; }
    io::timestamp_t GetTimestamp() { return time_stamp; }
    int GetIndex() { return index; }
    int GetSequenceId() { return sequence_id; }
    Eigen::Isometry3d GetPose()  { return Twb; }
    Eigen::Isometry3d GetUpdatePose()  { return Twb_update; }
    Eigen::Isometry3d GetCameraPose()  { return Twc; }
    Eigen::Isometry3d GetUpdateCameraPose()  { return Twc_update; }

    void SetPose(const Eigen::Isometry3d& _Twb);
    void SetUpdatePose(const Eigen::Isometry3d& _Twb_update);

    void SetLoopMessage(const Eigen::Isometry3d &_Tij, const int loop_id);
    void SetLoopMessageGt(const Eigen::Isometry3d &_Tij_gt, const int loop_id);

    io::timestamp_t time_stamp;
    int index;
    int sequence_id;
    Eigen::Isometry3d Twb;
    Eigen::Isometry3d Twc;

    Eigen::Isometry3d Twb_update;
    Eigen::Isometry3d Twc_update;

    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::Point2f> keypoints_norm;
    std::vector<int> mappoint_ids;

    cv::Mat descriptors;
    int loop_index;
    Eigen::Isometry3d loop_Tij;
    Eigen::Isometry3d loop_Tij_gt;

    DBoW3::BowVector dbow_vector;
};
