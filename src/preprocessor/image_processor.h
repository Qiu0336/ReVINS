#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp>
#include "camera/camera.h"
#include "utils/io.h"
#include "utils/timer.h"
#include "settings.h"

class PointFeature
{
public:
    PointFeature(){}
    PointFeature(int feature_id_, int first_frame_id_, cv::Point2f& point_, const std::vector<cv::Mat>& img_pyr):
        feature_id(feature_id_), first_frame_id(first_frame_id_)
    {
        cv::Size WinSize = cv::Size(2*half_patch_size+1, 2*half_patch_size+1);
        for(int level = 0; level < pyramid_levels; ++level)
        {
            src_patch[level] = cv::Mat(WinSize, CV_16S);
            last_src_patch[level] = cv::Mat(WinSize, CV_16S);
        }
        AddNewestPoint(first_frame_id_, img_pyr, point_);
    }

    void TrackPatch(const std::vector<cv::Mat>& dst_img_pyr, const std::vector<cv::Mat>& derivX_pyr, const std::vector<cv::Mat>& derivY_pyr,
                    cv::Point2f& cur_pt, uchar& status, float& error, const Eigen::Isometry3d& Tji,
                    bool use_penalty = false);
    void AddNewestPoint(const int frame_id, const std::vector<cv::Mat>& img_pyr, cv::Point2f& newest_pt);


    int feature_id;
    int first_frame_id;

    int src_frame_id;
    cv::Point2f src_corner;
    cv::Point2f src_undist_corner;
    cv::Mat src_patch[MAX_PARAMID_LEVELS];

    int last_src_frame_id;
    cv::Point2f last_src_corner;
    cv::Point2f last_src_undist_corner;
    cv::Mat last_src_patch[MAX_PARAMID_LEVELS];

    std::map<int, cv::Point2f> history_corners;
    std::map<int, cv::Point2f> history_undist_corners;

    bool solve_flag = false;
    double inv_dep = 0.1;
    Eigen::Vector3d position;
};


class ImageProcessor
{
public:

    struct TrackFeature{
        TrackFeature(){}
        TrackFeature(int feature_id_, int ref_frame_id_, cv::Point2f& point_):
            feature_id(feature_id_), ref_frame_id(ref_frame_id_), pre_point(point_)
        {}
        int feature_id;
        int ref_frame_id;
        cv::Point2f pre_point;
        cv::Point2f cur_point;
    };

    ImageProcessor(const std::string &yaml_path);
    cv::Mat GetImage(int id);
    std::map<int, Eigen::Vector2d> GetTrackedFeatures(const int id, const cv::Mat& origin_image, bool& is_keyframe,
                                                      std::map<int, Eigen::Isometry3d>& poses_list, std::map<int, Eigen::Vector3f>& mappoint_list);
    io::timestamp_t GetTimestamp(int id);

    std::vector<std::shared_ptr<PointFeature>> tracking_points;
    std::vector<cv::Mat> pre_img_pyr, cur_img_pyr;
    std::vector<cv::Mat> cur_img_dx_pyr, cur_img_dy_pyr;

    std::vector<int> keyframe_ids;

    int height, width;
    std::string image_data_path;
    std::string image_timestamp_path;
    std::vector<std::string> image_file_names;
    std::vector<io::timestamp_t> image_timestamps;
    int image_data_size;

    int last_keyframe_id;
    int global_feature_id;
};
