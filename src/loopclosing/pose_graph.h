#pragma once

#include <iostream>
#include <algorithm>
#include <fstream>
#include <ctime>
#include <sstream>
#include <mutex>
#include <queue>
#include <thread>
#include <unistd.h>
#include <eigen3/Eigen/Dense>

#include <eigen3/Eigen/Sparse>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "utils/io.h"
#include "utils/timer.h"
#include "keyframe.h"
#include "ceresloop.h"
#include "optimization/ceres_factor.h"
#include "preprocessor/image_processor.h"
#include "DBow3/src/DBoW3.h"
#include "DBow3/src/DescManip.h"

#include "pnp.h"

class PnPMapPoint
{
    public:
    PnPMapPoint(){}
    bool good_matched{false};
    bool good_triangulated{false};
    bool good_connected{false};
    int mappoint_id = -1;
    int id;
    Eigen::Vector3d world_point;
    std::map<int, int> loop_observations;
    std::map<int, int> cur_observations;
};

class PoseGraph
{
    public:
    PoseGraph();
    ~PoseGraph()
    {
        delete voc;
    }
    void Run();
    void AddKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop);
    int DetectLoop(std::shared_ptr<KeyFrame> keyframe);
    bool FindConnection(std::shared_ptr<KeyFrame> &kf_cur, const int loop_id);
    std::shared_ptr<KeyFrame> GetKeyFrame(const int index);

    void MatchTwoFrame(const std::shared_ptr<KeyFrame> &kf_query,
                       const std::shared_ptr<KeyFrame> &kf_train,
                       std::vector<cv::DMatch> &matches, const float radius,
                       const std::vector<int> &candidates);

    void MatchTwoFrame(const std::shared_ptr<KeyFrame> &kf_query,
                       const std::shared_ptr<KeyFrame> &kf_train,
                       std::vector<cv::DMatch> &matches, const float radius)
    {
        std::vector<int> candidates;
        candidates.resize(kf_query->GetKeypoints_norm().size());
        std::iota(candidates.begin(), candidates.end(), 0);
        MatchTwoFrame(kf_query, kf_train, matches, radius, candidates);
    }

    void MatchTwoFrameInCircle(const std::shared_ptr<KeyFrame> &kf_query,
                               const std::shared_ptr<KeyFrame> &kf_train,
                               std::vector<cv::DMatch> &matches, const float radius,
                               const std::vector<int> &candidates);

    void PushBuf(const int frame_id, const Eigen::Isometry3d& pose);

    void Optimization_Loop_4DoF();
    void Optimization_Loop_6DoF();
    void SetQuit(bool x);
    bool GetQuit();
    Eigen::Isometry3d DriftRemove(const Eigen::Isometry3d& pose);
    std::vector<std::shared_ptr<KeyFrame>> GetKeyframelist();

    std::vector<std::pair<int, int>> loop_pairs;
    std::vector<Eigen::Isometry3d> drifts;

    void AddNewLoop(int start_id, int end_id, Eigen::Isometry3d drift)
    {
        bool merge = false;
        int i = 0;
        for(; i < loop_pairs.size(); ++i)
        {
            if(loop_pairs[i].first >= start_id)
            {
                loop_pairs[i] = std::make_pair(start_id, end_id);
                drifts[i] = drift;
                ++i;
                merge = true;
                break;
            }
        }
        loop_pairs.resize(i);
        drifts.resize(i);

        if(!merge)
        {
            loop_pairs.emplace_back(start_id, end_id);
            drifts.push_back(drift);
        }
    }
    Eigen::Isometry3d GetDrift(int id)
    {
        Eigen::Isometry3d drift = Eigen::Isometry3d::Identity();

        for(int i = 0; i < loop_pairs.size(); ++i)
        {
            if(loop_pairs[i].second <= id)
            {
                drift = drifts[i];
            }
            else
                break;
        }
        return drift;
    }
    int GetFirstOptFrameId(int id)
    {
        int return_id = id;
        for(int i = 0; i < loop_pairs.size(); ++i)
        {
            if(loop_pairs[i].first < id && loop_pairs[i].second > id)
            {
                return_id = loop_pairs[i].first;
                break;
            }
        }
        return return_id;
    }
    int earliest_loop_index;

    std::mutex m_keyframelist;
    std::vector<std::shared_ptr<KeyFrame>> keyframelist;

    std::mutex m_buf;
    std::queue<int> frame_id_buf;
    std::queue<Eigen::Isometry3d> pose_buf;
    Eigen::Isometry3d T_drift;

    std::mutex m_quit;
    bool quit_flag;

    int keyframe_id;
    int last_loop;


    std::shared_ptr<ImageProcessor> image_processor_ptr;

    DBoW3::Vocabulary* voc;
    DBoW3::Database db;
};
