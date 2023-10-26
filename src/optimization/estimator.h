#pragma once
#include <iostream>
#include <algorithm>
#include <cmath>
// STL
#include <iterator>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <thread>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include "preprocessor/image_processor.h"
#include "preprocessor/imu_processor.h"
#include "feature_manager.h"
#include "initializer.h"
#include "margin.h"
#include "loopclosing/pose_graph.h"
#include "settings.h"

class Estimator
{
public:
    Estimator(){
        quit_flag = false;
        is_initialized = false;
        reintegrated_num = 0;
        camera_pose.setIdentity();

        last_marginalization_info = nullptr;
        if(integration_mode == IntegrationMode::SE23)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                double* T_ptr = new double[9];
                double* B_ptr = new double[6];
                para_T.push_back(T_ptr);
                para_B.push_back(B_ptr);
            }
        }
        else if(integration_mode == IntegrationMode::SO3)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                double* R_ptr = new double[9];
                double* V_ptr = new double[3];
                double* P_ptr = new double[3];
                double* bg_ptr = new double[3];
                double* ba_ptr = new double[3];
                para_R.push_back(R_ptr);
                para_V.push_back(V_ptr);
                para_P.push_back(P_ptr);
                para_bg.push_back(bg_ptr);
                para_ba.push_back(ba_ptr);
            }
        }
        else if(integration_mode == IntegrationMode::Quaternion)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                double* Q_ptr = new double[4];
                double* V_ptr = new double[3];
                double* P_ptr = new double[3];
                double* bg_ptr = new double[3];
                double* ba_ptr = new double[3];
                para_Q.push_back(Q_ptr);
                para_V.push_back(V_ptr);
                para_P.push_back(P_ptr);
                para_bg.push_back(bg_ptr);
                para_ba.push_back(ba_ptr);
            }
        }
        for(int i = 0; i < 2000; ++i)
        {
            double* ivd = new double[1];
            para_invdep.push_back(ivd);
        }
        initializer = std::make_shared<Initializer>();
    }
    ~Estimator()
    {
        if(integration_mode == IntegrationMode::SE23)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                delete [] para_T[i];
                delete [] para_B[i];
            }
        }
        else if(integration_mode == IntegrationMode::SO3)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                delete [] para_R[i];
                delete [] para_V[i];
                delete [] para_P[i];
                delete [] para_bg[i];
                delete [] para_ba[i];
            }
        }
        else if(integration_mode == IntegrationMode::Quaternion)
        {
            for(int i = 0; i <= sliding_window_size; ++i)
            {
                delete [] para_Q[i];
                delete [] para_V[i];
                delete [] para_P[i];
                delete [] para_bg[i];
                delete [] para_ba[i];
            }
        }
        for(int i = 0; i < 2000; ++i)
        {
            delete [] para_invdep[i];
        }
    }

    void PushBuf(int frame_id, bool is_keyframe, std::map<int, Eigen::Vector2d>& frame_features,
                 std::shared_ptr<ImuIntegration>& imu_integration);
    void Run();
    void UpdateCameraCSbyTs();
    void PubKeyframe();
    bool GetState(const int cur_id, const std::shared_ptr<ImuIntegration>& cur_imu,
                  std::map<int, Eigen::Isometry3d>& pose_list, std::map<int, Eigen::Vector3f>& mappoint_list);
    Eigen::Isometry3d GetCameraPose();
    void SetQuit(bool x);
    bool GetQuit();


    void Optimization();
    void Mat2Para();
    void Para2Mat();


    std::vector<double*> para_T, para_B;

    std::vector<double*> para_Q, para_R, para_V, para_P, para_bg, para_ba;
    std::vector<double*> para_invdep;

    MarginalizationInfo* last_marginalization_info;
    std::vector<double*> last_marginalization_parameter_blocks;

    std::shared_ptr<FeatureManager> feature_manager;
    std::vector<std::shared_ptr<ImuIntegration>> integrations;

    std::mutex m_busy;
    std::vector<Eigen::Matrix5d> Ts;
    std::vector<Eigen::Vector6d> Bs;

    std::vector<int> frame_ids;

    int frame_size;
    bool last_is_keyframe;


    std::vector<Eigen::Matrix3d> vRcw;
    std::vector<Eigen::Vector3d> vtcw;

    std::mutex m_camera_pose;
    Eigen::Isometry3d camera_pose;

    bool is_initialized;
    std::shared_ptr<Initializer> initializer;

    std::shared_ptr<PoseGraph> pose_graph_ptr;
    std::shared_ptr<ImuProcessor> imu_processor_ptr;
    int reintegrated_num;

    std::mutex m_buf;
    std::queue<int> frame_id_buf;
    std::queue<bool> is_keyframe_buf;
    std::queue<std::map<int, Eigen::Vector2d>> frame_features_buf;
    std::queue<std::shared_ptr<ImuIntegration>> imu_integration_buf;

    std::mutex m_quit;
    bool quit_flag;
};
