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
#include <thread>
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "utils/lie_group.h"
#include "camera/camera.h"

#include "preprocessor/image_processor.h"
#include "preprocessor/imu_processor.h"
#include "optimization/estimator.h"
#include "loopclosing/pose_graph.h"
#include "settings.h"
#include "drawer.h"

int main()
{
    std::string YamlPath = std::string(DATA_DIR) + "config.yaml";
    no_loop_save_path = std::string(DATA_DIR) + "result/result_no_loop.tum";
    loop_save_path = std::string(DATA_DIR) + "result/result_loop.tum";

    LoadConfigFile(YamlPath);

    std::shared_ptr<ImageProcessor> image_processor = std::make_shared<ImageProcessor>(YamlPath);
    std::shared_ptr<ImuProcessor> imu_processor = std::make_shared<ImuProcessor>(YamlPath);

    SetCameraCalib(YamlPath);
    std::shared_ptr<Estimator> estimator = std::make_shared<Estimator>();

    std::shared_ptr<PoseGraph> posegraph = std::make_shared<PoseGraph>();
    std::shared_ptr<Drawer> drawer = std::make_shared<Drawer>();

    estimator->imu_processor_ptr = imu_processor;
    estimator->pose_graph_ptr = posegraph;
    posegraph->image_processor_ptr = image_processor;
    drawer->estimator_ptr = estimator;
    drawer->posegraph_ptr = posegraph;
    std::shared_ptr<std::thread> estimator_thread = std::make_shared<std::thread>(&Estimator::Run, estimator);
    std::shared_ptr<std::thread> posegraph_thread = std::make_shared<std::thread>(&PoseGraph::Run, posegraph);
    std::shared_ptr<std::thread> drawer_thread = std::make_shared<std::thread>(&Drawer::Run, drawer);

    const int image_size = image_processor->image_data_size;
    int image_id = 0;
    io::timestamp_t timestamp_image = image_processor->GetTimestamp(image_id);

    while(timestamp_image < imu_processor->timestamp_start)
    {
        image_id++;
        if(image_id >= image_size)
        {
            std::cout << "image timestamp can not aligned !!!" << std::endl;
            return 0;
        }
        timestamp_image = image_processor->GetTimestamp(image_id);
    }

    bool first_image = true;
    io::timestamp_t last_time;
    io::timestamp_t cur_time;

    io::timestamp_t last_pub_time;

    for(; image_id < image_size; ++image_id)
    {
        Timer timer;
        timer.Start();
        bool is_keyframe;
        cur_time = image_processor->GetTimestamp(image_id);
        std::map<int, Eigen::Vector2d> frame_features;

        if(first_image)
        {
            first_image = false;
            std::shared_ptr<ImuIntegration> cur_imu = std::make_shared<ImuIntegration>(integration_mode);
            cv::Mat cur_image = image_processor->GetImage(image_id);

            cv::imshow("image", cur_image);
            cv::waitKey(1);

            if(image_clahe)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(cur_image, cur_image);
            }

            std::map<int, Eigen::Isometry3d> pose_list;
            std::map<int, Eigen::Vector3f> mappoint_list;
            frame_features = image_processor->GetTrackedFeatures(image_id, cur_image, is_keyframe, pose_list, mappoint_list);

            estimator->PushBuf(image_id, is_keyframe, frame_features, cur_imu);
            last_time = cur_time;
            last_pub_time = cur_time;
            continue;
        }
        else if((cur_time - last_pub_time)*1e-9 > pub_interval)
        {
            std::shared_ptr<ImuIntegration> cur_imu = imu_processor->GetImuIntegration(last_pub_time, cur_time, integration_mode);
            if(cur_imu == nullptr)
            {
                std::cout << "imu data ended ---" << std::endl;
                break;
            }

            cv::Mat cur_image = image_processor->GetImage(image_id);

            cv::imshow("image", cur_image);
            cv::waitKey(1);

            if(image_clahe)
            {
                cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
                clahe->apply(cur_image, cur_image);
            }

            std::map<int, Eigen::Isometry3d> pose_list;
            std::map<int, Eigen::Vector3f> mappoint_list;
            estimator->GetState(image_id, cur_imu, pose_list, mappoint_list);

            frame_features = image_processor->GetTrackedFeatures(image_id, cur_image, is_keyframe, pose_list, mappoint_list);
            estimator->PushBuf(image_id, is_keyframe, frame_features, cur_imu);
            last_pub_time = cur_time;
        }

        double t_process = timer.ElapsedSeconds();
        double T = (cur_time - last_time)*1e-9;
        if(t_process < T)
            usleep((T - t_process)*1e6);
        last_time = cur_time;
    }

    estimator->SetQuit(true);
    estimator_thread->join();

    posegraph->SetQuit(true);
    posegraph_thread->join();

    drawer->SetQuit(true);
    drawer_thread->join();

    return 0;
}
