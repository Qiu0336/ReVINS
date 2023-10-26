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
//Eigen
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "utils/lie_group.h"
#include "utils/io.h"
#include "camera/camera.h"

#define MAX_PARAMID_LEVELS 5


extern Eigen::Matrix3d RbcG;
extern Eigen::Vector3d tbcG;
extern Eigen::Matrix3d RcbG;
extern Eigen::Vector3d tcbG;
extern Eigen::Matrix5d TcbG;
extern Eigen::Matrix5d TbcG;

extern int wG, hG;
extern double fxG, fyG, cxG, cyG;
extern double fxiG, fyiG, cxiG, cyiG;
extern Eigen::Matrix3d KG,KiG;

extern double gravity_magnitude;
extern Eigen::Vector3d gravity_vector;
extern int imu_rate;
extern Eigen::Matrix12d imu_sigma;

extern int imu_aided_track;
extern int image_clahe;
extern int max_feature_num;
extern float fundamental_ransac_threshold;
extern int consider_illumination;
extern int consider_affine;

extern int half_patch_size;
extern int pyramid_levels;
extern Eigen::Matrix<double, 2, 4> patch_four_corners;
extern Eigen::Matrix<double, 4, 2> affine_right_multiplyer;


extern float pub_interval;
extern float keyframe_ave_parallax;
extern int keyframe_tracked_num;
extern float triangulate_parallax;
extern int sliding_window_size;
extern int initial_window_size;
extern int loop_closure_enable;
extern double feature_nosie_in_pixel;

enum class IntegrationMode {Quaternion, SO3, SE3, SE23};
extern IntegrationMode integration_mode;

extern std::string no_loop_save_path;
extern std::string loop_save_path;

void LoadConfigFile(const std::string &file);
void SetCameraCalib(const std::string &file);
Eigen::Vector2d Pixel2Norm(const Eigen::Vector2d &pixel);
Eigen::Vector3d Pixel2Hnorm(const Eigen::Vector2d &pixel);
Eigen::Vector2d Norm2Pixel(const Eigen::Vector2d &norm, bool distort = true);
Eigen::Vector2d Mappoint2Pixel(const Eigen::Vector3d &mappoint, bool distort = true);
