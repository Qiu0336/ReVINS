#include "settings.h"


Eigen::Matrix3d RbcG;
Eigen::Vector3d tbcG;
Eigen::Matrix3d RcbG;
Eigen::Vector3d tcbG;
Eigen::Matrix5d TcbG;
Eigen::Matrix5d TbcG;


int wG, hG;
double fxG, fyG, cxG, cyG;
double fxiG, fyiG, cxiG, cyiG;
Eigen::Matrix3d KG, KiG;
CameraModel::CameraPtr global_camera;


double gravity_magnitude = 9.81;
Eigen::Vector3d gravity_vector(0.0, 0.0, -gravity_magnitude);
int imu_rate;
Eigen::Matrix12d imu_sigma;


int imu_aided_track;
int image_clahe;
int max_feature_num;
float fundamental_ransac_threshold;
int consider_illumination;
int consider_affine;

int half_patch_size;
int pyramid_levels;
Eigen::Matrix<double, 2, 4> patch_four_corners;
Eigen::Matrix<double, 4, 2> affine_right_multiplyer;


float pub_interval;
float keyframe_ave_parallax;
int keyframe_tracked_num;
float triangulate_parallax;
int sliding_window_size;
int initial_window_size;
int loop_closure_enable;
double feature_nosie_in_pixel;

IntegrationMode integration_mode;
std::string no_loop_save_path;
std::string loop_save_path;

void LoadConfigFile(const std::string &file)
{
    cv::FileStorage fSettings(file, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }


    std::vector<double> extrinsics;
    fSettings["extrinsics"] >> extrinsics;
    RbcG << extrinsics[0], extrinsics[1], extrinsics[2],
            extrinsics[4], extrinsics[5], extrinsics[6],
            extrinsics[8], extrinsics[9], extrinsics[10];
    tbcG << extrinsics[3], extrinsics[7], extrinsics[11];
    RcbG = RbcG.transpose();
    tcbG = - RcbG*tbcG;
    TcbG.setIdentity();
    TcbG.block<3, 3>(0, 0) = RcbG;
    TcbG.block<3, 1>(0, 4) = tcbG;
    TbcG = InverseSE23(TcbG);


    double ng = fSettings["gyro_noise_density"];
    double na = fSettings["acce_noise_density"];
    double bg = fSettings["gyro_random_walk"];
    double ba = fSettings["acce_random_walk"];

    imu_rate = fSettings["imu_rate"];
    imu_sigma.setZero();

    imu_sigma.block<3, 3>(0, 0) = ng*ng*Eigen::Matrix3d::Identity();
    imu_sigma.block<3, 3>(3, 3) = na*na*Eigen::Matrix3d::Identity();
    imu_sigma.block<3, 3>(6, 6) = bg*bg*Eigen::Matrix3d::Identity();
    imu_sigma.block<3, 3>(9, 9) = ba*ba*Eigen::Matrix3d::Identity();


    imu_aided_track = fSettings["imu_aided_track"];
    global_camera = CameraModel::CameraFactory::instance()->generateCameraFromYamlFile(file);

    image_clahe = fSettings["image_clahe"];
    max_feature_num = fSettings["max_feature_num"];
    fundamental_ransac_threshold = fSettings["fundamental_ransac_threshold"];
    consider_illumination = fSettings["consider_illumination"];
    consider_affine = fSettings["consider_affine"];

    pyramid_levels = fSettings["pyramid_levels"];
    if(pyramid_levels > MAX_PARAMID_LEVELS)
        pyramid_levels = MAX_PARAMID_LEVELS;

    half_patch_size = fSettings["half_patch_size"];
    patch_four_corners <<
    - half_patch_size,   half_patch_size, - half_patch_size, half_patch_size,
    - half_patch_size, - half_patch_size,   half_patch_size, half_patch_size;

    affine_right_multiplyer = patch_four_corners.transpose()*
            (patch_four_corners*patch_four_corners.transpose()).inverse();

    float pub_frequent = fSettings["pub_frequent"];
    if(pub_frequent <= 0) pub_frequent = 10;
    pub_interval = 1.f/pub_frequent - 1e-2;
    keyframe_ave_parallax = fSettings["keyframe_ave_parallax"];
    keyframe_tracked_num = fSettings["keyframe_tracked_num"];
    triangulate_parallax = fSettings["triangulate_parallax"];
    sliding_window_size = fSettings["sliding_window_size"];
    initial_window_size = fSettings["initial_window_size"];
    feature_nosie_in_pixel = fSettings["feature_nosie_in_pixel"];
    loop_closure_enable = fSettings["loop_closure_enable"];

    int int_mode = fSettings["integration_mode"];
    switch(int_mode)
    {
        case 1: integration_mode = IntegrationMode::SE23; break;
        case 2: integration_mode = IntegrationMode::SO3; break;
        case 3: integration_mode = IntegrationMode::SE3; break;
        case 4: integration_mode = IntegrationMode::Quaternion; break;
        default: integration_mode = IntegrationMode::SE23;
        std::cerr << "faulty in integration mode setting!";
    }
}

void SetCameraCalib(const std::string &file)
{
    cv::FileStorage fSettings(file, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }

    std::vector<int> resolution;
    std::vector<double> intrinsic;
    fSettings["resolution"] >> resolution;
    fSettings["intrinsics"] >> intrinsic;
    wG = resolution[0];
    hG = resolution[1];
    fxG = intrinsic[0];
    fyG = intrinsic[1];
    cxG = intrinsic[2];
    cyG = intrinsic[3];

    KG.setIdentity();
    KG(0, 0) = fxG;
    KG(1, 1) = fyG;
    KG(0, 2) = cxG;
    KG(1, 2) = cyG;

    KiG = KG.inverse();
    fxiG = KiG(0,0);
    fyiG = KiG(1,1);
    cxiG = KiG(0,2);
    cyiG = KiG(1,2);
}

Eigen::Vector2d Pixel2Norm(const Eigen::Vector2d &pixel)
{
    Eigen::Vector2d norm;
    global_camera->pixel2norm(pixel, norm);
    return norm;
}
Eigen::Vector3d Pixel2Hnorm(const Eigen::Vector2d &pixel)
{
    return Pixel2Norm(pixel).homogeneous();
}

Eigen::Vector2d Norm2Pixel(const Eigen::Vector2d &norm, bool distort)
{
    Eigen::Vector2d pixel;
    global_camera->norm2pixel(norm, pixel, distort);
    return pixel;
}

Eigen::Vector2d Mappoint2Pixel(const Eigen::Vector3d &mappoint, bool distort)
{
    return Norm2Pixel(mappoint.hnormalized(), distort);
}
