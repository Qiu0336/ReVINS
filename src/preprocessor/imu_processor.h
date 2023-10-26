#pragma once

#include <string>
#include <cstring>
#include <vector>
#include <list>
#include <iostream>
#include <fstream>
#include <sstream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include "utils/io.h"
#include "utils/lie_group.h"
#include "settings.h"

class ImuIntegration
{
    public:
    ImuIntegration(const IntegrationMode mode_ = IntegrationMode::SE23, const Eigen::Vector6d bias_ = Eigen::Vector6d::Zero());
    ~ImuIntegration() {}
    void Clear();
    void Initialize();
    void SetBias(const Eigen::Vector6d bias);

    Eigen::Matrix5d GetUpdated_r(const Eigen::Vector6d& new_bias) const;
    Eigen::Matrix5d GetUpdated_r(const Eigen::Vector3d& new_bg, const Eigen::Vector3d& new_ba) const
    {
        Eigen::Vector6d new_bias;
        new_bias << new_bg, new_ba;
        return GetUpdated_r(new_bias);
    }


    Eigen::Quaterniond Get_dQ() const;
    Eigen::Matrix3d Get_dR() const;
    Eigen::Vector3d Get_dV() const;
    Eigen::Vector3d Get_dP() const;
    Eigen::Matrix5d Get_dr() const;
    double Get_dT() const;

    Eigen::Matrix3d Get_JRg() const;
    Eigen::Matrix3d Get_JVg() const;
    Eigen::Matrix3d Get_JVa() const;
    Eigen::Matrix3d Get_JPg() const;
    Eigen::Matrix3d Get_JPa() const;
    Eigen::Matrix<double, 9, 6> Get_Jrb() const;

    void IntegrateNewMeasurement(const Eigen::Vector3d& w1, const Eigen::Vector3d& w2,
                                 const Eigen::Vector3d& a1, const Eigen::Vector3d& a2,
                                 const double dt);

    bool IntegrationInterval(const io::ImuData& imu_data_, io::timestamp_t a, io::timestamp_t b,
                             bool record = false);
    bool Reintegrated();
    bool MergeIntegration(std::shared_ptr<ImuIntegration>& integ);
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    public:
    IntegrationMode mode;

    double dT;
    Eigen::Matrix5d r;
    Eigen::Matrix<double, 9, 6> Jrb;

    Eigen::Matrix15d Cov;
    Eigen::Matrix15d Info, SqrtInfo;
    Eigen::Matrix9d InfoPose, SqrtInfoPose;
    Eigen::Matrix5d Gamma;
    Eigen::Vector6d b;
    io::ImuData imu_data;
    io::timestamp_t time_start, time_end;
};

class ImuProcessor
{
public:
    ImuProcessor(const std::string &yaml_path);
    ImuProcessor(const io::ImuData &imu_data_);
    io::imu_data_t GetImuData(int id);
    io::timestamp_t GetTimestamp(int id);
    int GetIdxBeforeEqual(io::timestamp_t t);
    int GetIdxAfterEqual(io::timestamp_t t);
    std::shared_ptr<ImuIntegration> GetImuIntegration(io::timestamp_t t1, io::timestamp_t t2, IntegrationMode mode = IntegrationMode::SE23);

    void SetImuBias(Eigen::Vector6d& bias);

    std::vector<ImuIntegration*> preloaded_imu_integration;

    std::mutex m_imu_bias;
    Eigen::Vector6d imu_bias;

    std::string imu_data_path;
    io::ImuData imu_data;
    int imu_data_size;
    io::timestamp_t timestamp_start;
    io::timestamp_t timestamp_end;

};

