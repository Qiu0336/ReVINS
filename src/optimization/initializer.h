#pragma once

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <unistd.h>
#include <list>
#include <algorithm>
#include <vector>
#include <numeric>
#include <map>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "utils/io.h"
#include "utils/lie_group.h"
#include "settings.h"
#include "preprocessor/imu_processor.h"
#include "feature_manager.h"


class GyroscopeBiasCostFunction : public ceres::SizedCostFunction<3, 3>
{
    public:
    GyroscopeBiasCostFunction(const std::shared_ptr<ImuIntegration> pInt, const Eigen::Matrix3d& Ri, const Eigen::Matrix3d& Rj)
    : pInt(pInt), Ri(Ri), Rj(Rj)
    {
        SqrtInformation.setIdentity();
    }
    virtual ~GyroscopeBiasCostFunction() { }

    bool Evaluate(double const* const* parameters,
                  double* residuals,
                  double** jacobians) const override
    {
        Eigen::Map<const Eigen::Vector3d> bg(parameters[0]);

        Eigen::Vector6d bg6d;
        bg6d.head(3) = bg;
        bg6d.tail(3).setZero();
        const Eigen::Matrix3d R = pInt->GetUpdated_r(bg6d).block<3, 3>(0, 0);
        const Eigen::Matrix3d eR = R.transpose()*Ri.transpose()*Rj;
        const Eigen::Vector3d er = LogSO3(eR);
        Eigen::Map<Eigen::Vector3d> e(residuals);
        e = er;
        e = SqrtInformation*e;

        if (jacobians != nullptr)
        {
            if (jacobians[0] != nullptr)
            {
                const Eigen::Vector3d dbg = bg - pInt->b.head<3>();
                const Eigen::Matrix3d invJr = InverseRightJacobianSO3(er);

                Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = -invJr*eR.transpose()*RightJacobianSO3(pInt->Get_JRg()*dbg)*pInt->Get_JRg();
                J = SqrtInformation*J;
            }
        }
        return true;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    std::shared_ptr<ImuIntegration> pInt;
    Eigen::Matrix3d Ri, Rj;
    Eigen::Matrix3d SqrtInformation;
};



class Initializer
{
public:
    Initializer()
    {
        frame_size = 0;
        feature_manager = std::make_shared<FeatureManager>();
    };

    void Push(int cur_frame_id, bool cur_is_keyframe,
              std::map<int, Eigen::Vector2d>& cur_frame_features,
              std::shared_ptr<ImuIntegration>& cur_integration);

    void Apply(std::shared_ptr<FeatureManager> &feature_manager_,
               std::vector<std::shared_ptr<ImuIntegration>> &integrations_,
               std::vector<Eigen::Matrix5d> &Ts_,
               std::vector<Eigen::Vector6d> &Bs_,
               std::vector<int> &frame_ids_,
               bool &last_is_keyframe_);

    bool Run();
    bool InertialInit();
    void SolveGyroscopeBias();
    bool LinearAlignment();

    std::vector<int> frame_ids;
    std::vector<bool> is_keyframes;
    std::vector<std::shared_ptr<ImuIntegration>> integrations;
    std::shared_ptr<FeatureManager> feature_manager;
    int frame_size;
    std::vector<Eigen::Matrix3d> vRwc;
    std::vector<Eigen::Vector3d> vtwc;
    Eigen::Vector6d Bs;
    Eigen::Vector3d g;
    std::vector<Eigen::Vector3d> Vs;
    std::vector<Eigen::Matrix5d> Ts;
    double s;
};


