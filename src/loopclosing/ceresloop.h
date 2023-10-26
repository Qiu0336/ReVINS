#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "utils/lie_group.h"
#include "settings.h"
#include "camera/camera.h"

class FixedPose: public ceres::SizedCostFunction<2, 3>
{
    public:
    FixedPose(Eigen::Matrix3d& _Rwc, Eigen::Vector3d& _twc, Eigen::Vector2d& _puv_i) : Rwc(_Rwc), twc(_twc), puv_i(_puv_i)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> mappoint(parameters[0]);

        const Eigen::Vector3d pi = Rwc.transpose()*(mappoint - twc);
        const double dep = pi(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = pi.hnormalized() - puv_i;

        if(jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - pi(0)/(dep*dep),
                            0, 1.0/dep, - pi(1)/(dep*dep);

            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[0]);
                J = reduce*Rwc.transpose();
            }
        }
        return true;
    }
    Eigen::Matrix3d Rwc;
    Eigen::Vector3d twc;
    Eigen::Vector2d puv_i;
};

class RelaxedPose: public ceres::SizedCostFunction<2, 9, 3, 3>
{
    public:
    RelaxedPose(Eigen::Vector2d& _puv_i, Eigen::Matrix3d _Rji = Eigen::Matrix3d::Identity(),
                Eigen::Vector3d _tji = Eigen::Vector3d::Zero()) :
        puv_i(_puv_i), Rji(_Rji), tji(_tji)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Rcw(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> tcw(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> mappoint(parameters[2]);

        const Eigen::Vector3d pi = Rji*(Rcw*mappoint + tcw) + tji;
        const double dep = pi(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = pi.hnormalized() - puv_i;

        if(jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - pi(0)/(dep*dep),
                            0, 1.0/dep, - pi(1)/(dep*dep);

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - Rji*Rcw*Skew(mappoint);
                J.leftCols(3) = reduce*jaco;
                J.rightCols(6).setZero();
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = reduce*Rji;
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[2]);
                J = reduce*Rji*Rcw;
            }
        }
        return true;
    }
    Eigen::Vector2d puv_i;
    Eigen::Matrix3d Rji;
    Eigen::Vector3d tji;
};


template<typename T = double>
T NormalizeAngle(const T& angle_degrees)
{
    if (angle_degrees > T(180.0))
        return angle_degrees - T(360.0);
    else if (angle_degrees < T(-180.0))
        return angle_degrees + T(360.0);
    else
        return angle_degrees;
};

class AngleLocalParameterization
{
    public:

    template <typename T>
    bool operator()(const T* theta_radians, const T* delta_theta_radians, T* theta_radians_plus_delta) const
    {
        *theta_radians_plus_delta = NormalizeAngle(*theta_radians + *delta_theta_radians);
        return true;
    }
    static ceres::LocalParameterization* Create()
    {
        return (new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>);
    }
};

template <typename T>
void YawPitchRollToRotationMatrix(const T yaw, const T pitch, const T roll, T R[9])
{

    T y = yaw / T(180.0) * T(M_PI);
    T p = pitch / T(180.0) * T(M_PI);
    T r = roll / T(180.0) * T(M_PI);

    R[0] = cos(y) * cos(p);
    R[1] = -sin(y) * cos(r) + cos(y) * sin(p) * sin(r);
    R[2] = sin(y) * sin(r) + cos(y) * sin(p) * cos(r);
    R[3] = sin(y) * cos(p);
    R[4] = cos(y) * cos(r) + sin(y) * sin(p) * sin(r);
    R[5] = -cos(y) * sin(r) + sin(y) * sin(p) * cos(r);
    R[6] = -sin(p);
    R[7] = cos(p) * sin(r);
    R[8] = cos(p) * cos(r);
};

template <typename T>
void RotationMatrixTranspose(const T R[9], T inv_R[9])
{
    inv_R[0] = R[0];
    inv_R[1] = R[3];
    inv_R[2] = R[6];
    inv_R[3] = R[1];
    inv_R[4] = R[4];
    inv_R[5] = R[7];
    inv_R[6] = R[2];
    inv_R[7] = R[5];
    inv_R[8] = R[8];
};

template <typename T>
void RotationMatrixRotatePoint(const T R[9], const T t[3], T r_t[3])
{
    r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
    r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
    r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
};


struct Loop4DoF
{
    Loop4DoF(double t_x, double t_y, double t_z, double relative_yaw, double pitch_i, double roll_i, double _wt, double _wr)
        :t_x(t_x), t_y(t_y), t_z(t_z), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i), weight_t(_wt) , weight_r(_wr){}

    template <typename T>
    bool operator()(const T* const yaw_i, const T* ti, const T* yaw_j, const T* tj, T* residuals) const
    {
        T t_w_ij[3];
        t_w_ij[0] = tj[0] - ti[0];
        t_w_ij[1] = tj[1] - ti[1];
        t_w_ij[2] = tj[2] - ti[2];

        T w_R_i[9];
        YawPitchRollToRotationMatrix(yaw_i[0], T(pitch_i), T(roll_i), w_R_i);
        T i_R_w[9];
        RotationMatrixTranspose(w_R_i, i_R_w);
        T t_i_ij[3];
        RotationMatrixRotatePoint(i_R_w, t_w_ij, t_i_ij);

        residuals[0] = (t_i_ij[0] - T(t_x))*weight_t;
        residuals[1] = (t_i_ij[1] - T(t_y))*weight_t;
        residuals[2] = (t_i_ij[2] - T(t_z))*weight_t;
        residuals[3] = NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw)) * weight_r;

        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z,
                                       const double relative_yaw, const double pitch_i, const double roll_i, const double _wt, const double _wr)
    {
      return (new ceres::AutoDiffCostFunction<
              Loop4DoF, 4, 1, 3, 1, 3>(
                  new Loop4DoF(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i, _wt, _wr)));
    }

    double t_x, t_y, t_z;
    double relative_yaw, pitch_i, roll_i;
    double weight_t, weight_r;

};

class Loop6DoF: public ceres::SizedCostFunction<6, 6, 6>
{
    public:
    Loop6DoF(Eigen::Isometry3d &_Tij) : Tij(_Tij)
    {}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector6d> ti(parameters[0]);
        Eigen::Map<const Eigen::Vector6d> tj(parameters[1]);

        const Eigen::Isometry3d vTij = ExpSE3(ti).inverse()*ExpSE3(tj);

        Eigen::Map<Eigen::Vector6d> residual(residuals);
        Eigen::Vector6d Log_er = LogSE3(Tij.inverse()*vTij);
        residual = Log_er;

        if(jacobians)
        {
            const Eigen::Matrix6d JrSE3_rij= InverseRightJacobianSE3(Log_er);
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[0]);
                J = - JrSE3_rij*AdjointSE3(vTij.inverse());
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobians[1]);
                J = JrSE3_rij;
            }
        }
        return true;
    }
    Eigen::Isometry3d Tij;
};


class Loop7DoF: public ceres::SizedCostFunction<6, 3, 3, 1, 3, 3, 1>
{
    public:
    Loop7DoF(Eigen::Isometry3d &_Tij) : Tij(_Tij)
    {
        Eigen::Isometry3d Tji = _Tij.inverse();
        Rji = Tji.linear();
        tji = Tji.translation();
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector3d> fai_i(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twi(parameters[1]);
        const double si = parameters[2][0];
        Eigen::Map<const Eigen::Vector3d> fai_j(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> twj(parameters[4]);
        const double sj = parameters[5][0];

        const Eigen::Matrix3d Rwi = ExpSO3(fai_i);
        const Eigen::Matrix3d Rwj = ExpSO3(fai_j);

        Eigen::Map<Eigen::Vector6d> residual(residuals);
        Eigen::Matrix3d resR = Rwi.transpose()*Rwj*Rji;
        Eigen::Vector3d log_er = LogSO3(resR);
        residual.head<3>() = log_er;
        Eigen::Vector3d tmp1 = Rwi.transpose()*(exp(sj - si)*Rwj*tji + exp(-si)*(twj - twi));
        residual.tail<3>() = tmp1;


        if(jacobians)
        {
            const Eigen::Matrix3d invJrSO3 = InverseRightJacobianSO3(log_er);

            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[0]);
                J.setZero();
                J.block<3, 3>(0, 0) = -invJrSO3*resR.transpose();
                J.block<3, 3>(3, 0) = Skew(tmp1);
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3, 3>(3, 0) = -exp(-si)*Rwi.transpose();
            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Vector6d> J(jacobians[2]);
                J.setZero();
                J.tail<3>() = -tmp1;
            }
            if(jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[3]);
                J.block<3, 3>(0, 0) = invJrSO3*Rji.transpose();
                J.block<3, 3>(3, 0) = -exp(sj - si)*Rwi.transpose()*Rwj*Skew(tji);
            }
            if(jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 6, 3, Eigen::RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3, 3>(3, 0) = exp(-si)*Rwi.transpose();
            }
            if(jacobians[5])
            {
                Eigen::Map<Eigen::Vector6d> J(jacobians[5]);
                J.setZero();
                J.tail<3>() = exp(sj - si)*Rwi.transpose()*Rwj*tji;
            }
        }
        return true;
    }
    Eigen::Isometry3d Tij;
    Eigen::Matrix3d Rji;
    Eigen::Vector3d tji;
};

