#pragma once

#include <iostream>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "utils/lie_group.h"
#include "preprocessor/imu_processor.h"
#include "camera/camera.h"
#include "settings.h"


class RotationMatrixParameterization : public ceres::LocalParameterization
{
    public:
    virtual ~RotationMatrixParameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Matrix3d> R(x);
        Eigen::Map<Eigen::Matrix3d> result(x_plus_delta);
        result = R*ExpSO3(delta[0], delta[1], delta[2]);
        return true;
    }


    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 9, 3, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.topRows<3>().setIdentity();
        return true;
    }
    int GlobalSize() const override { return 9; }
    int LocalSize() const override { return 3; }
};


class QuaternionParameterization : public ceres::LocalParameterization
{
    public:
    virtual ~QuaternionParameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Quaterniond> q(x);
        Eigen::Map<const Eigen::Vector3d> ds(delta);
        Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta));
        Eigen::Map<Eigen::Quaterniond> update_q(x_plus_delta);
        update_q = (q * dq).normalized();
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
        J.setZero();
        J.topRows<3>().setIdentity();
        return true;
    }
    int GlobalSize() const override { return 4; }
    int LocalSize() const override { return 3; }
};


class SO3Parameterization : public ceres::LocalParameterization
{
    public:
    virtual ~SO3Parameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector3d> s(x);
        Eigen::Map<const Eigen::Vector3d> ds(delta);
        Eigen::Map<Eigen::Vector3d> update_s(x_plus_delta);
        update_s = LogSO3(ExpSO3(s)*ExpSO3(ds));
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    int GlobalSize() const override { return 3; }
    int LocalSize() const override { return 3; }
};


class SE3Parameterization : public ceres::LocalParameterization
{
    public:
    virtual ~SE3Parameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector6d> s(x);
        Eigen::Map<const Eigen::Vector6d> ds(delta);
        Eigen::Map<Eigen::Vector6d> update_s(x_plus_delta);
        update_s = LogSE3(ExpSE3(s)*ExpSE3(ds));
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    int GlobalSize() const override { return 6; }
    int LocalSize() const override { return 6; }
};


class SE23Parameterization : public ceres::LocalParameterization
{
    public:
    virtual ~SE23Parameterization() {}
    bool Plus(const double* x,
            const double* delta,
            double* x_plus_delta) const override
    {
        Eigen::Map<const Eigen::Vector9d> s(x);
        Eigen::Map<const Eigen::Vector9d> ds(delta);
        Eigen::Map<Eigen::Vector9d> update_s(x_plus_delta);
        update_s = LogSE23(ExpSE23(s)*ExpSE23(ds));
        return true;
    }

    bool ComputeJacobian(const double* x, double* jacobian) const override
    {
        Eigen::Map<Eigen::Matrix<double, 9, 9, Eigen::RowMajor>> J(jacobian);
        J.setIdentity();
        return true;
    }
    int GlobalSize() const override { return 9; }
    int LocalSize() const override { return 9; }
};


class IMUFactorQuaternion : public ceres::SizedCostFunction<15, 4, 3, 3, 4, 3, 3, 3, 3, 3, 3>
{
    public:
    IMUFactorQuaternion(std::shared_ptr<ImuIntegration> _pre_integration): pInt(_pre_integration)
    {
        sqrt_info = pInt->SqrtInfo;
        dt = pInt->dT;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[2]);
        Eigen::Quaterniond Qj(parameters[3][3], parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[4]);
        Eigen::Map<const Eigen::Vector3d> Pj(parameters[5]);
        Eigen::Map<const Eigen::Vector3d> bgi(parameters[6]);
        Eigen::Map<const Eigen::Vector3d> bai(parameters[7]);
        Eigen::Map<const Eigen::Vector3d> bgj(parameters[8]);
        Eigen::Map<const Eigen::Vector3d> baj(parameters[9]);

        const Eigen::Matrix5d delS = pInt->GetUpdated_r(bgi, bai);
        const Eigen::Matrix3d delR = delS.block<3, 3>(0, 0);
        const Eigen::Vector3d delV = delS.block<3, 1>(0, 3);
        const Eigen::Vector3d delP = delS.block<3, 1>(0, 4);

        const Eigen::Quaterniond delQ = Eigen::Quaterniond(delR);

        const Eigen::Vector3d dv = Qi.inverse()*(Vj - Vi - gravity_vector*dt);
        const Eigen::Vector3d dp = Qi.inverse()*(Pj - Pi - Vi*dt - 0.5*gravity_vector*dt*dt);

        Eigen::Map<Eigen::Vector15d> residual(residuals);
        residual.head<3>() = 2*(delQ.inverse()*(Qi.inverse()*Qj)).vec();
        residual.segment<3>(3) = dv - delV;
        residual.segment<3>(6) = dp - delP;
        residual.segment<3>(9) = bgj - bgi;
        residual.tail<3>() = baj - bai;
        residual = sqrt_info*residual;

        if(jacobians)
        {
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 15, 3, Eigen::RowMajor> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = - (Qleft(Qj.inverse()*Qi)*Qright(delQ)).bottomRightCorner<3, 3>();
                jaco.block<3, 3>(3, 0) = Skew(dv);
                jaco.block<3, 3>(6, 0) = Skew(dp);
                jaco = sqrt_info*jaco;
                J.setZero();
                J.leftCols<3>() = jaco;
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3, 3>(3, 0) = - Qi.inverse().toRotationMatrix();
                J.block<3, 3>(6, 0) = - Qi.inverse().toRotationMatrix()*dt;
                J = sqrt_info*J;
            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                J.block<3, 3>(6, 0) = - Qi.inverse().toRotationMatrix();
                J = sqrt_info*J;
            }
            if(jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 4, Eigen::RowMajor>> J(jacobians[3]);
                Eigen::Matrix<double, 15, 3, Eigen::RowMajor> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = Qleft(delQ.inverse()*Qi.inverse()*Qj).bottomRightCorner<3, 3>();
                jaco = sqrt_info*jaco;

                J.setZero();
                J.leftCols<3>() = jaco;
            }
            if(jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3, 3>(3, 0) = Qi.inverse().toRotationMatrix();
                J = sqrt_info*J;
            }
            if(jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[5]);
                J.setZero();
                J.block<3, 3>(6, 0) = Qi.inverse().toRotationMatrix();
                J = sqrt_info*J;
            }
            if(jacobians[6])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[6]);
                J.setZero();
                J.block<3, 3>(0, 0) = - Qleft(Qj.inverse()*Qi*pInt->Get_dQ()).bottomRightCorner<3, 3>()*pInt->Get_JRg();
                J.block<3, 3>(3, 0) = - pInt->Get_JVg();
                J.block<3, 3>(6, 0) = - pInt->Get_JPg();
                J.block<3, 3>(9, 0) = - Eigen::Matrix3d::Identity();
                J = sqrt_info*J;
            }
            if(jacobians[7])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[7]);
                J.setZero();
                J.block<3, 3>(3, 0) = - pInt->Get_JVa();
                J.block<3, 3>(6, 0) = - pInt->Get_JPa();
                J.block<3, 3>(12, 0) = - Eigen::Matrix3d::Identity();
                J = sqrt_info*J;
            }
            if(jacobians[8])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[8]);
                J.setZero();
                J.block<3, 3>(9, 0).setIdentity();
                J = sqrt_info*J;
            }
            if(jacobians[9])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[9]);
                J.setZero();
                J.block<3, 3>(12, 0).setIdentity();
                J = sqrt_info*J;
            }
        }
        return true;
    }
    std::shared_ptr<ImuIntegration> pInt;
    Eigen::Matrix15d sqrt_info;
    double dt;
};


class IMUFactorSO3 : public ceres::SizedCostFunction<15, 9, 3, 3, 9, 3, 3, 3, 3, 3, 3>
{
    public:
    IMUFactorSO3(std::shared_ptr<ImuIntegration> _pre_integration): pInt(_pre_integration)
    {
        sqrt_info = pInt->SqrtInfo;
        dt = pInt->dT;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Ri(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Vi(parameters[1]);
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[2]);
        Eigen::Map<const Eigen::Matrix3d> Rj(parameters[3]);
        Eigen::Map<const Eigen::Vector3d> Vj(parameters[4]);
        Eigen::Map<const Eigen::Vector3d> Pj(parameters[5]);
        Eigen::Map<const Eigen::Vector3d> bgi(parameters[6]);
        Eigen::Map<const Eigen::Vector3d> bai(parameters[7]);
        Eigen::Map<const Eigen::Vector3d> bgj(parameters[8]);
        Eigen::Map<const Eigen::Vector3d> baj(parameters[9]);

        const Eigen::Matrix5d delS = pInt->GetUpdated_r(bgi, bai);

        const Eigen::Matrix3d delR = delS.block<3, 3>(0, 0);
        const Eigen::Vector3d delV = delS.block<3, 1>(0, 3);
        const Eigen::Vector3d delP = delS.block<3, 1>(0, 4);


        Eigen::Map<Eigen::Vector15d> residual(residuals);
        const Eigen::Matrix3d expr = delR.transpose()*Ri.transpose()*Rj;
        const Eigen::Vector3d dv = Ri.transpose()*(Vj - Vi - gravity_vector*dt);
        const Eigen::Vector3d dp = Ri.transpose()*(Pj - Pi - Vi*dt - 0.5*gravity_vector*dt*dt);

        residual.head<3>() = LogSO3(expr);
        residual.segment<3>(3) = dv - delV;
        residual.segment<3>(6) = dp - delP;
        residual.segment<3>(9) = bgj - bgi;
        residual.tail<3>() = baj - bai;

        residual = sqrt_info*residual;

        if(jacobians)
        {
            const Eigen::Matrix3d invJr = InverseRightJacobianSO3(residual.head<3>());
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 15, 3, Eigen::RowMajor> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = - invJr*Rj.transpose()*Ri;
                jaco.block<3, 3>(3, 0) = Skew(dv);
                jaco.block<3, 3>(6, 0) = Skew(dp);
                jaco = sqrt_info*jaco;
                J.setZero();
                J.leftCols<3>() = jaco;
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[1]);
                J.setZero();
                J.block<3, 3>(3, 0) = - Ri.transpose();
                J.block<3, 3>(6, 0) = - Ri.transpose()*dt;
                J = sqrt_info*J;
            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[2]);
                J.setZero();
                J.block<3, 3>(6, 0) = - Ri.transpose();
                J = sqrt_info*J;
            }
            if(jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[3]);
                Eigen::Matrix<double, 15, 3, Eigen::RowMajor> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = invJr;
                jaco = sqrt_info*jaco;

                J.setZero();
                J.leftCols<3>() = jaco;
            }
            if(jacobians[4])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[4]);
                J.setZero();
                J.block<3, 3>(3, 0) = Ri.transpose();
                J = sqrt_info*J;
            }
            if(jacobians[5])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[5]);
                J.setZero();
                J.block<3, 3>(6, 0) = Ri.transpose();
                J = sqrt_info*J;
            }
            if(jacobians[6])
            {
                const Eigen::Vector3d dbg = bgi - pInt->b.head<3>();
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[6]);
                J.setZero();
                J.block<3, 3>(0, 0) = - invJr*expr.transpose()*RightJacobianSO3(pInt->Get_JRg()*dbg)*pInt->Get_JRg();
                J.block<3, 3>(3, 0) = - pInt->Get_JVg();
                J.block<3, 3>(6, 0) = - pInt->Get_JPg();
                J.block<3, 3>(9, 0) = - Eigen::Matrix3d::Identity();
                J = sqrt_info*J;
            }
            if(jacobians[7])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[7]);
                J.setZero();
                J.block<3, 3>(3, 0) = - pInt->Get_JVa();
                J.block<3, 3>(6, 0) = - pInt->Get_JPa();
                J.block<3, 3>(12, 0) = - Eigen::Matrix3d::Identity();
                J = sqrt_info*J;
            }
            if(jacobians[8])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[8]);
                J.setZero();
                J.block<3, 3>(9, 0).setIdentity();
                J = sqrt_info*J;
            }
            if(jacobians[9])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 3, Eigen::RowMajor>> J(jacobians[9]);
                J.setZero();
                J.block<3, 3>(12, 0).setIdentity();
                J = sqrt_info*J;
            }
        }
        return true;
    }
    std::shared_ptr<ImuIntegration> pInt;
    Eigen::Matrix15d sqrt_info;
    double dt;
};


class IMUFactorSE23: public ceres::SizedCostFunction<15, 9, 9, 6, 6>
{
    public:
    IMUFactorSE23(std::shared_ptr<ImuIntegration> _pre_integration): pInt(_pre_integration)
    {
        dt = pInt->dT;
        sqrt_info = pInt->SqrtInfo;
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector9d> si(parameters[0]);
        Eigen::Map<const Eigen::Vector9d> sj(parameters[1]);
        Eigen::Map<const Eigen::Vector6d> bi(parameters[2]);
        Eigen::Map<const Eigen::Vector6d> bj(parameters[3]);

        const Eigen::Matrix5d Twbi = ExpSE23(si);
        const Eigen::Matrix5d Twbj = ExpSE23(sj);

        const Eigen::Matrix5d Inv_e = InverseSE23(Twbj)*pInt->Gamma*FaiSE23(Twbi, dt);
        const Eigen::Matrix5d Inv_er = Inv_e*pInt->GetUpdated_r(bi);
        const Eigen::Matrix5d er = InverseSE23(Inv_er);

        const Eigen::Vector9d Log_er = LogSE23(er);

        Eigen::Map<Eigen::Vector15d> residual(residuals);
        residual.head<9>() = Log_er;
        residual.tail<6>() = bj - bi;
        residual = sqrt_info*residual;

        if(jacobians)
        {
            const Eigen::Matrix9d JrSE23_rij= InverseRightJacobianSE23(Log_er);
            if(jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[0]);
                J.block<9, 9>(0, 0) = - JrSE23_rij*AdjointSE23(Inv_e)*F_Mat_SE23(dt);
                J.block<6, 9>(9, 0).setZero();
                J = sqrt_info*J;
            }
            if(jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J(jacobians[1]);
                J.block<9, 9>(0, 0) = JrSE23_rij;
                J.block<6, 9>(9, 0).setZero();
                J = sqrt_info*J;
            }
            if(jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[2]);
                J.block<9, 6>(0, 0) = - JrSE23_rij*AdjointSE23(Inv_er)*pInt->Jrb;
                J.block<6, 6>(9, 0) = - Eigen::Matrix6d::Identity();
                J = sqrt_info*J;
            }
            if(jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 15, 6, Eigen::RowMajor>> J(jacobians[3]);
                J.setZero();
                J.block<6, 6>(9, 0) = Eigen::Matrix6d::Identity();
                J = sqrt_info*J;
            }
        }
        return true;
    }
    std::shared_ptr<ImuIntegration> pInt;
    double dt;
    Eigen::Matrix15d sqrt_info;
};


class ProjectionFactorQuaternion : public ceres::SizedCostFunction<2, 4, 3, 4, 3, 1>
{
    public:
    ProjectionFactorQuaternion(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j)
        : pts_i(_pts_i), pts_j(_pts_j)
    {
        Eigen::Vector2d p2d = Norm2Pixel(pts_j);
        Eigen::Vector2d dp{feature_nosie_in_pixel, feature_nosie_in_pixel};
        p2d += dp;
        Eigen::Vector2d cov = Pixel2Norm(p2d) - pts_j;
        SqrtInfo.setIdentity();
        SqrtInfo(0, 0) = 1.0/cov(0);
        SqrtInfo(1, 1) = 1.0/cov(1);
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Quaterniond Qi(parameters[0][3], parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[1]);
        Eigen::Quaterniond Qj(parameters[2][3], parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Map<const Eigen::Vector3d> Pj(parameters[3]);
        const double invdep_i = parameters[4][0];

        Eigen::Vector3d Pci = pts_i.homogeneous();
        Pci = Pci/invdep_i;
        Eigen::Vector3d Pbi = RbcG*Pci + tbcG;
        Eigen::Vector3d Pw = Qi*Pbi + Pi;
        Eigen::Vector3d Pbj = Qj.inverse()*(Pw - Pj);
        Eigen::Vector3d pcj = RcbG*Pbj + tcbG;
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        double invdep_j = 1.0/pcj.z();
        residual = pcj.hnormalized() - pts_j;
        residual = SqrtInfo * residual;

        const Eigen::Matrix3d Ri = Qi.toRotationMatrix();
        const Eigen::Matrix3d Rj = Qj.toRotationMatrix();

        if (jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << invdep_j,         0, - pcj.x()*invdep_j*invdep_j,
                              0, invdep_j, - pcj.y()*invdep_j*invdep_j;
            reduce = SqrtInfo * reduce;

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - RcbG*Rj.transpose()*Ri*Skew(Pbi);
                J.setZero();
                J.leftCols<3>() = reduce*jaco;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = RcbG*Rj.transpose();
                J = reduce*jaco;
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J(jacobians[2]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = RcbG*Skew(Pbj);
                J.setZero();
                J.leftCols<3>() = reduce*jaco;
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[3]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - RcbG*Rj.transpose();
                J = reduce*jaco;
            }
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[4]);
                J = - reduce*RcbG*Rj.transpose()*Ri*RbcG*Pci/invdep_i;
            }
        }
        return true;
    }
    Eigen::Vector2d pts_i, pts_j;
    Eigen::Matrix2d SqrtInfo;
};


class ProjectionFactorSO3 : public ceres::SizedCostFunction<2, 9, 3, 9, 3, 1>
{
    public:
    ProjectionFactorSO3(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j)
        : pts_i(_pts_i), pts_j(_pts_j)
    {
        Eigen::Vector2d p2d = Norm2Pixel(pts_j);
        Eigen::Vector2d dp{feature_nosie_in_pixel, feature_nosie_in_pixel};
        p2d += dp;
        Eigen::Vector2d cov = Pixel2Norm(p2d) - pts_j;
        SqrtInfo.setIdentity();
        SqrtInfo(0, 0) = 1.0/cov(0);
        SqrtInfo(1, 1) = 1.0/cov(1);
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Ri(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> Pi(parameters[1]);
        Eigen::Map<const Eigen::Matrix3d> Rj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> Pj(parameters[3]);
        const double invdep_i = parameters[4][0];

        Eigen::Vector3d Pci = pts_i.homogeneous();
        Pci = Pci/invdep_i;
        Eigen::Vector3d Pbi = RbcG*Pci + tbcG;
        Eigen::Vector3d Pw = Ri*Pbi + Pi;
        Eigen::Vector3d Pbj = Rj.transpose()*(Pw - Pj);
        Eigen::Vector3d pcj = RcbG*Pbj + tcbG;
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        double invdep_j = 1.0/pcj.z();
        residual = pcj.hnormalized() - pts_j;
        residual = SqrtInfo * residual;

        if (jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << invdep_j,         0, - pcj.x()*invdep_j*invdep_j,
                              0, invdep_j, - pcj.y()*invdep_j*invdep_j;
            reduce = SqrtInfo * reduce;

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - RcbG*Rj.transpose()*Ri*Skew(Pbi);
                J.leftCols<3>() = reduce*jaco;
                J.rightCols<6>().setZero();
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = RcbG*Rj.transpose();
                J = reduce*jaco;
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[2]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = RcbG*Skew(Pbj);
                J.leftCols<3>() = reduce*jaco;
                J.rightCols<6>().setZero();
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[3]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - RcbG*Rj.transpose();
                J = reduce*jaco;
            }
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[4]);
                J = - reduce*RcbG*Rj.transpose()*Ri*RbcG*Pci/invdep_i;
            }
        }
        return true;
    }
    Eigen::Vector2d pts_i, pts_j;
    Eigen::Matrix2d SqrtInfo;
};


class ProjectionFactorSE23 : public ceres::SizedCostFunction<2, 9, 9, 1>
{
    public:
    ProjectionFactorSE23(const Eigen::Vector2d &_pts_i, const Eigen::Vector2d &_pts_j)
        : pts_i(_pts_i), pts_j(_pts_j)
    {
        Eigen::Vector2d p2d = Norm2Pixel(pts_j);
        Eigen::Vector2d dp{feature_nosie_in_pixel, feature_nosie_in_pixel};
        p2d += dp;
        Eigen::Vector2d cov = Pixel2Norm(p2d) - pts_j;
        SqrtInfo.setIdentity();
        SqrtInfo(0, 0) = 1.0/cov(0);
        SqrtInfo(1, 1) = 1.0/cov(1);
    }
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Vector9d> si(parameters[0]);
        Eigen::Map<const Eigen::Vector9d> sj(parameters[1]);
        const double invdep_i = parameters[2][0];

        Eigen::Vector3d Pci = pts_i.homogeneous();
        Pci = Pci/invdep_i;
        Eigen::Vector5d Pci_SE23;
        Pci_SE23 << Pci, 0, 1.0;

        const Eigen::Matrix5d Twi = ExpSE23(si);
        const Eigen::Matrix5d Twj = ExpSE23(sj);

        const Eigen::Vector5d Pbi_SE23 = TbcG*Pci_SE23;
        const Eigen::Vector5d Pbj_SE23 = InverseSE23(Twj)*Twi*Pbi_SE23;
        const Eigen::Vector5d Pcj_SE23 = TcbG*Pbj_SE23;
        const Eigen::Vector3d Pcj = Pcj_SE23.head<3>();
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        const double invdep_j = 1.0/Pcj.z();
        residual = Pcj.hnormalized() - pts_j;
        residual = SqrtInfo * residual;

        if (jacobians)
        {
            Eigen::Matrix<double, 2, 5> reduce;
            reduce << invdep_j,         0, - Pcj.x()*invdep_j*invdep_j, 0, 0,
                              0, invdep_j, - Pcj.y()*invdep_j*invdep_j, 0, 0;
            reduce = SqrtInfo * reduce;
            Eigen::Matrix5d pre = TcbG*InverseSE23(Twj)*Twi;

            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 5, 9> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = - Skew(Pbi_SE23.head<3>());
                jaco.block<3, 3>(0, 6) = Eigen::Matrix3d::Identity();
                jaco = pre*jaco;
                J = reduce*jaco;
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[1]);
                Eigen::Matrix<double, 5, 9> jaco;
                jaco.setZero();
                jaco.block<3, 3>(0, 0) = Skew(Pbj_SE23.head<3>());
                jaco.block<3, 3>(0, 6) = - Eigen::Matrix3d::Identity();
                jaco = TcbG*jaco;
                J = reduce*jaco;
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[2]);
                Eigen::Vector5d jaco;
                jaco.setZero();
                jaco.head<3>() = -(pre*TbcG).block<3, 3>(0, 0)*Pci/invdep_i;
                J = reduce*jaco;
            }
        }
        return true;
    }
    Eigen::Vector2d pts_i, pts_j;
    Eigen::Matrix2d SqrtInfo;
};
