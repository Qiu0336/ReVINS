
#include "lie_group.h"



Eigen::Matrix3d Skew(const Eigen::Vector3d &w)
{
    Eigen::Matrix3d W;
    W << 0.0, -w.z(), w.y(), w.z(), 0.0, -w.x(), -w.y(),  w.x(), 0.0;
    return W;
}

Eigen::Matrix3d ExpSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + W + 0.5*W*W;
    else
        return Eigen::Matrix3d::Identity() + W*sin(theta)/theta + W*W*(1.0-cos(theta))/theta2;
}

Eigen::Vector3d LogSO3(const Eigen::Matrix3d &R)
{
    double costheta = 0.5*(R.trace()-1.0);
    if(costheta > +1.0) costheta = +1.0;
    if(costheta < -1.0) costheta = -1.0;
    const double theta = acos(costheta);
    const Eigen::Vector3d w(R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1));
    if(theta < 1e-4)
        return 0.5*w;
    else
        return 0.5*theta*w/sin(theta);
}

Eigen::Matrix3d LeftJacobianSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() + 0.5*W + W*W/6.0;
    else
        return Eigen::Matrix3d::Identity() + W*(1.0-cos(theta))/theta2 + W*W*(theta-sin(theta))/(theta2*theta);
}

Eigen::Matrix3d InverseLeftJacobianSO3(const Eigen::Vector3d &w)
{
    const double theta = w.norm();
    const double theta2 = theta*theta;
    Eigen::Matrix3d W = Skew(w);
    if(theta < 1e-4)
        return Eigen::Matrix3d::Identity() - 0.5*W + W*W/12.0;
    else
        return Eigen::Matrix3d::Identity() - 0.5*W + W*W*(1.0/theta2 - (1.0+cos(theta))/(2.0*theta*sin(theta)));
}

Eigen::Matrix6d AdjointSE3(const Eigen::Isometry3d& Ts)
{
    Eigen::Matrix3d R = Ts.rotation();
    Eigen::Vector3d t = Ts.translation();
    Eigen::Matrix6d res;
    res.setZero();
    res.block<3, 3>(0, 0) = R;
    res.block<3, 3>(3, 3) = R;
    res.block<3, 3>(3, 0) = Skew(t)*R;
    return res;
}

Eigen::Isometry3d ExpSE3(const Eigen::Vector6d &t)
{
    Eigen::Vector3d fai = t.head<3>();
    Eigen::Isometry3d Ts;
    Ts.setIdentity();
    Ts.linear() = ExpSO3(fai);
    Ts.translation() = LeftJacobianSO3(fai)*t.tail<3>();
    return Ts;
}

Eigen::Vector6d LogSE3(const Eigen::Isometry3d &Ts)
{
    Eigen::Vector6d t;
    Eigen::Matrix3d R = Ts.rotation();
    t.head<3>() = LogSO3(R);
    Eigen::Matrix3d InvJlogR = InverseLeftJacobianSO3(t.head<3>());
    t.tail<3>() = InvJlogR*Ts.translation();
    return t;
}

Eigen::Matrix6d LeftJacobianSE3(const Eigen::Vector6d &t)
{
    const Eigen::Vector3d r = t.head<3>();
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 24.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (0.5*fai2 + cos(fai) - 1)/(fai2*fai2);
        a3 = (fai - 1.5*sin(fai) + 0.5*fai*cos(fai))/(fai2*fai2*fai);
    }

    const Eigen::Matrix3d Jr = LeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(t.tail<3>());
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix6d res;
    res.setZero();
    res.block<3, 3>(0, 0) = Jr;
    res.block<3, 3>(3, 3) = Jr;
    res.block<3, 3>(3, 0) = Qrp;
    return res;
}

Eigen::Matrix6d InverseLeftJacobianSE3(const Eigen::Vector6d &t)
{
    const Eigen::Vector3d r = t.head<3>();
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 24.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (0.5*fai2 + cos(fai) - 1)/(fai2*fai2);
        a3 = (fai - 1.5*sin(fai) + 0.5*fai*cos(fai))/(fai2*fai2*fai);
    }

    const Eigen::Matrix3d InvJr = InverseLeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(t.tail(3));
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix6d res;
    res.setZero();
    res.block<3, 3>(0, 0) = InvJr;
    res.block<3, 3>(3, 3) = InvJr;
    res.block<3, 3>(3, 0) = -InvJr*Qrp*InvJr;
    return res;
}



Eigen::Matrix5d FaiSE23(const Eigen::Matrix5d& tmp_r, const double t)
{
    Eigen::Matrix5d res = tmp_r;
    res.block<3, 1>(0, 4) = tmp_r.block<3, 1>(0, 4) + tmp_r.block<3, 1>(0, 3)*t;
    return res;
}

Eigen::Matrix9d F_Mat_SE23(const double t)
{
    Eigen::Matrix9d F;
    F.setIdentity();
    F.block<3, 3>(6, 3) = t*Eigen::Matrix3d::Identity();
    return F;
}

Eigen::Matrix5d InverseSE23(const Eigen::Matrix5d& tmp_r)
{
    Eigen::Matrix5d res = tmp_r;
    Eigen::Matrix3d inv_R = tmp_r.block<3, 3>(0, 0).transpose();
    res.block<3, 3>(0, 0) = inv_R;
    res.block<3, 1>(0, 3) = -inv_R*tmp_r.block<3, 1>(0, 3);
    res.block<3, 1>(0, 4) = -inv_R*tmp_r.block<3, 1>(0, 4);
    return res;
}
Eigen::Matrix9d AdjointSE23(const Eigen::Matrix5d& r)
{
    Eigen::Matrix3d R = r.block<3, 3>(0, 0);
    Eigen::Vector3d v = r.block<3, 1>(0, 3);
    Eigen::Vector3d p = r.block<3, 1>(0, 4);
    Eigen::Matrix9d res;
    res.setZero();
    res.block<3, 3>(0, 0) = R;
    res.block<3, 3>(3, 3) = R;
    res.block<3, 3>(6, 6) = R;
    res.block<3, 3>(3, 0) = Skew(v)*R;
    res.block<3, 3>(6, 0) = Skew(p)*R;
    return res;
}
Eigen::Matrix5d ExpSE23(const Eigen::Vector9d& s)
{
    Eigen::Vector3d fai = s.head<3>();
    Eigen::Matrix5d r;
    r.setIdentity();
    r.block<3, 3>(0, 0) = ExpSO3(fai);
    r.block<3, 1>(0, 3) = LeftJacobianSO3(fai)*s.segment<3>(3);
    r.block<3, 1>(0, 4) = LeftJacobianSO3(fai)*s.tail<3>();
    return r;
}

Eigen::Vector9d LogSE23(const Eigen::Matrix5d& r)
{
    Eigen::Vector9d s;
    Eigen::Matrix3d R = r.block<3, 3>(0, 0);
    s.head<3>() = LogSO3(R);
    Eigen::Matrix3d InvJlogR = InverseLeftJacobianSO3(s.head<3>());
    s.segment<3>(3) = InvJlogR*r.block<3, 1>(0, 3);
    s.tail<3>() = InvJlogR*r.block<3, 1>(0, 4);
    return s;
}

Eigen::Matrix9d LeftJacobianSE23(const Eigen::Vector9d& s)
{
    const Eigen::Vector3d r = s.head<3>();
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 24.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (0.5*fai2 + cos(fai) - 1)/(fai2*fai2);
        a3 = (fai - 1.5*sin(fai) + 0.5*fai*cos(fai))/(fai2*fai2*fai);
    }

    const Eigen::Matrix3d Jr = LeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_v = Skew(s.segment<3>(3));
    const Eigen::Matrix3d Skew_rvr = Skew_r*Skew_v*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(s.tail<3>());
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrv = 0.5*Skew_v + a1*(Skew_r*Skew_v + Skew_v*Skew_r + Skew_rvr)
            + a2*(Skew_r2*Skew_v + Skew_v*Skew_r2 - 3*Skew_rvr)
            + a3*(Skew_rvr*Skew_r + Skew_r*Skew_rvr);
    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix9d res;
    res.setZero();
    res.block<3, 3>(0, 0) = Jr;
    res.block<3, 3>(3, 3) = Jr;
    res.block<3, 3>(6, 6) = Jr;
    res.block<3, 3>(3, 0) = Qrv;
    res.block<3, 3>(6, 0) = Qrp;
    return res;
}

Eigen::Matrix9d InverseLeftJacobianSE23(const Eigen::Vector9d& s)
{
    const Eigen::Vector3d r = s.head<3>();
    const double fai = r.norm();
    double a1, a2, a3;
    if(fai < 1e-4)
    {
        a1 = 1.0 / 6.0;
        a2 = 1.0 / 24.0;
        a3 = 1.0 / 120.0;
    }
    else
    {
        const double fai2 = fai*fai;
        a1 = (fai - sin(fai))/(fai2*fai);
        a2 = (0.5*fai2 + cos(fai) - 1)/(fai2*fai2);
        a3 = (fai - 1.5*sin(fai) + 0.5*fai*cos(fai))/(fai2*fai2*fai);
    }

    const Eigen::Matrix3d InvJr = InverseLeftJacobianSO3(r);
    const Eigen::Matrix3d Skew_r = Skew(r);
    const Eigen::Matrix3d Skew_r2 = Skew_r*Skew_r;
    const Eigen::Matrix3d Skew_v = Skew(s.segment<3>(3));
    const Eigen::Matrix3d Skew_rvr = Skew_r*Skew_v*Skew_r;
    const Eigen::Matrix3d Skew_p = Skew(s.tail<3>());
    const Eigen::Matrix3d Skew_rpr = Skew_r*Skew_p*Skew_r;

    const Eigen::Matrix3d Qrv = 0.5*Skew_v + a1*(Skew_r*Skew_v + Skew_v*Skew_r + Skew_rvr)
            + a2*(Skew_r2*Skew_v + Skew_v*Skew_r2 - 3*Skew_rvr)
            + a3*(Skew_rvr*Skew_r + Skew_r*Skew_rvr);
    const Eigen::Matrix3d Qrp = 0.5*Skew_p + a1*(Skew_r*Skew_p + Skew_p*Skew_r + Skew_rpr)
            + a2*(Skew_r2*Skew_p + Skew_p*Skew_r2 - 3*Skew_rpr)
            + a3*(Skew_rpr*Skew_r + Skew_r*Skew_rpr);
    Eigen::Matrix9d res;
    res.setZero();
    res.block<3, 3>(0, 0) = InvJr;
    res.block<3, 3>(3, 3) = InvJr;
    res.block<3, 3>(6, 6) = InvJr;
    res.block<3, 3>(3, 0) = -InvJr*Qrv*InvJr;
    res.block<3, 3>(6, 0) = -InvJr*Qrp*InvJr;
    return res;
}


// v2.normalized = R*v1.normalized;
Eigen::Matrix3d R_v2v(const Eigen::Vector3d &v1, const Eigen::Vector3d &v2)
{
    const Eigen::Vector3d v1n = v1.normalized();
    const Eigen::Vector3d v2n = v2.normalized();
    const Eigen::Vector3d axis = (v1n.cross(v2n)).normalized();
    const double theta = std::acos(v1n.dot(v2n));
    return ( ExpSO3(axis*theta) );
}

Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr / M_PI * 180.0;
}

Eigen::Matrix3d ypr2R(const Eigen::Vector3d &ypr)
{

    double y = ypr(0) / 180.0 * M_PI;
    double p = ypr(1) / 180.0 * M_PI;
    double r = ypr(2) / 180.0 * M_PI;

    Eigen::Matrix3d Rz;
    Rz << cos(y), -sin(y), 0,
        sin(y), cos(y), 0,
        0, 0, 1;

    Eigen::Matrix3d Ry;
    Ry << cos(p), 0., sin(p),
        0., 1., 0.,
        -sin(p), 0., cos(p);

    Eigen::Matrix3d Rx;
    Rx << 1., 0., 0.,
        0., cos(r), -sin(r),
        0., sin(r), cos(r);

    return Rz * Ry * Rx;
}

Eigen::Quaterniond deltaQ(const Eigen::Vector3d &theta)
{
    Eigen::Quaterniond dq;
    Eigen::Vector3d half_theta = 0.5*theta;
    dq.w() = 1.0;
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

Eigen::Matrix4d Qleft(const Eigen::Quaterniond &q)
{
    Eigen::Matrix4d ans;
    ans(0, 0) = q.w();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<1, 3>(0, 1) = q.vec().transpose();
    ans.block<3, 3>(1, 1) = q.w()*Eigen::Matrix3d::Identity() + Skew(q.vec());
    return ans;
}

Eigen::Matrix4d Qright(const Eigen::Quaterniond &q)
{
    Eigen::Matrix4d ans;
    ans(0, 0) = q.w();
    ans.block<3, 1>(1, 0) = q.vec();
    ans.block<1, 3>(0, 1) = q.vec().transpose();
    ans.block<3, 3>(1, 1) = q.w() * Eigen::Matrix3d::Identity() - Skew(q.vec());
    return ans;
}

Eigen::MatrixXd SqrtMatrix(const Eigen::MatrixXd& m)
{
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5f*(m + m.transpose()));
    Eigen::VectorXd ZeroXd = Eigen::VectorXd::Zero(m.rows(), 1);
    return solver.eigenvectors()*solver.eigenvalues().cwiseMax(ZeroXd).cwiseSqrt().asDiagonal()*solver.eigenvectors().transpose();
}

Eigen::Matrix3d RotationCorrect(const Eigen::Matrix3d R)
{
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3d U = svd.matrixU(), V = svd.matrixV();
    return U*V.transpose();
}
