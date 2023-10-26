#include "imu_processor.h"

ImuIntegration::ImuIntegration(const IntegrationMode mode_, const Eigen::Vector6d bias_)
    :mode(mode_), b(bias_)
{
    Clear();
}

void ImuIntegration::Clear()
{
    Initialize();
    b.setZero();
    imu_data.clear();
    time_start = 0;
    time_end = 0;
}

void ImuIntegration::Initialize()
{
    dT = 0.0;
    r.setIdentity();
    Jrb.setZero();
    Cov.setZero();
    Gamma.setIdentity();
}

void ImuIntegration::SetBias(const Eigen::Vector6d bias)
{
    b = bias;
}

Eigen::Matrix5d ImuIntegration::GetUpdated_r(const Eigen::Vector6d& new_bias) const {

    Eigen::Vector6d db = new_bias - b;
    if(mode == IntegrationMode::SE23)
    {
        return Get_dr()*ExpSE23(Get_Jrb()*db);
    }
    else if(mode == IntegrationMode::SE3)
    {
        Eigen::Matrix6d Jposeb = Get_Jrb().topRows<6>();
        Eigen::Isometry3d dpose;
        dpose.linear() = Get_dR();
        dpose.translation() = Get_dV();
        Eigen::Isometry3d update_pose = dpose*ExpSE3(Jposeb*db);

        Eigen::Matrix5d update_state = Eigen::Matrix5d::Identity();
        update_state.block<3, 3>(0, 0) = update_pose.linear();
        update_state.block<3, 1>(0, 3) = update_pose.translation();
        update_state.block<3, 1>(0, 4) = Get_dP() + Get_JPg()*(db.head<3>()) + Get_JPa()*(db.tail<3>());
        return update_state;
    }
    else if(mode == IntegrationMode::SO3)
    {
        Eigen::Matrix5d update_state = Eigen::Matrix5d::Identity();
        update_state.block<3, 3>(0, 0) = Get_dR()*ExpSO3(Get_JRg()*db.head<3>());
        update_state.block<3, 1>(0, 3) = Get_dV() + Get_JVg()*(db.head<3>()) + Get_JVa()*(db.tail<3>());
        update_state.block<3, 1>(0, 4) = Get_dP() + Get_JPg()*(db.head<3>()) + Get_JPa()*(db.tail<3>());
        return update_state;
    }
    else if(mode == IntegrationMode::Quaternion)
    {
        Eigen::Matrix5d update_state = Eigen::Matrix5d::Identity();
        update_state.block<3, 3>(0, 0) = (Get_dQ()*deltaQ(Get_JRg()*db.head<3>())).toRotationMatrix();
        update_state.block<3, 1>(0, 3) = Get_dV() + Get_JVg()*(db.head<3>()) + Get_JVa()*(db.tail<3>());
        update_state.block<3, 1>(0, 4) = Get_dP() + Get_JPg()*(db.head<3>()) + Get_JPa()*(db.tail<3>());
        return update_state;
    }
}

Eigen::Quaterniond ImuIntegration::Get_dQ() const{
    Eigen::Quaterniond dQ(r.block<3, 3>(0, 0));
    return dQ;
}
Eigen::Matrix3d ImuIntegration::Get_dR() const{
    return r.block<3, 3>(0, 0);
}
Eigen::Vector3d ImuIntegration::Get_dV() const{
    return r.block<3, 1>(0, 3);
}
Eigen::Vector3d ImuIntegration::Get_dP() const{
    return r.block<3, 1>(0, 4);
}
Eigen::Matrix5d ImuIntegration::Get_dr() const{
    return r;
}
double ImuIntegration::Get_dT() const{
    return dT;
}

Eigen::Matrix3d ImuIntegration::Get_JRg() const{
    return Jrb.block<3, 3>(0, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JVg() const{
    return Jrb.block<3, 3>(3, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JVa() const{
    return Jrb.block<3, 3>(3, 3);
}
Eigen::Matrix3d ImuIntegration::Get_JPg() const{
    return Jrb.block<3, 3>(6, 0);
}
Eigen::Matrix3d ImuIntegration::Get_JPa() const{
    return Jrb.block<3, 3>(6, 3);
}
Eigen::Matrix<double, 9, 6> ImuIntegration::Get_Jrb() const{
    return Jrb;
}

void ImuIntegration::IntegrateNewMeasurement(const Eigen::Vector3d& w1, const Eigen::Vector3d& w2,
                                             const Eigen::Vector3d& a1, const Eigen::Vector3d& a2,
                                             const double dt)
{
    const Eigen::Vector3d w = 0.5*(w1 + w2) - b.head<3>();
    const Eigen::Vector3d a = 0.5*(a1 + a2) - b.tail<3>();
    const Eigen::Matrix3d Skew_w = Skew(w);
    const Eigen::Matrix3d Skew_a = Skew(a);
    const Eigen::Matrix3d Rij_1 = r.block<3, 3>(0, 0);
    const Eigen::Matrix3d Rjj_1 = ExpSO3(w*dt).transpose();

    Eigen::Matrix15d A;
    A.setIdentity();
    Eigen::Matrix<double, 15, 12> B;
    B.setZero();

    if(mode == IntegrationMode::SE23)
    {
        A.block<3, 3>(0, 0) = Rjj_1;
        A.block<3, 3>(3, 0) = - Rjj_1*Skew_a*dt;
        A.block<3, 3>(3, 3) = Rjj_1;
        A.block<3, 3>(6, 0) = - 0.5*Rjj_1*Skew_a*dt*dt;
        A.block<3, 3>(6, 3) = Rjj_1*dt;
        A.block<3, 3>(6, 6) = Rjj_1;

        B.block<3, 3>(0, 0) = RightJacobianSO3(w*dt)*dt;
        B.block<3, 3>(3, 3) = Rjj_1*dt;
        B.block<3, 3>(6, 3) = 0.5*Rjj_1*dt*dt;
        B.block<6, 6>(9, 6) = Eigen::Matrix6d::Identity()*dt;

        A.block<9, 6>(0, 9) = - B.block<9, 6>(0, 0);
    }
    else if(mode == IntegrationMode::SO3)
    {
        A.block<3, 3>(0, 0) = Rjj_1;
        A.block<3, 3>(3, 0) = -Rij_1*Skew_a*dt;
        A.block<3, 3>(6, 0) = -0.5*Rij_1*Skew_a*dt*dt;
        A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity()*dt;

        B.block<3, 3>(0, 0) = RightJacobianSO3(w*dt)*dt;
        B.block<3, 3>(3, 3) = Rij_1*dt;
        B.block<3, 3>(6, 3) = 0.5*Rij_1*dt*dt;
        B.block<6, 6>(9, 6) = Eigen::Matrix6d::Identity()*dt;

        A.block<9, 6>(0, 9) = - B.block<9, 6>(0, 0);
    }
    else if(mode == IntegrationMode::SE3)
    {
        A.block<3, 3>(0, 0) = Rjj_1;
        A.block<3, 3>(3, 0) = - Rjj_1*Skew_a*dt;
        A.block<3, 3>(3, 3) = Rjj_1;
        A.block<3, 3>(6, 0) = -0.5*Rij_1*Skew_a*dt*dt;
        A.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity()*dt;

        B.block<3, 3>(0, 0) = RightJacobianSO3(w*dt)*dt;
        B.block<3, 3>(3, 3) = Rjj_1*dt;
        B.block<3, 3>(6, 3) = 0.5*Rij_1*dt*dt;
        B.block<6, 6>(9, 6) = Eigen::Matrix6d::Identity()*dt;

        A.block<9, 6>(0, 9) = - B.block<9, 6>(0, 0);
    }
    else if(mode == IntegrationMode::Quaternion)
    {
        const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
        const Eigen::Matrix3d Rij = Rij_1*ExpSO3(w*dt);
        const Eigen::Vector3d un_a1 = a1 - b.tail<3>();
        const Eigen::Vector3d un_a2 = a2 - b.tail<3>();
        const Eigen::Matrix3d Skew_ua1 = Skew(un_a1);
        const Eigen::Matrix3d Skew_ua2 = Skew(un_a2);

        A.block<3, 3>(0, 0) = I - Skew_w*dt;
        A.block<3, 3>(3, 0) = -0.5*(Rij_1*Skew_ua1 + Rij*Skew_ua2*(I - Skew_w*dt))*dt;
        A.block<3, 3>(6, 0) = -0.25*(Rij_1*Skew_ua1 + Rij*Skew_ua2*(I - Skew_w*dt))*dt*dt;
        A.block<3, 3>(6, 3) = I*dt;

        B.block<3, 3>(0, 0) = I*dt;
        B.block<3, 3>(3, 0) = -0.5*Rij*Skew_ua2*dt*dt;
        B.block<3, 3>(3, 3) = 0.5*(Rij_1 + Rij)*dt;
        B.block<3, 3>(6, 0) = -0.25*Rij*Skew_ua2*dt*dt*dt;
        B.block<3, 3>(6, 3) = 0.25*(Rij_1 + Rij)*dt*dt;
        B.block<6, 6>(9, 6) = Eigen::Matrix6d::Identity()*dt;

        A.block<9, 6>(0, 9) = - B.block<9, 6>(0, 0);
    }

    Cov = A*Cov*A.transpose() + B*imu_sigma*B.transpose();
    Jrb = A.block<9, 9>(0, 0)*Jrb + A.block<9, 6>(0, 9);
    dT += dt;

    Eigen::Matrix5d new_r;
    new_r.setIdentity();
    new_r.block<3, 3>(0, 0) = ExpSO3(w*dt);
    new_r.block<3, 1>(0, 3) = a*dt;
    new_r.block<3, 1>(0, 4) = 0.5*a*dt*dt;
    r = FaiSE23(r, dt)*new_r;
}


bool ImuIntegration::IntegrationInterval(const io::ImuData& imu_data_, io::timestamp_t a, io::timestamp_t b, bool record)
{
    if(a >= b)
    {
        std::cerr << "Integration time error !!!";
        return false;
    }
    Eigen::Vector3d w1, w2, a1, a2;
    io::timestamp_t t1, t2;
    double dt;
    auto it = imu_data_.cbegin();
    w1 = it->w();
    a1 = it->a();
    t1 = it->timestamp;
    if(t1 > a)
    {
        dt = (t1 - a)*1e-9;
        IntegrateNewMeasurement(w1, w1, a1, a1, dt);
    }

    for(it ++; it != imu_data_.cend(); it ++)
    {
        w2 = it->w();
        a2 = it->a();
        t2 = it->timestamp;
        dt = (t2 - t1)*1e-9;

        if(t1 >= a && t2 <= b)
        {
            IntegrateNewMeasurement(w1, w2, a1, a2, dt);
        }
        else if(t1 < a && t2 > a)
        {
            double dl = (a - t1)*1e-9;
            double dr = (t2 - a)*1e-9;
            IntegrateNewMeasurement(w1*dr/dt + w2*dl/dt, w2, a1*dr/dt + a2*dl/dt, a2, dr);
        }
        else if(t1 < b && t2 > b)
        {
            double dl = (b - t1)*1e-9;
            double dr = (t2 - b)*1e-9;
            IntegrateNewMeasurement(w1, w1*dr/dt + w2*dl/dt, a1, a1*dr/dt + a2*dl/dt, dl);
        }

        w1 = w2;
        a1 = a2;
        t1 = t2;
    }
    if(record)
    {
        imu_data = imu_data_;
        time_start = a;
        time_end = b;
    }

    Info = Cov.inverse();
    SqrtInfo = SqrtMatrix(Info);
    InfoPose = Cov.block<9, 9>(0, 0).inverse();
    SqrtInfoPose = SqrtMatrix(InfoPose);
    Gamma.block<3, 1>(0, 3) = gravity_vector*dT;
    Gamma.block<3, 1>(0, 4) = 0.5*gravity_vector*dT*dT;
    return true;
}

bool ImuIntegration::Reintegrated()
{
    if(imu_data.empty())
        return false;
    Initialize();
    return IntegrationInterval(imu_data, time_start, time_end);
}

bool ImuIntegration::MergeIntegration(std::shared_ptr<ImuIntegration>& integ)
{
    if(time_end != integ->time_start && !imu_data.empty())
    {
        std::cerr << "Merge integration failed, becase of timestamp not aligned" << std::endl;
        return false;
    }

    SetBias(integ->b);
    if(imu_data.empty())
    {
        IntegrationInterval(integ->imu_data, integ->time_start, integ->time_end, true);
    }
    else
    {
        IntegrationInterval(integ->imu_data, integ->time_start, integ->time_end);
        time_end = integ->time_end;
        const io::timestamp_t end_timestamp = imu_data.back().timestamp;
        auto iter = integ->imu_data.begin();
        while(iter->timestamp <= end_timestamp)
        {
            ++iter;
        }
        imu_data.insert(imu_data.end(), iter, integ->imu_data.end());
    }
    return true;
}

ImuProcessor::ImuProcessor(const std::string &yaml_path)
{
    cv::FileStorage fSettings(yaml_path, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }
    imu_data_path = std::string(fSettings["ImuPath"]);
    imu_data = io::Read_File<io::ImuData::value_type>(imu_data_path);
    imu_data_size = imu_data.size();
    timestamp_start = imu_data[0].timestamp;
    timestamp_end = imu_data[imu_data_size - 1].timestamp;
    imu_bias.setZero();

    std::cout << " read " << imu_data.size() << " imu data !!!" << std::endl;
}

ImuProcessor::ImuProcessor(const io::ImuData &imu_data_)
{
    imu_data = imu_data_;
    imu_data_size = imu_data.size();
    timestamp_start = imu_data[0].timestamp;
    timestamp_end = imu_data[imu_data_size - 1].timestamp;
    imu_bias.setZero();
}

io::imu_data_t ImuProcessor::GetImuData(int id)
{
    if(id < 0 || id >= imu_data_size)
        std::cerr << "GetImuData error !!!" << std::endl;
    return imu_data[id];
}

io::timestamp_t ImuProcessor::GetTimestamp(int id)
{
    if(id >= 0 && id < imu_data_size)
        return imu_data[id].timestamp;
    std::cerr << "GetImuTimestamp error !!!" << std::endl;
    return -1;
}

int ImuProcessor::GetIdxBeforeEqual(io::timestamp_t t)
{
    if(t < timestamp_start || t >= timestamp_end)
    {
        std::cerr << "GetIdxBeforeEqual error !!!" << std::endl;
        return -1;
    }
    int idx = std::max(0, (int)(std::floor((t - timestamp_start)*1e-9*imu_rate) - 1));

    io::timestamp_t idx_t = GetTimestamp(idx);
    if(idx_t < 0) return -1;
    while(idx_t > t)
    {
        --idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    while(idx_t <= t)
    {
        ++idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    --idx;
    return idx;
}

int ImuProcessor::GetIdxAfterEqual(io::timestamp_t t)
{
    if(t <= timestamp_start || t > timestamp_end)
    {
        std::cerr << "GetIdxBeforeEqual error !!!" << std::endl;
        return -1;
    }
    int idx = std::max(0, (int)(std::floor((t - timestamp_start)*1e-9*imu_rate) - 1));

    io::timestamp_t idx_t = GetTimestamp(idx);
    if(idx_t < 0) return -1;
    while(idx_t > t)
    {
        --idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    while(idx_t < t)
    {
        ++idx;
        idx_t = GetTimestamp(idx);
        if(idx_t < 0) return -1;
    }
    return idx;
}

std::shared_ptr<ImuIntegration> ImuProcessor::GetImuIntegration(io::timestamp_t t1, io::timestamp_t t2, IntegrationMode mode)
{
    int start_idx = GetIdxBeforeEqual(t1);
    int end_idx = GetIdxAfterEqual(t2);
    if(start_idx == -1 || end_idx == -1)
        return nullptr;
    io::ImuData imu_data_tg;
    imu_data_tg.insert(imu_data_tg.begin(), imu_data.begin() + start_idx, imu_data.begin() + end_idx + 1);
    m_imu_bias.lock();
    std::shared_ptr<ImuIntegration> imu_integration = std::make_shared<ImuIntegration>(mode, imu_bias);
    m_imu_bias.unlock();
    imu_integration->IntegrationInterval(imu_data_tg, t1, t2, true);
    return imu_integration;
}

void ImuProcessor::SetImuBias(Eigen::Vector6d& bias)
{
    std::unique_lock<std::mutex> lock(m_imu_bias);
    imu_bias = bias;
}

