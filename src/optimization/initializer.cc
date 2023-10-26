
#include "initializer.h"

void Initializer::Push(int cur_frame_id, bool cur_is_keyframe,
                       std::map<int, Eigen::Vector2d>& cur_frame_features,
                       std::shared_ptr<ImuIntegration>& cur_integration)
{
    if(frame_size == 0 && cur_is_keyframe)
    {
        frame_ids.push_back(cur_frame_id);
        is_keyframes.push_back(cur_is_keyframe);
        feature_manager->AddFeatures(cur_frame_id, cur_frame_features);
        ++frame_size;
    }
    else if(frame_size > 0)
    {
        frame_ids.push_back(cur_frame_id);
        is_keyframes.push_back(cur_is_keyframe);
        feature_manager->AddFeatures(cur_frame_id, cur_frame_features);
        integrations.push_back(cur_integration);
        ++frame_size;
    }
}

void Initializer::Apply(std::shared_ptr<FeatureManager> &feature_manager_,
                        std::vector<std::shared_ptr<ImuIntegration>> &integrations_,
                        std::vector<Eigen::Matrix5d> &Ts_,
                        std::vector<Eigen::Vector6d> &Bs_,
                        std::vector<int> &frame_ids_,
                        bool &last_is_keyframe_)
{
    Ts_.clear();
    Bs_.clear();
    frame_ids_.clear();
    integrations_.clear();

    std::vector<int> deleted_frame_ids;
    for(int i = 0; i < frame_size - 1; ++i)
    {
        if(is_keyframes[i])
        {
            Ts_.push_back(Ts[i]);
            frame_ids_.push_back(frame_ids[i]);
            integrations_.push_back(integrations[i]);
        }
        else
        {
            deleted_frame_ids.push_back(frame_ids[i]);
            integrations_.back()->MergeIntegration(integrations[i]);
        }
    }

    Ts_.push_back(Ts[frame_size - 1]);
    frame_ids_.push_back(frame_ids[frame_size - 1]);
    last_is_keyframe_ = is_keyframes[frame_size - 1];
    Bs_.resize(Ts.size(), Bs);

    for(auto f_id : deleted_frame_ids)
    {
        feature_manager->DeleteFeatures(f_id);
    }
    feature_manager_ = feature_manager;
}


bool Initializer::Run()
{
    if(frame_size < initial_window_size)
        return false;
    else
    {
        vRwc.resize(frame_size);
        vtwc.resize(frame_size);
        if(feature_manager->StructureFromMotion(frame_ids, vRwc, vtwc)
                && InertialInit())
        {
            return true;
        }
        feature_manager->ResetSolveFlag();

        int next_keyframe_idx = -1;
        for(int i = 1; i < frame_size; ++i)
        {
            if(is_keyframes[i])
            {
                next_keyframe_idx = i;
                break;
            }
        }
        if(next_keyframe_idx < 0)
        {
            frame_ids.clear();
            is_keyframes.clear();
            integrations.clear();
            feature_manager->Clear();
            frame_size = 0;
        }
        else
        {
            int remain_size = frame_size - next_keyframe_idx;
            for(int i = 0; i < next_keyframe_idx; ++i)
            {
                feature_manager->DeleteFeatures(frame_ids[i]);;
            }

            for(int i = 0; i < remain_size; ++i)
            {
                frame_ids[i] = frame_ids[next_keyframe_idx + i];
                is_keyframes[i] = is_keyframes[next_keyframe_idx + i];
            }
            for(int i = 0; i < remain_size - 1; ++i)
            {
                integrations[i] = integrations[next_keyframe_idx + i];
            }

            frame_ids.resize(remain_size);
            is_keyframes.resize(remain_size);
            integrations.resize(remain_size - 1);
            frame_size = remain_size;
        }
        return false;
    }
}

bool Initializer::InertialInit()
{
    SolveGyroscopeBias();
    Bs.tail<3>().setZero();

    for(int i = 0; i < frame_size - 1; ++i)
    {
        integrations[i]->SetBias(Bs);
        integrations[i]->Reintegrated();
    }

    if(!LinearAlignment())
    {
        std::cout << "LinearAlignment failed !" << std::endl;
        Bs.setZero();
        for(int i = 0; i < frame_size - 1; ++i)
        {
            integrations[i]->SetBias(Bs);
            integrations[i]->Reintegrated();
        }
        return false;
    }
    Ts.resize(frame_size, Eigen::Matrix5d::Identity());
    for(int i = 0; i < frame_size; ++i)
    {
        Ts[i].block<3, 3>(0, 0) = vRwc[i]*RcbG;
        Ts[i].block<3, 1>(0, 3) = Vs[i];
        Ts[i].block<3, 1>(0, 4) = vRwc[i]*tcbG + s*vtwc[i];
    }
    Eigen::Matrix5d Ts0;
    Ts0 = InverseSE23(Ts[0]);
    Ts0.block(0, 3, 3, 1).setZero();
    for(int i = 0; i < frame_size; ++i)
    {
        Ts[i] = Ts0*Ts[i];
    }

    g = Ts0.block<3, 3>(0, 0)*g;
    Eigen::Matrix3d Rgw = R_v2v(g, gravity_vector);
    double yaw = R2ypr(Rgw).x();
    Rgw = ypr2R(Eigen::Vector3d{-yaw, 0, 0})*Rgw;

    g = Rgw*g;
    std::cout << g.transpose() << std::endl;
    Eigen::Matrix5d Tgw;
    Tgw.setIdentity();
    Tgw.block<3, 3>(0, 0) = Rgw;
    for(int i = 0; i < frame_size; ++i)
    {
        Ts[i] = Tgw*Ts[i];
    }

    return true;
}

void Initializer::SolveGyroscopeBias()
{
    double* bg_ptr = new double[3];
    Eigen::Map<Eigen::Vector3d> bg(bg_ptr);
    bg.setZero();
    ceres::Problem problem;
    for(int i = 0; i < frame_size - 1; ++i)
    {
        Eigen::Matrix3d Rwbi = vRwc[i]*RcbG;
        Eigen::Matrix3d Rwbj = vRwc[i + 1]*RcbG;
        ceres::CostFunction* cost_function = new GyroscopeBiasCostFunction(integrations[i], Rwbi, Rwbj);
        problem.AddResidualBlock(cost_function, nullptr, bg_ptr);
    }
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    Bs.head<3>() = bg;
    delete[] bg_ptr;
}

bool Initializer::LinearAlignment()
{
    const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
    int n_state = frame_size*3 + 3 + 1;
    Eigen::MatrixXd A{n_state, n_state};
    A.setZero();
    Eigen::VectorXd b{n_state};
    b.setZero();

    for(int i = 0; i < frame_size - 1; ++i)
    {
        Eigen::MatrixXd tmp_A(6, 10);
        tmp_A.setZero();
        Eigen::VectorXd tmp_b(6);
        tmp_b.setZero();

        const double dt = integrations[i]->dT;

        tmp_A.block<3, 3>(0, 0) = dt*I;
        tmp_A.block<3, 3>(0, 6) = 0.5*dt*dt*I;
        tmp_A.block<3, 1>(0, 9) = (vtwc[i] - vtwc[i + 1])/100.0;
        tmp_b.block<3, 1>(0, 0) = (vRwc[i + 1] - vRwc[i])*tcbG - vRwc[i]*RcbG*integrations[i]->r.block<3, 1>(0, 4);

        tmp_A.block<3, 3>(3, 0) = I;
        tmp_A.block<3, 3>(3, 3) = -I;
        tmp_A.block<3, 3>(3, 6) = dt*I;
        tmp_b.block<3, 1>(3, 0) = -vRwc[i]*RcbG*integrations[i]->r.block<3, 1>(0, 3);

        Eigen::MatrixXd r_A = tmp_A.transpose()*tmp_A;
        Eigen::VectorXd r_b = tmp_A.transpose()*tmp_b;

        A.block<6, 6>(i*3, i*3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i*3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i*3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i*3) += r_A.bottomLeftCorner<4, 6>();
    }

    A = A * 1000.0;
    b = b * 1000.0;
    Eigen::VectorXd x{n_state};
    x = A.ldlt().solve(b);
    s = x(n_state - 1)/100.0;
    g = x.segment<3>(n_state - 4);
    if(fabs(g.norm() - gravity_magnitude) > 1.0 || s < 0)
    {
        return false;
    }

    Eigen::Matrix3d Rwg = R_v2v(gravity_vector, g);
    g = Rwg*gravity_vector;
    Eigen::VectorXd x_{n_state - 1};

    for(int it = 0; it < 4; ++it)
    {
        Eigen::MatrixXd A_{n_state - 1, n_state - 1};
        A_.setZero();
        Eigen::VectorXd b_{n_state - 1};
        b_.setZero();
        for(int i = 0; i < frame_size - 1; ++i)
        {
            Eigen::MatrixXd tmp_A(6, 9);
            tmp_A.setZero();
            Eigen::VectorXd tmp_b(6);
            tmp_b.setZero();

            const double dt = integrations[i]->dT;

            tmp_A.block<3, 3>(0, 0) = dt*I;
            tmp_A.block<3, 2>(0, 6) = -0.5*dt*dt*(Rwg*Skew(gravity_vector)).leftCols<2>();
            tmp_A.block<3, 1>(0, 8) = (vtwc[i] - vtwc[i + 1])/100.0;
            tmp_b.block<3, 1>(0, 0) = (vRwc[i + 1] - vRwc[i])*tcbG - vRwc[i]*RcbG*integrations[i]->r.block<3, 1>(0, 4) - 0.5*dt*dt*g;

            tmp_A.block<3, 3>(3, 0) = I;
            tmp_A.block<3, 3>(3, 3) = -I;
            tmp_A.block<3, 2>(3, 6) = -dt*(Rwg*Skew(gravity_vector)).leftCols<2>();
            tmp_b.block<3, 1>(3, 0) = -vRwc[i]*RcbG*integrations[i]->r.block<3, 1>(0, 3) - dt*g;

            Eigen::MatrixXd r_A = tmp_A.transpose()*tmp_A;
            Eigen::VectorXd r_b = tmp_A.transpose()*tmp_b;

            A_.block<6, 6>(i*3, i*3) += r_A.topLeftCorner<6, 6>();
            b_.segment<6>(i*3) += r_b.head<6>();

            A_.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b_.tail<3>() += r_b.tail<3>();

            A_.block<6, 3>(i*3, n_state - 4) += r_A.topRightCorner<6, 3>();
            A_.block<3, 6>(n_state - 4, i*3) += r_A.bottomLeftCorner<3, 6>();
        }
        A_ = A_ * 1000.0;
        b_ = b_ * 1000.0;
        x_ = A_.ldlt().solve(b_);
        Rwg = Rwg*ExpSO3(x_(n_state - 4), x_(n_state - 3), 0);
        g = Rwg*gravity_vector;
    }
    s = x_(n_state - 2)/100.0;
    Vs.resize(frame_size);
    for(int i = 0; i < frame_size; ++i)
    {
        Vs[i] = x_.segment<3>(3*i);
    }
    return true;
}
