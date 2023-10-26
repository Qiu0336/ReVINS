#include "estimator.h"

double imu_max_gap_time = 3.0;

void Estimator::PushBuf(int frame_id, bool is_keyframe, std::map<int, Eigen::Vector2d>& frame_features,
                        std::shared_ptr<ImuIntegration>& imu_integration)
{
    std::unique_lock<std::mutex> lock(m_buf);
    frame_id_buf.push(frame_id);
    is_keyframe_buf.push(is_keyframe);
    frame_features_buf.push(frame_features);
    imu_integration_buf.push(imu_integration);
}

void Estimator::Run()
{
    while(!GetQuit() || !frame_id_buf.empty())
    {
        std::chrono::milliseconds dura(1);
        std::this_thread::sleep_for(dura);

        bool getting_data = false;
        int cur_frame_id;
        bool cur_is_keyframe;
        std::map<int, Eigen::Vector2d> cur_frame_features;
        std::shared_ptr<ImuIntegration> cur_imu_integration;
        m_buf.lock();
        if(!frame_id_buf.empty())
        {
            cur_frame_id = frame_id_buf.front();
            cur_is_keyframe = is_keyframe_buf.front();
            cur_frame_features = frame_features_buf.front();
            cur_imu_integration = imu_integration_buf.front();

            frame_id_buf.pop();
            is_keyframe_buf.pop();
            frame_features_buf.pop();
            imu_integration_buf.pop();
            getting_data = true;
        }
        m_buf.unlock();

        if(getting_data == true)
        {
            m_busy.lock();
            if(!is_initialized)
            {
                initializer->Push(cur_frame_id, cur_is_keyframe, cur_frame_features, cur_imu_integration);
                if(initializer->Run())
                {
                    is_initialized = true;
                    std::cout << "initialization success !!!" << std::endl;
                    initializer->Apply(feature_manager, integrations, Ts, Bs, frame_ids, last_is_keyframe);

                    imu_processor_ptr->SetImuBias(Bs.back());
                    m_buf.lock();
                    reintegrated_num = imu_integration_buf.size() + 1;
                    m_buf.unlock();

                    UpdateCameraCSbyTs();
                    feature_manager->ResetSolveFlag();
                    feature_manager->Triangulation(vRcw, vtcw, frame_ids);

                    if(frame_ids.size() > sliding_window_size)
                    {
                        const int redundant = frame_ids.size() - sliding_window_size;
                        for(int i = 0; i < redundant; ++i)
                        {
                            feature_manager->DeleteFeatures(frame_ids[i], vRcw, vtcw, frame_ids);
                        }
                        for(size_t i = redundant; i < frame_ids.size(); ++i)
                        {
                            frame_ids[i - redundant] = frame_ids[i];
                            Ts[i - redundant] = Ts[i];
                            Bs[i - redundant] = Bs[i];
                        }
                        frame_ids.resize(sliding_window_size);
                        Ts.resize(sliding_window_size);
                        Bs.resize(sliding_window_size);
                        UpdateCameraCSbyTs();
                        for(size_t i = redundant; i < integrations.size(); ++i)
                        {
                            integrations[i - redundant] = integrations[i];
                        }
                        integrations.resize(integrations.size() - redundant);
                    }

                    frame_size = frame_ids.size();

                    Optimization();
                }
            }

            else
            {
                if(reintegrated_num > 0)
                {
                    cur_imu_integration->SetBias(Bs.back());
                    cur_imu_integration->Reintegrated();
                    --reintegrated_num;
                }

                double dt = cur_imu_integration->dT;
                const Eigen::Matrix5d cur_Twi = cur_imu_integration->Gamma*FaiSE23(Ts.back(), dt)*cur_imu_integration->r;

                feature_manager->AddFeatures(cur_frame_id, cur_frame_features);

                if(frame_size < sliding_window_size + 1)
                {
                    frame_ids.push_back(cur_frame_id);
                    Ts.push_back(cur_Twi);
                    Eigen::Vector6d bs = Bs.back();
                    Bs.push_back(bs);
                    integrations.push_back(cur_imu_integration);
                    ++frame_size;
                }
                else
                {
                    frame_ids.back() = cur_frame_id;
                    Ts.back() = cur_Twi;
                    integrations.back() = cur_imu_integration;
                }

                UpdateCameraCSbyTs();
                feature_manager->Triangulation(vRcw, vtcw, frame_ids);
                Optimization();
                feature_manager->ResetOutliers();

                imu_processor_ptr->SetImuBias(Bs.back());
                const Eigen::Matrix5d Twc = Ts.back()*TbcG;
                m_camera_pose.lock();
                camera_pose.linear() = Twc.block<3, 3>(0, 0);
                camera_pose.translation() = Twc.block<3, 1>(0, 4);
                m_camera_pose.unlock();

                UpdateCameraCSbyTs();
                if(last_is_keyframe)
                {
                    PubKeyframe();
                    if(frame_size == sliding_window_size + 1)
                    {
                        feature_manager->DeleteFeatures(frame_ids[0], vRcw, vtcw, frame_ids);
                        for(int i = 1; i < frame_size; ++i)
                        {
                            frame_ids[i - 1] = frame_ids[i];
                            Ts[i - 1] = Ts[i];
                            Bs[i - 1] = Bs[i];
                        }
                        for(int i = 1; i < frame_size - 1; ++i)
                        {
                            std::swap(integrations[i - 1], integrations[i]);
                        }
                    }
                }
                else
                {
                    feature_manager->DeleteFeatures(frame_ids[frame_size - 2], vRcw, vtcw, frame_ids);
                    frame_ids[frame_size - 2] = frame_ids[frame_size - 1];
                    Ts[frame_size - 2] = Ts[frame_size - 1];
                    Bs[frame_size - 2] = Bs[frame_size - 1];
                    integrations[frame_size - 3]->MergeIntegration(integrations.back());
                    if(frame_size < sliding_window_size + 1)
                    {
                        --frame_size;
                        frame_ids.resize(frame_size);
                        Ts.resize(frame_size);
                        Bs.resize(frame_size);
                        integrations.resize(frame_size - 1);
                    }
                }
            }
            last_is_keyframe = cur_is_keyframe;
            m_busy.unlock();
        }
    }
}

void Estimator::Optimization()
{
    ceres::Problem problem;

    ceres::LocalParameterization *quaternion_local_parameterization = new QuaternionParameterization();
    ceres::LocalParameterization *se23_local_parameterization = new SE23Parameterization();
    ceres::LocalParameterization *rotation_local_parameterization = new RotationMatrixParameterization();

    Mat2Para();
    if(last_marginalization_info)
    {
        MarginalizationFactor* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, nullptr, last_marginalization_parameter_blocks);
    }

    for(int i = 0; i < frame_size - 1; ++i)
    {
        if(integrations[i]->dT > imu_max_gap_time)
            continue;
        if(integration_mode == IntegrationMode::SE23)
        {
            IMUFactorSE23* imu_factor = new IMUFactorSE23(integrations[i]);
            problem.AddResidualBlock(imu_factor, nullptr,
                                     para_T[i], para_T[i + 1], para_B[i], para_B[i + 1]);
        }
        else if(integration_mode == IntegrationMode::SO3)
        {
            IMUFactorSO3* imu_factor = new IMUFactorSO3(integrations[i]);
            problem.AddResidualBlock(imu_factor, nullptr,
                                     para_R[i], para_V[i], para_P[i],
                                     para_R[i + 1], para_V[i + 1], para_P[i + 1],
                                     para_bg[i], para_ba[i], para_bg[i + 1], para_ba[i + 1]);
        }
        else if(integration_mode == IntegrationMode::Quaternion)
        {
            IMUFactorQuaternion* imu_factor = new IMUFactorQuaternion(integrations[i]);
            problem.AddResidualBlock(imu_factor, nullptr,
                                     para_Q[i], para_V[i], para_P[i],
                                     para_Q[i + 1], para_V[i + 1], para_P[i + 1],
                                     para_bg[i], para_ba[i], para_bg[i + 1], para_ba[i + 1]);
        }
    }

    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);


    int feature_idx = -1;
    for(auto &f_id : feature_manager->solved_feature_ids)
    {
        ++feature_idx;
        auto& cur_feature = feature_manager->all_features[f_id];
        Eigen::Vector2d pt_i = cur_feature->reference_point.tail<2>();
        int i = std::distance(frame_ids.begin(), std::find(frame_ids.begin(), frame_ids.end(), cur_feature->reference_frame_id));

        for(int j = i + 1; j < frame_size; ++j)
        {
            if(cur_feature->IsObservedInFrame(frame_ids[j]))
            {
                Eigen::Vector2d pts_j = cur_feature->tracked_points[frame_ids[j]].tail<2>();
                if(integration_mode == IntegrationMode::SE23)
                {
                    ProjectionFactorSE23 *f = new ProjectionFactorSE23(pt_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_T[i], para_T[j], para_invdep[feature_idx]);
                }
                else if(integration_mode == IntegrationMode::SO3)
                {
                    ProjectionFactorSO3 *f = new ProjectionFactorSO3(pt_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_R[i], para_P[i], para_R[j], para_P[j], para_invdep[feature_idx]);
                }
                else if(integration_mode == IntegrationMode::Quaternion)
                {
                    ProjectionFactorQuaternion *f = new ProjectionFactorQuaternion(pt_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Q[i], para_P[i], para_Q[j], para_P[j], para_invdep[feature_idx]);
                }
            }
        }
    }

    for(int i = 0; i < frame_size; ++i)// SE23本地参数化
    {
        if(integration_mode == IntegrationMode::SE23)
            problem.SetParameterization(para_T[i], se23_local_parameterization);
        else if(integration_mode == IntegrationMode::SO3)
            problem.SetParameterization(para_R[i], rotation_local_parameterization);
        else if(integration_mode == IntegrationMode::Quaternion)
            problem.SetParameterization(para_Q[i], quaternion_local_parameterization);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.num_threads = 2;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 8;
    options.max_solver_time_in_seconds = 0.05;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    Para2Mat();
    if(frame_size <= sliding_window_size)
        return;
    if(last_is_keyframe)
    {
        MarginalizationInfo* marginalization_info = new MarginalizationInfo();
        Mat2Para();

        if(last_marginalization_info)
        {
            std::vector<int> drop_set;
            for(int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); ++i)
            {
                if(integration_mode == IntegrationMode::SE23)
                {
                    if(last_marginalization_parameter_blocks[i] == para_T[0] ||
                       last_marginalization_parameter_blocks[i] == para_B[0])
                        drop_set.push_back(i);
                    if(drop_set.size() == 2)
                        break;
                }
                else if(integration_mode == IntegrationMode::SO3)
                {
                    if(last_marginalization_parameter_blocks[i] == para_R[0] ||
                       last_marginalization_parameter_blocks[i] == para_V[0] ||
                       last_marginalization_parameter_blocks[i] == para_P[0] ||
                       last_marginalization_parameter_blocks[i] == para_bg[0] ||
                       last_marginalization_parameter_blocks[i] == para_ba[0])
                        drop_set.push_back(i);
                    if(drop_set.size() == 5)
                        break;
                }
                else if(integration_mode == IntegrationMode::Quaternion)
                {
                    if(last_marginalization_parameter_blocks[i] == para_Q[0] ||
                       last_marginalization_parameter_blocks[i] == para_V[0] ||
                       last_marginalization_parameter_blocks[i] == para_P[0] ||
                       last_marginalization_parameter_blocks[i] == para_bg[0] ||
                       last_marginalization_parameter_blocks[i] == para_ba[0])
                        drop_set.push_back(i);
                    if(drop_set.size() == 5)
                        break;
                }
            }

            MarginalizationFactor* marginalization_factor = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                                           last_marginalization_parameter_blocks,
                                                                           drop_set);
            marginalization_info->AddResidualBlockInfo(residual_block_info);
        }

        if(integrations[0]->dT < imu_max_gap_time)
        {
            if(integration_mode == IntegrationMode::SE23)
            {
                IMUFactorSE23* imu_factor = new IMUFactorSE23(integrations[0]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr, std::vector<double*>{para_T[0], para_T[1],
                                                                               para_B[0], para_B[1]}, std::vector<int>{0, 2});
                marginalization_info->AddResidualBlockInfo(residual_block_info);
            }
            else if(integration_mode == IntegrationMode::SO3)
            {
                IMUFactorSO3* imu_factor = new IMUFactorSO3(integrations[0]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr, std::vector<double*>{para_R[0], para_V[0], para_P[0],
                                                                               para_R[1], para_V[1], para_P[1], para_bg[0], para_ba[0], para_bg[1], para_ba[1]},
                                                                               std::vector<int>{0, 1, 2, 6, 7});
                marginalization_info->AddResidualBlockInfo(residual_block_info);
            }
            else if(integration_mode == IntegrationMode::Quaternion)
            {
                IMUFactorQuaternion* imu_factor = new IMUFactorQuaternion(integrations[0]);
                ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(imu_factor, nullptr, std::vector<double*>{para_Q[0], para_V[0], para_P[0],
                                                                               para_Q[1], para_V[1], para_P[1], para_bg[0], para_ba[0], para_bg[1], para_ba[1]},
                                                                               std::vector<int>{0, 1, 2, 6, 7});
                marginalization_info->AddResidualBlockInfo(residual_block_info);
            }
        }


        int feature_idx = -1;
        for(auto &f_id : feature_manager->solved_feature_ids)
        {
            ++feature_idx;
            auto& cur_feature = feature_manager->all_features[f_id];
            if(cur_feature->reference_frame_id != frame_ids[0]) continue;
            Eigen::Vector2d pt_0 = cur_feature->reference_point.tail<2>();
            for(int j = 1; j < frame_size; ++j)
            {
                if(cur_feature->IsObservedInFrame(frame_ids[j]))
                {
                    Eigen::Vector2d pt_j = cur_feature->tracked_points[frame_ids[j]].tail<2>();

                    if(integration_mode == IntegrationMode::SE23)
                    {
                        ProjectionFactorSE23 *f = new ProjectionFactorSE23(pt_0, pt_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function, std::vector<double*>{para_T[0], para_T[j], para_invdep[feature_idx]},
                                                                                       std::vector<int>{0, 2});
                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                    else if(integration_mode == IntegrationMode::SO3)
                    {
                        ProjectionFactorSO3 *f = new ProjectionFactorSO3(pt_0, pt_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function, std::vector<double*>{para_R[0], para_P[0], para_R[j], para_P[j], para_invdep[feature_idx]},
                                                                                       std::vector<int>{0, 1, 4});
                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                    else if(integration_mode == IntegrationMode::Quaternion)
                    {
                        ProjectionFactorQuaternion *f = new ProjectionFactorQuaternion(pt_0, pt_j);
                        ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(f, loss_function, std::vector<double*>{para_Q[0], para_P[0], para_Q[j], para_P[j], para_invdep[feature_idx]},
                                                                                       std::vector<int>{0, 1, 4});
                        marginalization_info->AddResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        marginalization_info->PreMarginalize();

        marginalization_info->Marginalize();

        std::unordered_map<long, double*> addr_shift;
        if(integration_mode == IntegrationMode::SE23)
        {
            for(int i = 1; i < frame_size; ++i)
            {
                addr_shift[reinterpret_cast<long>(para_T[i])] = para_T[i - 1];
                addr_shift[reinterpret_cast<long>(para_B[i])] = para_B[i - 1];
            }
        }
        else if(integration_mode == IntegrationMode::SO3)
        {
            for(int i = 1; i < frame_size; ++i)
            {
                addr_shift[reinterpret_cast<long>(para_R[i])] = para_R[i - 1];
                addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i - 1];
                addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i - 1];
                addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i - 1];
                addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i - 1];
            }
        }
        else if(integration_mode == IntegrationMode::Quaternion)
        {
            for(int i = 1; i < frame_size; ++i)
            {
                addr_shift[reinterpret_cast<long>(para_Q[i])] = para_Q[i - 1];
                addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i - 1];
                addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i - 1];
                addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i - 1];
                addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i - 1];
            }
        }

        std::vector<double*> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);

        if(last_marginalization_info)
            delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginalization_parameter_blocks = parameter_blocks;
    }

    else
    {
        if(  last_marginalization_info &&
            (  (integration_mode == IntegrationMode::SE23 && count(begin(last_marginalization_parameter_blocks), end(last_marginalization_parameter_blocks), para_T[frame_size - 2])) ||
               (integration_mode == IntegrationMode::SO3 && count(begin(last_marginalization_parameter_blocks), end(last_marginalization_parameter_blocks), para_R[frame_size - 2])) ||
               (integration_mode == IntegrationMode::Quaternion && count(begin(last_marginalization_parameter_blocks), end(last_marginalization_parameter_blocks), para_Q[frame_size - 2]))
             )
           )
        {
            MarginalizationInfo* marginalization_info = new MarginalizationInfo();
            Mat2Para();
            if(last_marginalization_info)
            {
                std::vector<int> drop_set;
                if(integration_mode == IntegrationMode::SE23)
                {
                    for(int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); ++i)
                    {
                        if(last_marginalization_parameter_blocks[i] == para_T[frame_size - 2])
                        {
                            drop_set.push_back(i);
                            break;
                        }
                    }
                }
                else if(integration_mode == IntegrationMode::SO3)
                {
                    for(int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); ++i)
                    {
                        if(last_marginalization_parameter_blocks[i] == para_P[frame_size - 2] ||
                           last_marginalization_parameter_blocks[i] == para_R[frame_size - 2])
                            drop_set.push_back(i);
                        if(drop_set.size() == 2)
                            break;
                    }
                }
                else if(integration_mode == IntegrationMode::Quaternion)
                {
                    for(int i = 0; i < static_cast<int>(last_marginalization_parameter_blocks.size()); ++i)
                    {
                        if(last_marginalization_parameter_blocks[i] == para_P[frame_size - 2] ||
                           last_marginalization_parameter_blocks[i] == para_Q[frame_size - 2])
                            drop_set.push_back(i);
                        if(drop_set.size() == 2)
                            break;
                    }
                }

                if(!drop_set.empty())
                {
                    MarginalizationFactor* marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                    ResidualBlockInfo *residual_block_info = new ResidualBlockInfo(marginalization_factor, nullptr,
                                                                                   last_marginalization_parameter_blocks, drop_set);
                    marginalization_info->AddResidualBlockInfo(residual_block_info);
                }
            }

            marginalization_info->PreMarginalize();
            marginalization_info->Marginalize();
            std::unordered_map<long, double*> addr_shift;

            if(integration_mode == IntegrationMode::SE23)
            {
                for(int i = 0; i < frame_size; ++i)
                {
                    if(i == frame_size - 2) continue;
                    else if(i == frame_size - 1)
                    {
                        addr_shift[reinterpret_cast<long>(para_T[i])] = para_T[i - 1];
                        addr_shift[reinterpret_cast<long>(para_B[i])] = para_B[i - 1];
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_T[i])] = para_T[i];
                        addr_shift[reinterpret_cast<long>(para_B[i])] = para_B[i];
                    }
                }
            }
            else if(integration_mode == IntegrationMode::SO3)
            {
                for(int i = 0; i < frame_size; ++i)
                {
                    if(i == frame_size - 2) continue;
                    else if(i == frame_size - 1)
                    {
                        addr_shift[reinterpret_cast<long>(para_R[i])] = para_R[i - 1];
                        addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i - 1];
                        addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i - 1];
                        addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i - 1];
                        addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i - 1];
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_R[i])] = para_R[i];
                        addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i];
                        addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i];
                        addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i];
                        addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i];
                    }
                }
            }
            else if(integration_mode == IntegrationMode::Quaternion)
            {
                for(int i = 0; i < frame_size; ++i)
                {
                    if(i == frame_size - 2) continue;
                    else if(i == frame_size - 1)
                    {
                        addr_shift[reinterpret_cast<long>(para_Q[i])] = para_Q[i - 1];
                        addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i - 1];
                        addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i - 1];
                        addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i - 1];
                        addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i - 1];
                    }
                    else
                    {
                        addr_shift[reinterpret_cast<long>(para_Q[i])] = para_Q[i];
                        addr_shift[reinterpret_cast<long>(para_V[i])] = para_V[i];
                        addr_shift[reinterpret_cast<long>(para_P[i])] = para_P[i];
                        addr_shift[reinterpret_cast<long>(para_bg[i])] = para_bg[i];
                        addr_shift[reinterpret_cast<long>(para_ba[i])] = para_ba[i];
                    }
                }
            }

            std::vector<double*> parameter_blocks = marginalization_info->GetParameterBlocks(addr_shift);

            if(last_marginalization_info)
                delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginalization_parameter_blocks = parameter_blocks;
        }
    }
}

void Estimator::Mat2Para()
{
    if(integration_mode == IntegrationMode::SE23)
    {
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Map<Eigen::Vector9d> tmp_T(para_T[i]);
            Eigen::Map<Eigen::Vector6d> tmp_B(para_B[i]);
            tmp_T = LogSE23(Ts[i]);
            tmp_B = Bs[i];
        }
    }
    else if(integration_mode == IntegrationMode::SO3)
    {
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Map<Eigen::Matrix3d> tmp_R(para_R[i]);
            Eigen::Map<Eigen::Vector3d> tmp_V(para_V[i]);
            Eigen::Map<Eigen::Vector3d> tmp_P(para_P[i]);
            Eigen::Map<Eigen::Vector3d> tmp_bg(para_bg[i]);
            Eigen::Map<Eigen::Vector3d> tmp_ba(para_ba[i]);
            tmp_R = Ts[i].block<3, 3>(0, 0);
            tmp_V = Ts[i].block<3, 1>(0, 3);
            tmp_P = Ts[i].block<3, 1>(0, 4);
            tmp_bg = Bs[i].head<3>();
            tmp_ba = Bs[i].tail<3>();
        }
    }
    else if(integration_mode == IntegrationMode::Quaternion)
    {
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Map<Eigen::Vector4d> tmp_Q(para_Q[i]);
            Eigen::Map<Eigen::Vector3d> tmp_V(para_V[i]);
            Eigen::Map<Eigen::Vector3d> tmp_P(para_P[i]);
            Eigen::Map<Eigen::Vector3d> tmp_bg(para_bg[i]);
            Eigen::Map<Eigen::Vector3d> tmp_ba(para_ba[i]);
            Eigen::Quaterniond q(Ts[i].block<3, 3>(0, 0));
            tmp_Q.x() = q.x();
            tmp_Q.y() = q.y();
            tmp_Q.z() = q.z();
            tmp_Q.w() = q.w();
            tmp_V = Ts[i].block<3, 1>(0, 3);
            tmp_P = Ts[i].block<3, 1>(0, 4);
            tmp_bg = Bs[i].head<3>();
            tmp_ba = Bs[i].tail<3>();
        }
    }
    Eigen::VectorXd invdep_vector = feature_manager->GetInvDepthVector();
    const int points_count = feature_manager->solved_feature_ids.size();
    for(int i = 0; i < points_count; ++i)
    {
        para_invdep[i][0] = invdep_vector[i];
    }
}

void Estimator::Para2Mat()
{
    Eigen::Matrix3d origin_R0 = Ts[0].block<3, 3>(0, 0);
    Eigen::Vector3d origin_P0 = Ts[0].block<3, 1>(0, 4);
    Eigen::Matrix5d T_diff = Eigen::Matrix5d::Identity();

    if(integration_mode == IntegrationMode::SE23)
    {
        Eigen::Map<Eigen::Vector9d> opt_s0(para_T[0]);
        Eigen::Matrix5d opt_T0 = ExpSE23(opt_s0);
        Eigen::Matrix3d opt_R0 = opt_T0.block<3, 3>(0, 0);
        Eigen::Vector3d opt_P0 = opt_T0.block<3, 1>(0, 4);

        Eigen::Vector3d ypr_origin_R0 = R2ypr(origin_R0);
        Eigen::Vector3d ypr_opt_R0 = R2ypr(opt_R0);
        double y_diff = ypr_origin_R0.x() - ypr_opt_R0.x();
        T_diff.block<3, 3>(0, 0) = ypr2R(Eigen::Vector3d(y_diff, 0, 0));
        if(abs(abs(ypr_origin_R0.y()) - 90) < 1.0 || abs(abs(ypr_opt_R0.y()) - 90) < 1.0)// 奇异性？
        {
            T_diff.block<3, 3>(0, 0) = origin_R0 * opt_R0.transpose();
        }
        T_diff.block<3, 1>(0, 4) = - T_diff.block<3, 3>(0, 0)*opt_P0 + origin_P0;
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Map<Eigen::Vector9d> opt_Ti(para_T[i]);
            Eigen::Map<Eigen::Vector6d> opt_bi(para_B[i]);
            Ts[i] = T_diff*ExpSE23(opt_Ti);
            Bs[i] = opt_bi;
        }
    }
    else if(integration_mode == IntegrationMode::SO3)
    {
        Eigen::Map<Eigen::Matrix3d> opt_R0(para_R[0]);
        Eigen::Map<Eigen::Vector3d> opt_P0(para_P[0]);

        Eigen::Vector3d ypr_origin_R0 = R2ypr(origin_R0);
        Eigen::Vector3d ypr_opt_R0 = R2ypr(opt_R0);
        double y_diff = ypr_origin_R0.x() - ypr_opt_R0.x();
        T_diff.block<3, 3>(0, 0) = ypr2R(Eigen::Vector3d(y_diff, 0, 0));
        if(abs(abs(ypr_origin_R0.y()) - 90) < 1.0 || abs(abs(ypr_opt_R0.y()) - 90) < 1.0)
        {
            T_diff.block<3, 3>(0, 0) = origin_R0 * opt_R0.transpose();
        }
        T_diff.block<3, 1>(0, 4) = - T_diff.block<3, 3>(0, 0)*opt_P0 + origin_P0;
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Map<Eigen::Matrix3d> opt_Ri(para_R[i]);
            Eigen::Map<Eigen::Vector3d> opt_Vi(para_V[i]);
            Eigen::Map<Eigen::Vector3d> opt_Pi(para_P[i]);
            Eigen::Map<Eigen::Vector3d> opt_bgi(para_bg[i]);
            Eigen::Map<Eigen::Vector3d> opt_bai(para_ba[i]);

            Eigen::Matrix5d opt_Ti = Eigen::Matrix5d::Identity();
            opt_Ti.block<3, 3>(0, 0) = opt_Ri;
            opt_Ti.block<3, 1>(0, 3) = opt_Vi;
            opt_Ti.block<3, 1>(0, 4) = opt_Pi;
            Ts[i] = T_diff*opt_Ti;
            Bs[i].head<3>() = opt_bgi;
            Bs[i].tail<3>() = opt_bai;
        }
    }
    else if(integration_mode == IntegrationMode::Quaternion)
    {
        Eigen::Matrix3d opt_R0 = Eigen::Quaterniond(para_Q[0][3], para_Q[0][0], para_Q[0][1], para_Q[0][2]).toRotationMatrix();
        Eigen::Map<Eigen::Vector3d> opt_P0(para_P[0]);

        Eigen::Vector3d ypr_origin_R0 = R2ypr(origin_R0);
        Eigen::Vector3d ypr_opt_R0 = R2ypr(opt_R0);
        double y_diff = ypr_origin_R0.x() - ypr_opt_R0.x();
        T_diff.block<3, 3>(0, 0) = ypr2R(Eigen::Vector3d(y_diff, 0, 0));
        if(abs(abs(ypr_origin_R0.y()) - 90) < 1.0 || abs(abs(ypr_opt_R0.y()) - 90) < 1.0)
        {
            T_diff.block<3, 3>(0, 0) = origin_R0 * opt_R0.transpose();
        }

        T_diff.block<3, 1>(0, 4) = - T_diff.block<3, 3>(0, 0)*opt_P0 + origin_P0;
        for(int i = 0; i < frame_size; ++i)
        {
            Eigen::Matrix3d opt_Ri = Eigen::Quaterniond(para_Q[i][3], para_Q[i][0], para_Q[i][1], para_Q[i][2]).toRotationMatrix();
            Eigen::Map<Eigen::Vector3d> opt_Vi(para_V[i]);
            Eigen::Map<Eigen::Vector3d> opt_Pi(para_P[i]);
            Eigen::Map<Eigen::Vector3d> opt_bgi(para_bg[i]);
            Eigen::Map<Eigen::Vector3d> opt_bai(para_ba[i]);

            Eigen::Matrix5d opt_Ti = Eigen::Matrix5d::Identity();
            opt_Ti.block<3, 3>(0, 0) = opt_Ri;
            opt_Ti.block<3, 1>(0, 3) = opt_Vi;
            opt_Ti.block<3, 1>(0, 4) = opt_Pi;
            Ts[i] = T_diff*opt_Ti;
            Bs[i].head<3>() = opt_bgi;
            Bs[i].tail<3>() = opt_bai;
        }
    }

    int points_count = feature_manager->solved_feature_ids.size();
    Eigen::VectorXd invdep_vector(points_count);
    for(int i = 0; i < points_count; ++i)
    {
        invdep_vector[i] = para_invdep[i][0];
    }
    feature_manager->SetInvDepthVector(invdep_vector);
}


void Estimator::UpdateCameraCSbyTs()
{
    if(vRcw.size() != Ts.size())
    {
        vRcw.resize(Ts.size());
        vtcw.resize(Ts.size());
    }
    for(size_t i = 0; i < Ts.size(); ++i)
    {
        const Eigen::Matrix5d Tcw = InverseSE23(Ts[i]*TbcG);
        vRcw[i] = Tcw.block<3, 3>(0, 0);
        vtcw[i] = Tcw.block<3, 1>(0, 4);
    }
}

void Estimator::PubKeyframe()
{
    const int frame_size = Ts.size();
    Eigen::Matrix5d ex_pose = Ts[frame_size - 2];
    Eigen::Isometry3d pose;
    pose.linear() = ex_pose.block<3, 3>(0, 0);
    pose.translation() = ex_pose.block<3, 1>(0, 4);
    pose_graph_ptr->PushBuf(frame_ids[frame_size - 2], pose);
}

bool Estimator::GetState(const int cur_id, const std::shared_ptr<ImuIntegration>& cur_imu, std::map<int, Eigen::Isometry3d>& pose_list, std::map<int, Eigen::Vector3f>& mappoint_list)
{
    std::unique_lock<std::mutex> lock(m_busy);
    if(!is_initialized)
        return false;

    for(int i = 0; i < frame_ids.size(); ++i)
    {
        Eigen::Isometry3d pose;
        Eigen::Matrix5d Tsc = Ts[i]*TbcG;
        pose.linear() = RotationCorrect(Tsc.block<3, 3>(0, 0));
        pose.translation() = Tsc.block<3, 1>(0, 4);
        pose_list[frame_ids[i]] = pose;
    }

    if(reintegrated_num > 0)
    {
        cur_imu->SetBias(Bs.back());
        cur_imu->Reintegrated();
    }
    Eigen::Matrix5d cur_Tsc = cur_imu->Gamma*FaiSE23(Ts.back(), cur_imu->dT)*cur_imu->r*TbcG;
    Eigen::Isometry3d cur_pose;
    cur_pose.linear() = RotationCorrect(cur_Tsc.block<3, 3>(0, 0));
    cur_pose.translation() = cur_Tsc.block<3, 1>(0, 4);
    pose_list[cur_id] = cur_pose;

    if(!feature_manager->solved_feature_ids.empty())
    {
        for(auto f_id : feature_manager->solved_feature_ids)
        {
            auto f = feature_manager->all_features[f_id];
            int ref_frame_id = f->reference_frame_id;
            Eigen::Vector3d normalized_point = f->reference_point.tail<2>().homogeneous();
            double dep = 1.0/f->inv_dep;
            Eigen::Vector3d position = pose_list[ref_frame_id]*(normalized_point*dep);
            mappoint_list[f->feature_id] = position.cast<float>();
        }
    }
    return true;
}

Eigen::Isometry3d Estimator::GetCameraPose()
{
    std::unique_lock<std::mutex> lock(m_camera_pose);
    return camera_pose;
}

void Estimator::SetQuit(bool x)
{
    std::unique_lock<std::mutex> lock(m_quit);
    quit_flag = x;
}

bool Estimator::GetQuit()
{
    std::unique_lock<std::mutex> lock(m_quit);
    return quit_flag;
}
