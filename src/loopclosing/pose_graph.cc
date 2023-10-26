
#include "pose_graph.h"

template<typename T1, typename T2>
void ReduceVector(std::vector<T1> &v, std::vector<T2> s)
{
    int j = 0;
    for(int i = 0, iend = int(v.size()); i < iend; ++i)
        if(s[i])
            v[j++] = v[i];
    v.resize(j);
}


struct AreaCircle {
    AreaCircle(const float r) : r2(r*r) {}
    bool operator()(const cv::Point2f &p1, const cv::Point2f &p2) { return (p1 - p2).dot(p1 - p2) < r2; }
    float r2;
};

struct AreaLine {
    AreaLine(const Eigen::Vector3d &_line, const float r) : line(_line), r2(r * r)
    {
        inv_denominator = 1.0 / line.head(2).squaredNorm();
    }
    bool operator()(const cv::Point2f &pt)
    {
        const Eigen::Vector3d p(pt.x, pt.y, 1);
        const float num2 = line.dot(p);
        const float squareDist2 = num2 * num2 * inv_denominator;
        return squareDist2 < r2;
    }
    Eigen::Vector3d line;
    float r2;
    float inv_denominator;
};

int HammingDist(const cv::Mat &des1, const cv::Mat &des2)
{
    int dist = 0;
    for(int i = 0; i < 32; i++)
    {
        const std::bitset<8> &a = des1.at<uchar>(i);
        const std::bitset<8> &b = des2.at<uchar>(i);
        const std::bitset<8> c = a ^ b;
        dist += c.count();
    }
    return dist;
}

void PoseGraph::MatchTwoFrame(const std::shared_ptr<KeyFrame> &kf_query,
                              const std::shared_ptr<KeyFrame> &kf_train,
                              std::vector<cv::DMatch> &matches, const float radius,
                              const std::vector<int> &candidates)
{
    auto kpt_query = kf_query->GetKeypoints_norm();
    auto kpt_train = kf_train->GetKeypoints_norm();
    auto des_query = kf_query->GetDescriptors();
    auto des_train = kf_train->GetDescriptors();

    matches.resize(candidates.size());
    int matches_num = 0;

    Eigen::Isometry3d T1 = kf_query->GetCameraPose();
    Eigen::Isometry3d T2 = kf_train->GetCameraPose();
    Eigen::Matrix3d R1 = T1.linear();
    Eigen::Matrix3d R2 = T2.linear();
    Eigen::Vector3d t1 = T1.translation();
    Eigen::Vector3d t2 = T2.translation();
    const Eigen::Matrix3d E21 = R2.transpose()*Skew(t1 - t2)*R1;

    for(auto i : candidates)
    {
        cv::DMatch best_match(i, 0, 256);
        cv::DMatch second_best_match(i, 0, 256);
        const cv::Point2f &kpt1 = kpt_query[i];

        const Eigen::Vector3d p1(kpt1.x, kpt1.y, 1);
        const Eigen::Vector3d ep_line = E21 * p1;

        AreaLine InLine(ep_line, radius);

        const cv::Mat &des1 = des_query.row(i);
        for(int j = 0; j < des_train.rows; ++j)
        {
            const cv::Point2f &kpt2 = kpt_train[j];
            if(!InLine(kpt2)) continue;

            const cv::Mat &des2 = des_train.row(j);
            int dist = HammingDist(des1, des2);
            if(dist < best_match.distance)
            {
                second_best_match = best_match;
                best_match.distance = dist;
                best_match.trainIdx = j;
            }
            else if(dist < second_best_match.distance)
            {
                second_best_match.distance = dist;
                second_best_match.trainIdx = j;
            }
        }
        if(best_match.distance < 80 &&
                best_match.distance < 0.9 * second_best_match.distance)
            matches[matches_num++] = best_match;
    }
    matches.resize(matches_num);
}

void PoseGraph::MatchTwoFrameInCircle(const std::shared_ptr<KeyFrame> &kf_query,
                                      const std::shared_ptr<KeyFrame> &kf_train,
                                      std::vector<cv::DMatch> &matches, const float radius,
                                      const std::vector<int> &candidates)
{
    auto kpt_query = kf_query->GetKeypoints();
    auto kpt_train = kf_train->GetKeypoints();
    auto des_query = kf_query->GetDescriptors();
    auto des_train = kf_train->GetDescriptors();

    matches.resize(candidates.size());
    int matches_num = 0;
    AreaCircle InCircle(radius);

    for(auto i : candidates)
    {
        cv::DMatch best_match(i, 0, 256);
        cv::DMatch second_best_match(i, 0, 256);
        const cv::KeyPoint &kpt1 = kpt_query[i];
        const cv::Mat &des1 = des_query.row(i);
        for(int j = 0; j < des_train.rows; ++j)
        {
            const cv::KeyPoint &kpt2 = kpt_train[j];
            if(!InCircle(kpt1.pt, kpt2.pt)) continue;

            const cv::Mat &des2 = des_train.row(j);
            int dist = HammingDist(des1, des2);
            if(dist < best_match.distance)
            {
                second_best_match = best_match;
                best_match.distance = dist;
                best_match.trainIdx = j;
            }
            else if(dist < second_best_match.distance)
            {
                second_best_match.distance = dist;
                second_best_match.trainIdx = j;
            }
        }
        if(best_match.distance < 80
                && best_match.distance < 0.9 * second_best_match.distance)
            matches[matches_num++] = best_match;
    }
    matches.resize(matches_num);
}


PoseGraph::PoseGraph()
{
    T_drift.setIdentity();
    earliest_loop_index = -1;
    last_loop = -1;

    keyframe_id = 0;

    voc = new DBoW3::Vocabulary(std::string(DATA_DIR) + "DBow3/orbvoc.dbow3");
    db.setVocabulary(*voc, false);
    SetQuit(false);
}

void PoseGraph::PushBuf(const int frame_id, const Eigen::Isometry3d& pose)
{
    std::unique_lock<std::mutex> lock(m_buf);
    frame_id_buf.push(frame_id);
    pose_buf.push(pose);
}

void PoseGraph::Run()
{
    while(!GetQuit())
    {
        std::chrono::milliseconds dura(5);
        std::this_thread::sleep_for(dura);

        int frame_id_msg;
        Eigen::Isometry3d pose_msg;
        io::timestamp_t timestamp_msg;
        cv::Mat image_msg;
        bool getting_data = false;

        m_buf.lock();

        if(!frame_id_buf.empty())
        {
            frame_id_msg = frame_id_buf.front();
            pose_msg = pose_buf.front();
            frame_id_buf.pop();
            pose_buf.pop();

            timestamp_msg = image_processor_ptr->GetTimestamp(frame_id_msg);
            image_msg = image_processor_ptr->GetImage(frame_id_msg);

            getting_data = true;
        }
        m_buf.unlock();

        if(getting_data)
        {
            std::shared_ptr<KeyFrame> keyframe = std::make_shared<KeyFrame>(timestamp_msg, keyframe_id, frame_id_msg, pose_msg, image_msg);
            AddKeyFrame(keyframe, !!loop_closure_enable);
            ++keyframe_id;
        }
    }

    std::ofstream foutC1(no_loop_save_path, std::ios::out | std::ios::trunc);
    foutC1.setf(std::ios::fixed, std::ios::floatfield);
    for(auto kf : keyframelist)
    {
        Eigen::Matrix3d tmp_R1 = kf->Twb.rotation();
        Eigen::Vector3d tmp_t1 = kf->Twb.translation();
        Eigen::Quaterniond tmp_Q1 = Eigen::Quaterniond(tmp_R1);
        foutC1.precision(9);
        foutC1 << kf->time_stamp*1e-9 << " ";
        foutC1.precision(6);
        foutC1 << tmp_t1.x() << " "
              << tmp_t1.y() << " "
              << tmp_t1.z() << " "
              << tmp_Q1.x() << " "
              << tmp_Q1.y() << " "
              << tmp_Q1.z() << " "
              << tmp_Q1.w() << std::endl;
    }
    foutC1.close();

    if(loop_closure_enable)
    {
        std::ofstream foutC2(loop_save_path, std::ios::out | std::ios::trunc);
        foutC2.setf(std::ios::fixed, std::ios::floatfield);
        for(auto kf : keyframelist)
        {
            Eigen::Matrix3d tmp_R2 = kf->Twb_update.rotation();
            Eigen::Vector3d tmp_t2 = kf->Twb_update.translation();
            Eigen::Quaterniond tmp_Q2 = Eigen::Quaterniond(tmp_R2);
            foutC2.precision(9);
            foutC2 << kf->time_stamp*1e-9 << " ";
            foutC2.precision(6);
            foutC2 << tmp_t2.x() << " "
                  << tmp_t2.y() << " "
                  << tmp_t2.z() << " "
                  << tmp_Q2.x() << " "
                  << tmp_Q2.y() << " "
                  << tmp_Q2.z() << " "
                  << tmp_Q2.w() << std::endl;
        }
        foutC2.close();
    }
    std::cout << "total saved frame size: " << keyframelist.size() << std::endl;
}

void PoseGraph::AddKeyFrame(std::shared_ptr<KeyFrame> cur_kf, bool flag_detect_loop)
{
    Eigen::Isometry3d Twb = cur_kf->GetPose();
    Eigen::Isometry3d Twb_update = T_drift*Twb;
    cur_kf->SetUpdatePose(Twb_update);

    voc->transform(cur_kf->GetDescriptors(), cur_kf->dbow_vector);

    m_keyframelist.lock();
    keyframelist.push_back(cur_kf);
    m_keyframelist.unlock();

    if(flag_detect_loop)
    {

        int loop_index = DetectLoop(cur_kf);

        if(loop_index != -1)
        {
            if(FindConnection(cur_kf, loop_index))
            {
                std::cout << "FindConnection" << std::endl;
                m_keyframelist.lock();
                Optimization_Loop_4DoF();
                m_keyframelist.unlock();
            }
        }
    }

    db.add(cur_kf->GetDescriptors());
}

int PoseGraph::DetectLoop(std::shared_ptr<KeyFrame> keyframe)
{
    const int frame_index = keyframe->GetIndex();
    if(frame_index < 100 || frame_index - last_loop < 20)
        return -1;

    DBoW3::QueryResults ret;
    db.query(keyframe->GetDescriptors(), ret, 5, frame_index - 20);

    bool find_loop = false;
    const float threshold = 0.03;
    if(ret.size() >= 1 && ret[0].Score > threshold)
    {
        find_loop = true;
    }
    if(find_loop)
    {
        int min_index = -1;
        for(unsigned int i = 0; i < ret.size(); i++)
        {
            if(min_index == -1 || (ret[i].Id < min_index && ret[i].Score > threshold))
            {
                int loop_idx = ret[i].Id;
                Eigen::Isometry3d loop_pose = GetKeyFrame(loop_idx)->GetUpdatePose();
                Eigen::Isometry3d cur_pose = keyframe->GetUpdatePose();

                double distance = (loop_pose.translation() - cur_pose.translation()).norm();
                double d_yaw = fabs(NormalizeAngle(R2ypr(loop_pose.linear()).x() - R2ypr(cur_pose.linear()).x()));
                if(distance < 10.0 && d_yaw < 20.0)
                    min_index = ret[i].Id;
            }
        }
        return min_index;
    }
    else
        return -1;
}

bool PoseGraph::FindConnection(std::shared_ptr<KeyFrame> &kf_cur, const int loop_id)
{
    double threshold1 = 10.0;
    double threshold2 = 5.0;

    int min_pnp_num = 15;
    std::shared_ptr<KeyFrame> kf_old = GetKeyFrame(loop_id);

    std::vector<PnPMapPoint> mappoints;
    for(int i = 0, iend = kf_old->GetKeypoints().size(); i < iend; ++i)
    {
        PnPMapPoint mappoint;
        mappoint.id = i;
        mappoint.loop_observations.insert(std::make_pair(loop_id, i));
        mappoints.push_back(mappoint);
    }

    const int side_winsize = 3;
    const int start_id = std::max(0, loop_id - side_winsize);
    const int end_id = loop_id + side_winsize;
    for(int cur_id = start_id; cur_id <= end_id; ++cur_id)
    {
        const std::shared_ptr<KeyFrame> kf_neighbor = GetKeyFrame(cur_id);
        if(!kf_neighbor || cur_id == loop_id)
            continue;
        else
        {
            std::vector<cv::DMatch> neighbor_matches;
            MatchTwoFrame(kf_old, kf_neighbor, neighbor_matches, threshold1/fxG);
            for(auto &m : neighbor_matches)
            {
                mappoints[m.queryIdx].loop_observations.insert(std::make_pair(cur_id, m.trainIdx));
            }
        }
    }

    int idx_count = -1;
    for(auto it_p3d = mappoints.begin(); it_p3d != mappoints.end(); ++it_p3d)
    {
        ++idx_count;
        const int obs_size = it_p3d->loop_observations.size();
        if(obs_size < 3)
            continue;
        it_p3d->good_matched = true;

        Eigen::Isometry3d T0 = GetKeyFrame(loop_id)->GetCameraPose();

        int svd_idx = 0;
        Eigen::MatrixXd svd_A(2*obs_size, 4);
        for(auto &it_pt : it_p3d->loop_observations)
        {
            std::shared_ptr<KeyFrame> kf_process = GetKeyFrame(it_pt.first);
            Eigen::Isometry3d T1 = kf_process->GetCameraPose();

            Eigen::Isometry3d T10 = T1.inverse()*T0;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = T10.rotation();
            P.rightCols<1>() = T10.translation();

            cv::Point2f p = kf_process->GetKeypoints_norm()[it_pt.second];
            Eigen::Vector3d pun(p.x, p.y, 1.0);
            Eigen::Vector3d f = pun.normalized();
            svd_A.row(svd_idx++) = f[0]*P.row(2) - f[2]*P.row(0);
            svd_A.row(svd_idx++) = f[1]*P.row(2) - f[2]*P.row(1);
        }

        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Eigen::Vector3d svd = svd_V.hnormalized();

        if(svd.z() > 0.1)
        {
            const Eigen::Vector3d p3d = T0*svd;

            double ave_err = 0;
            for(auto &obs : it_p3d->loop_observations)
            {
                std::shared_ptr<KeyFrame> kf_process = GetKeyFrame(obs.first);
                cv::Point2f pun = kf_process->GetKeypoints_norm()[obs.second];

                Eigen::Isometry3d Ti = kf_process->GetCameraPose();
                const Eigen::Vector3d _p3d = Ti.inverse()*p3d;
                ave_err += (_p3d.hnormalized() - Eigen::Vector2d(pun.x, pun.y)).norm();
            }
            ave_err /= obs_size;
            if(ave_err < threshold2/fxG)
            {
                it_p3d->world_point = p3d;
                it_p3d->good_triangulated = true;
            }
        }
    }

    std::vector<int> solved_candidates_old;
    for(auto &mappoint : mappoints)
    {
        if(mappoint.good_triangulated)
            solved_candidates_old.push_back(mappoint.id);
    }
    std::vector<cv::DMatch> matches;
    MatchTwoFrameInCircle(kf_old, kf_cur, matches, 150, solved_candidates_old);

    if(matches.size() < min_pnp_num)
        return false;

    std::vector<Eigen::Vector2d> matched_norm1, matched_norm2;
    matched_norm1.resize(matches.size());
    matched_norm2.resize(matches.size());
    auto keypoints_norm1 = kf_old->GetKeypoints_norm();
    auto keypoints_norm2 = kf_cur->GetKeypoints_norm();
    for(int i = 0, iend = matches.size(); i < iend; ++i)
    {
        cv::Point2f &p1 = keypoints_norm1[matches[i].queryIdx];
        cv::Point2f &p2 = keypoints_norm2[matches[i].trainIdx];
        matched_norm1[i] = Eigen::Vector2d(p1.x, p1.y);
        matched_norm2[i] = Eigen::Vector2d(p2.x, p2.y);
    }

    FundamentalEstimator fundamentalestimator(0.6, threshold2/fxG, 0.99);
    std::vector<bool> status_F(matches.size(), false);
    Timer timer;
    timer.Start();
    bool F_solve_flag = fundamentalestimator.FundamentalRansac(matched_norm1, matched_norm2, status_F);

    ReduceVector(matches, status_F);

    if(matches.size() < min_pnp_num || !F_solve_flag)
    {
        return false;
    }

    std::vector<PnPMatch> pnp_matches;
    for(auto &m : matches)
    {
        mappoints[m.queryIdx].good_connected = true;
        const Eigen::Vector3d p = mappoints[m.queryIdx].world_point;
        const cv::Point2f p2d = kf_cur->GetKeypoints_norm()[m.trainIdx];
        pnp_matches.push_back(PnPMatch(p, Eigen::Vector2d(p2d.x, p2d.y)));
        mappoints[m.queryIdx].cur_observations.insert(std::make_pair(kf_cur->index, m.trainIdx));
    }

    std::map<int, Eigen::Matrix3d> Rcref;
    std::map<int, Eigen::Vector3d> tcref;

    {
        int added_match = 0;
        const int cur_id = kf_cur->GetIndex();
        const int side_winsize_cur = 2;
        const int l_id = std::max(cur_id - side_winsize_cur, 0);
        const int r_id = std::min(cur_id + side_winsize_cur, keyframe_id);

        Eigen::Isometry3d Tref = kf_cur->GetCameraPose();
        for(int id = l_id; id <= r_id; ++id)
        {
            if(id == cur_id)
            {
                Rcref[id] = Eigen::Matrix3d::Identity();
                tcref[id] = Eigen::Vector3d::Zero();
                continue;
            }
        }
    }

    PnPEstimator pnpestimator(0.65, threshold2/fxG, 0.99, 10);
    std::vector<bool> status2(matches.size(), false);

    Eigen::Isometry3d Twc_new = kf_cur->GetCameraPose();
    Eigen::Isometry3d Tcw_new = Twc_new.inverse();

    bool solve_flag = pnpestimator.PnPRansac(pnp_matches, Tcw_new, status2);

    if(solve_flag)
    {
        ReduceVector(matches, status2);

        Twc_new = Tcw_new.inverse();

        Eigen::Isometry3d Tcb(RcbG);
        Tcb.pretranslate(tcbG);
        Eigen::Isometry3d Twb_cur = Twc_new*Tcb;

        Eigen::Isometry3d T_old = kf_old->GetPose();

        Eigen::Isometry3d T12_new = T_old.inverse()*Twb_cur;
        kf_cur->SetLoopMessage(T12_new, loop_id);

        if(earliest_loop_index > loop_id || earliest_loop_index == -1)
            earliest_loop_index = loop_id;
        last_loop = kf_cur->index;

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.num_threads = 1;
        options.max_solver_time_in_seconds = 0.5;
        options.max_num_iterations = 10;
        ceres::Solver::Summary summary;
        ceres::LossFunction *loss_function;
        loss_function = new ceres::CauchyLoss(1.0);
        RotationMatrixParameterization* local_parameterization = new RotationMatrixParameterization;

        double ptr_rotation[9];
        double ptr_translation[3];
        double ptr_mappoint[matches.size()][3];
        Eigen::Map<Eigen::Matrix3d> R_tmp(ptr_rotation);
        Eigen::Map<Eigen::Vector3d> t_tmp(ptr_translation);
        R_tmp = Tcw_new.rotation();
        t_tmp = Tcw_new.translation();

        for(int i = 0; i < matches.size(); ++i)
        {
            int old_id = matches[i].queryIdx;

            Eigen::Vector3d world_point = mappoints[old_id].world_point;
            ptr_mappoint[i][0] = world_point.x();
            ptr_mappoint[i][1] = world_point.y();
            ptr_mappoint[i][2] = world_point.z();

            for(auto & obs : mappoints[old_id].loop_observations)
            {
                Eigen::Matrix3d Rwc_obs;
                Eigen::Vector3d twc_obs;
                Eigen::Vector2d puv_obs;
                std::shared_ptr<KeyFrame> kf = GetKeyFrame(obs.first);

                Eigen::Isometry3d T_obs = kf->GetCameraPose();
                Rwc_obs = T_obs.rotation();
                twc_obs = T_obs.translation();

                cv::Point2f puv = kf->GetKeypoints_norm()[obs.second];
                puv_obs << puv.x, puv.y;
                FixedPose* factor = new FixedPose(Rwc_obs, twc_obs, puv_obs);
                problem.AddResidualBlock(factor, nullptr, ptr_mappoint[i]);
            }
            for(auto & obs : mappoints[old_id].cur_observations)
            {
                const int frame_id = obs.first;
                Eigen::Vector2d puv_obs;
                std::shared_ptr<KeyFrame> kf = GetKeyFrame(frame_id);
                cv::Point2f puv = kf->GetKeypoints_norm()[obs.second];
                puv_obs << puv.x, puv.y;
                RelaxedPose* factor = new RelaxedPose(puv_obs, Rcref[frame_id], tcref[frame_id]);
                problem.AddResidualBlock(factor, nullptr, ptr_rotation, ptr_translation, ptr_mappoint[i]);
            }
        }

        problem.SetParameterization(ptr_rotation, local_parameterization);

        ceres::Solve(options, &problem, &summary);

        Tcw_new.linear() = R_tmp;
        Tcw_new.translation() = t_tmp;

        Twc_new = Tcw_new.inverse();
        Twb_cur = Twc_new*Tcb;
        T_old = kf_old->GetPose();
        T12_new = T_old.inverse()*Twb_cur;
        kf_cur->SetLoopMessage(T12_new, loop_id);
        if(earliest_loop_index > loop_id || earliest_loop_index == -1)
            earliest_loop_index = loop_id;
        last_loop = kf_cur->index;
        return true;
    }
    else
    {
        return false;
    }
}

std::shared_ptr<KeyFrame> PoseGraph::GetKeyFrame(const int index)
{
    if(index < 0 || index >= (int)keyframelist.size())
        return nullptr;
    else
        return *(keyframelist.begin() + index);
}

void PoseGraph::Optimization_Loop_4DoF()
{
    Timer time;
    time.Start();

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 1;
    options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.1);
    int kfsize = keyframelist.size();
    int first_loop_index = GetFirstOptFrameId(keyframelist.back()->loop_index);

    double t_array[kfsize - first_loop_index][3];
    Eigen::Matrix3d R_array[kfsize - first_loop_index];
    double euler_array[kfsize - first_loop_index][3];
    ceres::LocalParameterization* angle_local_parameterization =
            AngleLocalParameterization::Create();

    Eigen::Isometry3d T_before_opt = keyframelist[first_loop_index]->GetPose();
    for(int kfidx = first_loop_index, i = 0; kfidx < kfsize; ++kfidx, ++i)
    {
        Eigen::Vector3d tmp_t;
        Eigen::Matrix3d tmp_r;
        Eigen::Isometry3d tmp_T = keyframelist[kfidx]->GetPose();
        t_array[i][0] = tmp_T.translation().x();
        t_array[i][1] = tmp_T.translation().y();
        t_array[i][2] = tmp_T.translation().z();
        R_array[i] = tmp_T.rotation();
        Eigen::Vector3d euler_angle = R2ypr(R_array[i]);
        euler_array[i][0] = euler_angle.x();
        euler_array[i][1] = euler_angle.y();
        euler_array[i][2] = euler_angle.z();

        problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
        problem.AddParameterBlock(t_array[i], 3);

        for(int j = 1; j < 5; j++)
        {
            if(i - j >= 0)
            {
                Eigen::Vector3d relative_t(t_array[i][0] - t_array[i-j][0], t_array[i][1] - t_array[i-j][1], t_array[i][2] - t_array[i-j][2]);
                relative_t = R_array[i-j].transpose()*relative_t;
                double relative_yaw = euler_array[i][0] - euler_array[i-j][0];

                Eigen::Vector3d euler_conncected = R2ypr(R_array[i-j]);
                ceres::CostFunction* cost_function = Loop4DoF::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                          relative_yaw, euler_conncected.y(), euler_conncected.z(), 1.f, 0.1f);

                problem.AddResidualBlock(cost_function, nullptr, euler_array[i-j], t_array[i-j], euler_array[i], t_array[i]);
            }
        }

        const int loopid = keyframelist[kfidx]->loop_index;
        if(loopid >= 0)
        {
            Eigen::Isometry3d loop_Tij = keyframelist[kfidx]->loop_Tij;
            Eigen::Vector3d relative_t = loop_Tij.translation();
            int connected_index = loopid - first_loop_index;
            Eigen::Matrix3d R_loop = R_array[connected_index];
            Eigen::Vector3d euler_conncected = R2ypr(R_loop);
            Eigen::Matrix3d relative_R = loop_Tij.rotation();
            double relative_yaw = NormalizeAngle(R2ypr(R_loop*relative_R).x() - euler_conncected.x());
            ceres::CostFunction* cost_function = Loop4DoF::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                                      relative_yaw, euler_conncected.y(), euler_conncected.z(), 1.f, 0.1f);
            problem.AddResidualBlock(cost_function, loss_function, euler_array[connected_index],
                                     t_array[connected_index], euler_array[i], t_array[i]);
        }
    }

    ceres::Solve(options, &problem, &summary);

    Eigen::Isometry3d T_after_opt;
    T_after_opt.linear() = ypr2R(Eigen::Vector3d(euler_array[0][0], euler_array[0][1], euler_array[0][2]));
    T_after_opt.translation() = Eigen::Vector3d(t_array[0][0], t_array[0][1], t_array[0][2]);

    Eigen::Isometry3d dT = T_before_opt*T_after_opt.inverse();
    Eigen::Isometry3d drift = GetDrift(first_loop_index)*dT;

    for(int kfidx = first_loop_index + 1, i = 1; kfidx < kfsize; ++kfidx, ++i)
    {
        Eigen::Isometry3d T;
        T.linear() = ypr2R(Eigen::Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
        T.translation() = Eigen::Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
        Eigen::Isometry3d recursive_T = drift*T;
        keyframelist[kfidx]->SetUpdatePose(recursive_T);
    }

    Eigen::Isometry3d origin_T = keyframelist.back()->GetPose();
    Eigen::Isometry3d update_T = keyframelist.back()->GetUpdatePose();

    double yaw_drift = R2ypr(update_T.rotation()).x() - R2ypr(origin_T.rotation()).x();

    T_drift.linear() = ypr2R(Eigen::Vector3d(yaw_drift, 0, 0));
    T_drift.translation() = update_T.translation() - T_drift.linear()*origin_T.translation();
    AddNewLoop(first_loop_index, keyframelist.back()->GetIndex(), T_drift);
}


void PoseGraph::Optimization_Loop_6DoF()
{
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.num_threads = 1;
    options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function;
    loss_function = new ceres::HuberLoss(0.1);


    int kfsize = keyframelist.size();
    std::vector<double*> T_array;
    for(int i = earliest_loop_index; i < kfsize; ++i)
    {
        double* T_ptr = new double[6];
        T_array.push_back(T_ptr);
    }
    Eigen::Isometry3d Tw_array[kfsize - earliest_loop_index];
    ceres::LocalParameterization *local_parameterization = new SE3Parameterization();

    Eigen::Isometry3d T_before_opt = keyframelist[earliest_loop_index]->GetPose();

    for(int kfidx = earliest_loop_index, i = 0; kfidx < kfsize; ++kfidx, ++i)
    {
        Eigen::Isometry3d tmp_T = keyframelist[kfidx]->GetPose();
        Tw_array[i] = tmp_T;
        Eigen::Map<Eigen::Vector6d> para_T(T_array[i]);
        para_T = LogSE3(tmp_T);
        problem.AddParameterBlock(T_array[i], 6, local_parameterization);

        for(int j = 1; j < 5; ++j)
        {
            if(i - j >= 0)
            {
                Eigen::Isometry3d Tij = Tw_array[i - j].inverse()*Tw_array[i];
                Loop6DoF *f = new Loop6DoF(Tij);
                problem.AddResidualBlock(f, nullptr, T_array[i - j], T_array[i]);
            }
        }

        const int loopid = keyframelist[kfidx]->loop_index;
        if(loopid >= 0)
        {
            Eigen::Isometry3d Tij = keyframelist[kfidx]->loop_Tij;
            Loop6DoF *f = new Loop6DoF(Tij);
            problem.AddResidualBlock(f, loss_function, T_array[loopid - earliest_loop_index], T_array[i]);
        }
    }
    ceres::Solve(options, &problem, &summary);

    Eigen::Map<Eigen::Vector6d> para_Ts0(T_array[0]);
    Eigen::Isometry3d Ts0 = ExpSE3(para_Ts0);
    Eigen::Isometry3d T_after_opt = Ts0;
    Eigen::Isometry3d dT = T_before_opt*T_after_opt.inverse();

    for(int kfidx = earliest_loop_index + 1, i = 1; kfidx < kfsize; ++kfidx, ++i)
    {
        Eigen::Map<Eigen::Vector6d> para_Ts(T_array[i]);
        Eigen::Isometry3d opt_T = ExpSE3(para_Ts);
        Eigen::Isometry3d recursive_T = dT*opt_T;
        keyframelist[kfidx]->SetUpdatePose(recursive_T);
    }

    Eigen::Isometry3d origin_T = keyframelist.back()->GetPose();
    Eigen::Isometry3d update_T = keyframelist.back()->GetUpdatePose();
    T_drift = update_T*origin_T.inverse();
}


void PoseGraph::SetQuit(bool x)
{
    std::unique_lock<std::mutex> lock(m_quit);
    quit_flag = x;
}

bool PoseGraph::GetQuit()
{
    std::unique_lock<std::mutex> lock(m_quit);
    return quit_flag;
}

Eigen::Isometry3d PoseGraph::DriftRemove(const Eigen::Isometry3d& pose)
{
    return T_drift*pose;
}

std::vector<std::shared_ptr<KeyFrame>> PoseGraph::GetKeyframelist()
{
    std::unique_lock<std::mutex> lock(m_keyframelist);
    return keyframelist;
}

