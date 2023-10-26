
#include "feature_manager.h"

bool TrackedFeature::IsObservedInFrame(const int frame_id_)
{
    return (tracked_points.find(frame_id_) != tracked_points.end());
}

float TrackedFeature::GetParallaxToReference(const int frame_id)
{
    if(IsObservedInFrame(frame_id) && frame_id != reference_frame_id)
        return (tracked_points[frame_id] - reference_point).head<2>().norm();
    return 0;
}

float TrackedFeature::GetParallax(const int frame_id1, const int frame_id2)
{
    if(IsObservedInFrame(frame_id1) && IsObservedInFrame(frame_id2))
        return (tracked_points[frame_id1] - tracked_points[frame_id2]).head<2>().norm();
    return 0;
}

bool TrackedFeature::Triangulate(const Eigen::Matrix3d& R1w, const Eigen::Vector3d& t1w, const int frame_id1,
                                 const Eigen::Matrix3d& R2w, const Eigen::Vector3d& t2w, const int frame_id2)
{
    const Eigen::Vector2d point0 = reference_point.tail<2>();
    const Eigen::Vector2d point1 = tracked_points[frame_id2].tail<2>();
    Eigen::Matrix<double, 3, 4> Pose0, Pose1;
    Pose0.block<3, 3>(0, 0) = R1w;
    Pose0.block<3, 1>(0, 3) = t1w;
    Pose1.block<3, 3>(0, 0) = R2w;
    Pose1.block<3, 1>(0, 3) = t2w;

    Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
    design_matrix.row(0) = point0[0]*Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0[1]*Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1[0]*Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1[1]*Pose1.row(2) - Pose1.row(1);

    Eigen::Vector4d triangulated_point;
    triangulated_point =
              design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();

    position = triangulated_point.hnormalized();
    Eigen::Vector3d triangulated_point0 = Pose0*(triangulated_point/triangulated_point(3));
    float estmated_depth = triangulated_point0.z();
    if(estmated_depth < 0)
        return false;
    inv_dep = 1.0/estmated_depth;
    solve_flag = true;
    return true;
}


void FeatureManager::Clear()
{
    all_features.clear();
    unsolved_feature_ids.clear();
    solved_feature_ids.clear();
    all_frame_ids.clear();
}


void FeatureManager::AddFeatures(int frame_id, std::map<int, Eigen::Vector2d>& new_features)
{
    all_frame_ids.insert(frame_id);
    for(auto &f : new_features)
    {
        auto it = all_features.find(f.first);
        if(it != all_features.end())
        {
            it->second->tracked_points[frame_id].head<2>() = f.second;
            it->second->tracked_points[frame_id].tail<2>() = Pixel2Norm(f.second);
        }
        else
        {
            std::shared_ptr<TrackedFeature> feature = std::make_shared<TrackedFeature>(f.first, frame_id, f.second);
            all_features[f.first] = feature;
            unsolved_feature_ids.insert(f.first);
        }
    }
}


void FeatureManager::DeleteFeatures(int frame_id)
{
    if(all_frame_ids.find(frame_id) == all_frame_ids.end())
        return;

    const int size = all_features.size();
    int idx = 0;
    for(auto it = all_features.begin(), it_next = all_features.begin();
        idx < size; it = it_next, ++idx)
    {
        ++it_next;
        std::shared_ptr<TrackedFeature> feature_ptr = it->second;
        if(feature_ptr->reference_frame_id == frame_id)
        {
            if(feature_ptr->tracked_points.size() == 1)
            {
                if(feature_ptr->solve_flag)
                    solved_feature_ids.erase(it->first);
                else
                    unsolved_feature_ids.erase(it->first);
                feature_ptr->tracked_points.clear();
                all_features.erase(it->first);
            }
            else
            {
                feature_ptr->tracked_points.erase(feature_ptr->tracked_points.begin());
                feature_ptr->reference_frame_id = feature_ptr->tracked_points.begin()->first;
                feature_ptr->reference_point = feature_ptr->tracked_points.begin()->second;

                if(feature_ptr->solve_flag)
                {
                    feature_ptr->solve_flag = false;
                    solved_feature_ids.erase(it->first);
                    unsolved_feature_ids.insert(it->first);
                }
            }
        }
        else if(feature_ptr->IsObservedInFrame(frame_id))
        {
            feature_ptr->tracked_points.erase(frame_id);
            if(feature_ptr->tracked_points.size() == 1 && feature_ptr->solve_flag)
            {
                feature_ptr->solve_flag = false;
                solved_feature_ids.erase(it->first);
                unsolved_feature_ids.insert(it->first);
            }
        }
    }
    all_frame_ids.erase(frame_id);
}

void FeatureManager::DeleteFeatures(int frame_id, const std::vector<Eigen::Matrix3d>& vRcw, const std::vector<Eigen::Vector3d>& vtcw,
                                    const std::vector<int>& vframe_ids)
{
    if(all_frame_ids.find(frame_id) == all_frame_ids.end())
        return;

    const int size = all_features.size();
    int idx = 0;
    for(auto it = all_features.begin(), it_next = all_features.begin();
        idx < size; it = it_next, ++idx)
    {
        ++it_next;
        std::shared_ptr<TrackedFeature> feature_ptr = it->second;
        if(feature_ptr->reference_frame_id == frame_id)
        {
            if(feature_ptr->tracked_points.size() == 1)
            {
                if(feature_ptr->solve_flag)
                    solved_feature_ids.erase(it->first);
                else
                    unsolved_feature_ids.erase(it->first);
                feature_ptr->tracked_points.clear();
                all_features.erase(it->first);
            }
            else
            {
                int origin_ref_frame_id = feature_ptr->reference_frame_id;
                Eigen::Vector3d origin_ref_point = feature_ptr->reference_point.tail<2>().homogeneous();

                feature_ptr->tracked_points.erase(feature_ptr->tracked_points.begin());
                feature_ptr->reference_frame_id = feature_ptr->tracked_points.begin()->first;
                feature_ptr->reference_point = feature_ptr->tracked_points.begin()->second;

                if(feature_ptr->solve_flag)
                {
                    if(feature_ptr->tracked_points.size() < 3)
                    {
                        feature_ptr->solve_flag = false;
                        solved_feature_ids.erase(it->first);
                        unsolved_feature_ids.insert(it->first);
                    }
                    else
                    {
                        int new_ref_frame_id = feature_ptr->reference_frame_id;
                        double depth = 1.0/feature_ptr->inv_dep;
                        int id1 = std::distance(vframe_ids.begin(), std::find(vframe_ids.begin(), vframe_ids.end(), origin_ref_frame_id));
                        int id2 = std::distance(vframe_ids.begin(), std::find(vframe_ids.begin(), vframe_ids.end(), new_ref_frame_id));
                        Eigen::Vector3d new_ref_point = vRcw[id2]*vRcw[id1].transpose()*(origin_ref_point*depth - vtcw[id1]) + vtcw[id2];
                        if(new_ref_point.z() > 0)
                            feature_ptr->inv_dep = 1.0/new_ref_point.z();
                        else
                        {
                            solved_feature_ids.erase(it->first);
                            feature_ptr->tracked_points.clear();
                            all_features.erase(it->first);
                        }
                    }
                }
            }
        }
        else if(feature_ptr->IsObservedInFrame(frame_id))
        {
            feature_ptr->tracked_points.erase(frame_id);
            if(feature_ptr->tracked_points.size() < 3 && feature_ptr->solve_flag)
            {
                feature_ptr->solve_flag = false;
                solved_feature_ids.erase(it->first);
                unsolved_feature_ids.insert(it->first);
            }
        }
    }
    all_frame_ids.erase(frame_id);
}

std::vector<int> FeatureManager::GetCorresponding(int ref_frame_id, int dst_frame_id)
{
    std::vector<int> features_ids;
    for(auto &feature : all_features)
    {
        if(feature.second->IsObservedInFrame(ref_frame_id) &&
           feature.second->IsObservedInFrame(dst_frame_id))
        {
            features_ids.emplace_back(feature.first);
        }
    }
    return features_ids;
}

bool FeatureManager::StructureFromMotion(std::vector<int>& frame_ids, std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Vector3d>& T)
{
    const int frame_size = frame_ids.size();
    bool two_frame_recovery = false;
    Eigen::Matrix3d Rl0;
    Eigen::Vector3d tl0;
    R[0].setIdentity();
    T[0].setZero();

    std::vector<int> two_frame_features_ids;
    int l;

    for(int i = 1; i < frame_size; ++i)
    {
        two_frame_features_ids = GetCorresponding(frame_ids[0], frame_ids[i]);
        if(two_frame_features_ids.size() < 20) continue;

        float sum_parallax = 0.0f;
        float average_parallax;
        std::vector<cv::Point2f> ll, rr;
        for(auto f_id : two_frame_features_ids)
        {
            const auto &feature = all_features[f_id];
            sum_parallax += feature->GetParallaxToReference(frame_ids[i]);
            Eigen::Vector2d np1 = feature->reference_point.tail<2>();
            Eigen::Vector2d np2 = feature->tracked_points[frame_ids[i]].tail<2>();
            ll.push_back(cv::Point2f(np1.x(), np1.y()));
            rr.push_back(cv::Point2f(np2.x(), np2.y()));
        }
        average_parallax = sum_parallax / int(two_frame_features_ids.size());
        if(average_parallax < 10) continue;

        cv::Mat mask;
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);

        cv::Mat E = cv::findEssentialMat(ll, rr, cameraMatrix, cv::FM_RANSAC, 0.99, 0.3/fxG, mask);

        cv::Mat rot, trans;
        int inlier_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);
        if(inlier_cnt > 15)
        {
            cv::cv2eigen(rot, Rl0);
            cv::cv2eigen(trans, tl0);
            l = i;
            std::cout << "average_parallax:" << average_parallax << ", choose frame " << l << " to triangulate the whole structure." << std::endl;
            std::cout << "inliers count: " << inlier_cnt << std::endl;
            two_frame_recovery = true;
            break;
        }
    }

    if(!two_frame_recovery)
    {
        std::cout << "two frame initilization failed!" << std::endl;
        return false;
    }
    R[l] = Rl0;
    T[l] = tl0;

    for(auto f_id : two_frame_features_ids)
    {
       if(all_features[f_id]->GetParallaxToReference(frame_ids[l]) > triangulate_parallax)
       {
           if(all_features[f_id]->Triangulate(R[0], T[0], frame_ids[0], Rl0, tl0, frame_ids[l]))
           {
               solved_feature_ids.insert(f_id);
               unsolved_feature_ids.erase(f_id);
           }
       }
    }
    std::cout << "first solved point number: " << solved_feature_ids.size() << std::endl;

    for(int i = 1; i < l; ++i)
    {
        R[i] = R[i - 1];
        T[i] = T[i - 1];
        if(!SolvePnPNewFrame(frame_ids[i], R[i], T[i]))
            return false;
    }

    for(int i = l - 1; i > 0; --i)
    {
        TriangulateNewFrame(R, T, frame_ids, R[i], T[i], frame_ids[i]);
    }

    for(int i = l + 1; i < frame_size; ++i)
    {

        R[i] = R[i - 1];
        T[i] = T[i - 1];
        if(!SolvePnPNewFrame(frame_ids[i], R[i], T[i]))
            return false;

        TriangulateNewFrame(R, T, frame_ids, R[i], T[i], frame_ids[i]);
    }


    std::cout << "total solved points number: " << solved_feature_ids.size() << "/" << all_features.size() << std::endl;
    ceres::Problem problem;
    RotationMatrixParameterization* local_parameterization = new RotationMatrixParameterization;

    double c_rotation[frame_size][9];
    double c_translation[frame_size][3];
    for(int i = 0; i < frame_size; i++)
    {
        Eigen::Map<Eigen::Matrix3d> Ri(c_rotation[i]);
        Eigen::Map<Eigen::Vector3d> ti(c_translation[i]);
        Ri = R[i].transpose();
        ti = -Ri*T[i];
    }
    for(auto &f_id : solved_feature_ids)
    {
        auto &feature = all_features[f_id];
        Eigen::Vector2d pt_i = feature->reference_point.tail<2>();
        int frame_idx1 = std::distance(frame_ids.begin(), std::find(frame_ids.begin(), frame_ids.end(), feature->reference_frame_id));
        for(auto &tracked_point : feature->tracked_points)
        {
            if(tracked_point.first == feature->reference_frame_id)
                continue;
            int frame_idx2 = std::distance(frame_ids.begin(), std::find(frame_ids.begin(), frame_ids.end(), tracked_point.first));
            Eigen::Vector2d pt_j = tracked_point.second.tail<2>();
            ceres::CostFunction* cost_function = new StructureFromMotionFactor(pt_i, pt_j);
            problem.AddResidualBlock(cost_function, nullptr, c_rotation[frame_idx1], c_translation[frame_idx1],
                                     c_rotation[frame_idx2], c_translation[frame_idx2], &all_features[f_id]->MutableInvDep());
        }
    }
    problem.SetParameterBlockConstant(c_rotation[0]);
    problem.SetParameterBlockConstant(c_translation[0]);
    problem.SetParameterBlockConstant(c_translation[l]);
    for(int i = 0; i < frame_size; i++)
    {
        problem.SetParameterization(c_rotation[i], local_parameterization);
    }

    ceres::Solver::Options options;
    options.num_threads = 2;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.5;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if(summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
        std::cout << "vision only BA converge" << std::endl;
    }
    else
    {
        std::cout << "vision only BA not converge " << std::endl;
        return false;
    }

    for(int i = 0; i < frame_size; ++i)
    {
        Eigen::Map<Eigen::Matrix3d> Rwi(c_rotation[i]);
        Eigen::Map<Eigen::Vector3d> twi(c_translation[i]);
        R[i] = Rwi;
        T[i] = twi;
    }
    return true;
}

bool FeatureManager::SolvePnPNewFrame(const int& cur_frame_id, Eigen::Matrix3d& cur_Rcw, Eigen::Vector3d& cur_tcw)
{
    std::vector<cv::Point2f> pts_2_vector;
    std::vector<cv::Point3f> pts_3_vector;
    for(auto f_id : solved_feature_ids)
    {
        const auto &feature = all_features[f_id];
        if(feature->IsObservedInFrame(cur_frame_id))
        {
            Eigen::Vector2d pt_norm = feature->tracked_points[cur_frame_id].tail<2>();
            Eigen::Vector3d pt_3d = all_features[f_id]->position;
            pts_2_vector.emplace_back(pt_norm.x(), pt_norm.y());
            pts_3_vector.emplace_back(pt_3d.x(), pt_3d.y(), pt_3d.z());
        }
    }
    if(pts_2_vector.size() < 15)
    {
        std::cout << "Triangulate later frames failed" << std::endl;
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(cur_Rcw, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(cur_tcw, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, 1);
    if(!pnp_succ)
    {
        std::cout << "cv::solvePnP failed !!!" << std::endl;
        return false;
    }
    cv::Rodrigues(rvec, r);
    Eigen::Matrix3d R_pnp;
    cv::cv2eigen(r, R_pnp);
    Eigen::Vector3d T_pnp;
    cv::cv2eigen(t, T_pnp);
    cur_Rcw = R_pnp;
    cur_tcw = T_pnp;
    return true;
}


void FeatureManager::TriangulateNewFrame(const std::vector<Eigen::Matrix3d>& vRcw,
                                         const std::vector<Eigen::Vector3d>& vtcw,
                                         const std::vector<int>& vframe_ids,
                                         const Eigen::Matrix3d& cur_Rcw, const Eigen::Vector3d& cur_tcw,
                                         const int& cur_frame_id)
{
    for(auto it = unsolved_feature_ids.begin(); it != unsolved_feature_ids.end();)
    {
        auto &feature = all_features[*it];
        if(feature->GetParallaxToReference(cur_frame_id) > triangulate_parallax)
        {
            int start_idx = std::distance(vframe_ids.begin(), std::find(vframe_ids.begin(), vframe_ids.end(), feature->reference_frame_id));

            if(feature->Triangulate(vRcw[start_idx], vtcw[start_idx], feature->reference_frame_id, cur_Rcw, cur_tcw, cur_frame_id))
            {
                solved_feature_ids.insert(*it);
                it = unsolved_feature_ids.erase(it);
                continue;
            }
        }
        ++it;
    }
}


void FeatureManager::Triangulation(const std::vector<Eigen::Matrix3d>& vRcw,
                                   const std::vector<Eigen::Vector3d>& vtcw,
                                   const std::vector<int>& vframe_ids)
{
    const int frame_size = vframe_ids.size();
    for(auto it = unsolved_feature_ids.begin(); it != unsolved_feature_ids.end();)
    {
        const auto &feature = all_features[*it];
        int start_idx = std::distance(vframe_ids.begin(), std::find(vframe_ids.begin(), vframe_ids.end(), feature->reference_frame_id));
        std::vector<int> tracked_ids;
        float max_parallax = 0;
        for(int i = start_idx + 1; i < frame_size; ++i)
        {
            if(feature->IsObservedInFrame(vframe_ids[i]))
            {
                tracked_ids.emplace_back(i);
                float cur_parallax = feature->GetParallaxToReference(vframe_ids[i]);
                max_parallax = std::max(max_parallax, cur_parallax);
            }
        }

        if(max_parallax > triangulate_parallax &&
                tracked_ids.size() > 2)
        {
            Eigen::Matrix3d R0 = vRcw[start_idx].transpose();
            Eigen::Vector3d t0 = -R0*vtcw[start_idx];

            int svd_idx = 0;
            Eigen::MatrixXd svd_A(2*(tracked_ids.size() + 1), 4);

            Eigen::Matrix<double, 3, 4> P0;
            P0.block<3, 3>(0, 0).setIdentity();
            P0.block<3, 1>(0, 3).setZero();
            Eigen::Vector3d p_norm0 = feature->reference_point.tail<2>().homogeneous();
            Eigen::Vector3d f0 = p_norm0.normalized();
            svd_A.row(svd_idx++) = f0[0]*P0.row(2) - f0[2]*P0.row(0);
            svd_A.row(svd_idx++) = f0[1]*P0.row(2) - f0[2]*P0.row(1);

            for(auto &idx : tracked_ids)
            {
                Eigen::Matrix<double, 3, 4> P;
                P.leftCols(3) = vRcw[idx]*R0;
                P.rightCols(1) = vRcw[idx]*t0 + vtcw[idx];

                Eigen::Vector3d pun = feature->tracked_points[vframe_ids[idx]].tail<2>().homogeneous();
                Eigen::Vector3d f = pun.normalized();
                svd_A.row(svd_idx++) = f[0]*P.row(2) - f[2]*P.row(0);
                svd_A.row(svd_idx++) = f[1]*P.row(2) - f[2]*P.row(1);
            }
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double inverse_depth = svd_V[3] / svd_V[2];

            if(inverse_depth > 0 && inverse_depth < 1e2)
//            if(inverse_depth > 0.01 && inverse_depth < 1e2)
            {
                feature->solve_flag = true;
                feature->inv_dep = inverse_depth;
                solved_feature_ids.insert(*it);
                it = unsolved_feature_ids.erase(it);
                continue;
            }
        }
        ++it;
    }
}

Eigen::VectorXd FeatureManager::GetInvDepthVector()
{
    Eigen::VectorXd dep_vec(solved_feature_ids.size());
    int feature_index = 0;
    for(auto &feature_id : solved_feature_ids)
    {
        dep_vec(feature_index++) = all_features[feature_id]->inv_dep;
    }
    return dep_vec;
}

void FeatureManager::SetInvDepthVector(const Eigen::VectorXd &dep_vec)
{
    int feature_index = 0;
    for(auto &feature_id : solved_feature_ids)
    {
        all_features[feature_id]->inv_dep = dep_vec(feature_index++);
    }
}

void FeatureManager::ResetSolveFlag()
{
    solved_feature_ids.clear();
    for(auto& feature : all_features)
    {
        if(feature.second->solve_flag)
        {
            feature.second->solve_flag = false;
            unsolved_feature_ids.insert(feature.first);
        }
    }
}

void FeatureManager::ResetOutliers()
{
    for(auto it = solved_feature_ids.begin(); it != solved_feature_ids.end();)
    {
        if(all_features[*it]->inv_dep < 0 || all_features[*it]->inv_dep > 1e2)
//        if(all_features[*it]->inv_dep < 0.01 || all_features[*it]->inv_dep > 1e2)
        {
            all_features[*it]->solve_flag = false;
            unsolved_feature_ids.insert(*it);
            it = solved_feature_ids.erase(it);
            continue;
        }
        ++it;
    }
}

void FeatureManager::DeleteOutliers()
{
    for(auto it = solved_feature_ids.begin(); it != solved_feature_ids.end();)
    {
        if(all_features[*it]->inv_dep < 0 || all_features[*it]->inv_dep > 1e2)
//        if(all_features[*it]->inv_dep < 0.01 || all_features[*it]->inv_dep > 1e2)
        {
            all_features[*it]->tracked_points.clear();
            all_features.erase(*it);
            it = solved_feature_ids.erase(it);
            continue;
        }
        ++it;
    }
}
