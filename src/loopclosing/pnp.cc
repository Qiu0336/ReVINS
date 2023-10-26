
#include "pnp.h"

inline int NumRequiredIterations(const double inlier_ratio, const double prob_missing_best_model,
                                 const int sample_size, const int min_iterations, const int max_iterations)
{
    if(inlier_ratio <= 0.0)
        return max_iterations;
    if(inlier_ratio >= 1.0)
        return min_iterations;

    const double kProbNonInlierSample =
            1.0 - std::pow(inlier_ratio, static_cast<double>(sample_size));
    const double kLogDenominator = std::log(kProbNonInlierSample);
    const double kLogNumerator = std::log(prob_missing_best_model);

    double num_iters = std::ceil(kLogNumerator / kLogDenominator + 0.5);
    int num_req_iterations = std::min(static_cast<int>(num_iters), max_iterations);
    num_req_iterations = std::max(min_iterations, num_req_iterations);
    return num_req_iterations;
}

void PnPEstimator::SolvePnP(const std::vector<PnPMatch> &matches, Eigen::Matrix3d &Rcw, Eigen::Vector3d &tcw)
{
    Eigen::Matrix3d r = Rcw;
    Eigen::Vector3d t = tcw;
    int iterations = 10;
    double cost = 0, lastCost = 0;
    for(int iter = 0; iter < iterations; iter++)
    {
        Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Vector6d b = Eigen::Vector6d::Zero();

        cost = 0;
        for(int i = 0, iend = matches.size(); i < iend; ++i)
        {
            const PnPMatch m = matches[i];
            Eigen::Vector3d pc = m.Rc_ref*(r*m.point3d + t) + m.tc_ref;
            double inv_z = 1.0 / pc.z();
            double inv_z2 = inv_z * inv_z;
            Eigen::Vector2d e = pc.hnormalized() - m.point2d_norm;

            cost += e.squaredNorm();

            Eigen::Matrix<double, 2, 3> JeP;
            JeP << inv_z, 0, - pc.x()*inv_z2,
                   0, inv_z, - pc.y()*inv_z2;
            Eigen::Matrix<double, 2, 6> J;
            J.leftCols(3) = - JeP*m.Rc_ref*r*Skew(m.point3d);
            J.rightCols(3) = JeP*m.Rc_ref;

            H += J.transpose() * J;
            b += -J.transpose() * e;
        }

        Eigen::Vector6d dx;
        dx = H.ldlt().solve(b);

        if(isnan(dx[0]))
        {
            break;
        }

        if(iter > 0 && cost >= lastCost)
        {
            break;
        }

        // update your estimation
        r = r*ExpSO3(dx.head<3>());
        t = t + dx.tail<3>();
        lastCost = cost;

        if(dx.norm() < 1e-6) // converge
        {
            break;
        }
    }
    Rcw = r;
    tcw = t;
}


int PnPEstimator::Evaluate(const std::vector<PnPMatch> &matches,
                           const Eigen::Matrix3d &Rcw, const Eigen::Vector3d &tcw, std::vector<bool> &inliners)
{
    int inliners_num = 0;
    inliners.resize(matches.size());
    for(int i = 0, iend = matches.size(); i < iend; i++)
    {
        const PnPMatch m = matches[i];
        Eigen::Vector3d pc = m.Rc_ref*(Rcw*m.point3d + tcw) + m.tc_ref;
        Eigen::Vector2d e = pc.hnormalized() - m.point2d_norm;
        const double err = e.norm();
        if(err < max_error)
        {
            inliners_num++;
            inliners[i] = true;
        }
        else
            inliners[i] = false;
    }
    return inliners_num;
}

bool PnPEstimator::PnPRansac(const std::vector<PnPMatch> &matches,
                             Eigen::Isometry3d &Tcw, std::vector<bool> &inliners)
{
    const int num_samples = matches.size();

    int res_inliers_num = 0;
    Eigen::Vector3d res_tcw;
    Eigen::Matrix3d res_Rcw;
    std::vector<bool> res_inliner_mask(num_samples, false);

    std::vector<bool> inlier_mask(num_samples);
    int required_iteration = max_iterations;
    std::vector<PnPMatch> res_matches;
    for(int i = 0; i < required_iteration; ++i)
    {
        // 1. Select sample
        std::vector<int> minimal_sample;
        random_sampler.ShuffleSample(num_samples, sample_size, minimal_sample);

        std::vector<PnPMatch> select_matches(sample_size);
        for(int j = 0; j < sample_size; ++j)
        {
            select_matches[j] = matches[minimal_sample[j]];
        }

        // 2. Estimate model
        Eigen::Matrix3d R_ = Tcw.rotation();
        Eigen::Vector3d t_ = Tcw.translation();

        SolvePnP(select_matches, R_, t_);

        // 3. Iterate through all estimated models.
        int inliers_num = Evaluate(matches, R_, t_, inlier_mask);

        // Save as best subset if better than all previous subsets.
        if(inliers_num > res_inliers_num)
        {
            res_inliner_mask = inlier_mask;
            res_tcw = t_;
            res_Rcw = R_;
            res_inliers_num = inliers_num;
            res_matches = select_matches;
        }

        // 4. update required iteration num
        double inlier_ratio = static_cast<double>(res_inliers_num) / static_cast<double>(num_samples);
        required_iteration = NumRequiredIterations(inlier_ratio, confidence, sample_size,
                                                   min_iterations, required_iteration);

        if(i == required_iteration)
            return false;
    }

    double inlier_ratio = static_cast<double>(res_inliers_num) / static_cast<double>(num_samples);

    std::cout << "inlier_ratio: " << inlier_ratio << ", inliner_size: " << res_inliers_num << ", required_iter: " << required_iteration << std::endl;

    if(inlier_ratio < min_inlier_ratio)
        return false;

    Eigen::Matrix3d R_ = Tcw.rotation();
    Eigen::Vector3d t_ = Tcw.translation();
    SolvePnP(res_matches, R_, t_);
    Tcw.linear() = res_Rcw;
    Tcw.translation() = res_tcw;
    inliners = res_inliner_mask;
    return true;
}




void CenterAndNormalizeImagePoints(const std::vector<Eigen::Vector2d> &points,
                                   std::vector<Eigen::Vector2d> *normed_points, Eigen::Matrix3d *matrix)
{
    Eigen::Vector2d centroid(0, 0);
    for(const Eigen::Vector2d &point : points)
    {
        centroid += point;
    }
    centroid /= points.size();

    // Root mean square error to centroid of all points
    double rms_mean_dist = 0;
    for(const Eigen::Vector2d &point : points)
    {
        rms_mean_dist += (point - centroid).squaredNorm();
    }
    rms_mean_dist = std::sqrt(rms_mean_dist / points.size());

    // Compose normalization matrix
    const double norm_factor = std::sqrt(2.0) / rms_mean_dist;
    *matrix << norm_factor, 0, -norm_factor * centroid(0), 0, norm_factor, -norm_factor * centroid(1), 0, 0, 1;

    // Apply normalization matrix
    normed_points->resize(points.size());

    const double M_00 = (*matrix)(0, 0);
    const double M_01 = (*matrix)(0, 1);
    const double M_02 = (*matrix)(0, 2);
    const double M_10 = (*matrix)(1, 0);
    const double M_11 = (*matrix)(1, 1);
    const double M_12 = (*matrix)(1, 2);
    const double M_20 = (*matrix)(2, 0);
    const double M_21 = (*matrix)(2, 1);
    const double M_22 = (*matrix)(2, 2);

    for(size_t i = 0; i < points.size(); ++i)
    {
        const double p_0 = points[i](0);
        const double p_1 = points[i](1);

        const double np_0 = M_00 * p_0 + M_01 * p_1 + M_02;
        const double np_1 = M_10 * p_0 + M_11 * p_1 + M_12;
        const double np_2 = M_20 * p_0 + M_21 * p_1 + M_22;

        const double inv_np_2 = 1.0 / np_2;
        (*normed_points)[i](0) = np_0 * inv_np_2;
        (*normed_points)[i](1) = np_1 * inv_np_2;
    }
}

void FundamentalEstimator::SolveFundamental(const std::vector<Eigen::Vector2d> &points1, const std::vector<Eigen::Vector2d> &points2,
                                            Eigen::Matrix3d &F_matrix)
{
    // Center and normalize image points for better numerical stability.
    std::vector<Eigen::Vector2d> normed_points1;
    std::vector<Eigen::Vector2d> normed_points2;
    Eigen::Matrix3d points1_norm_matrix;
    Eigen::Matrix3d points2_norm_matrix;
    CenterAndNormalizeImagePoints(points1, &normed_points1, &points1_norm_matrix);
    CenterAndNormalizeImagePoints(points2, &normed_points2, &points2_norm_matrix);

    // Setup homogeneous linear equation as x2' * F * x1 = 0.
    Eigen::Matrix<double, Eigen::Dynamic, 9> cmatrix(points1.size(), 9);
    for(size_t i = 0; i < points1.size(); ++i)
    {
        cmatrix.block<1, 3>(i, 0) = normed_points1[i].homogeneous();
        cmatrix.block<1, 3>(i, 0) *= normed_points2[i].x();
        cmatrix.block<1, 3>(i, 3) = normed_points1[i].homogeneous();
        cmatrix.block<1, 3>(i, 3) *= normed_points2[i].y();
        cmatrix.block<1, 3>(i, 6) = normed_points1[i].homogeneous();
    }

    // Solve for the nullspace of the constraint matrix.
    Eigen::JacobiSVD<Eigen::Matrix<double, Eigen::Dynamic, 9>> cmatrix_svd(cmatrix, Eigen::ComputeFullV);
    const Eigen::VectorXd cmatrix_nullspace = cmatrix_svd.matrixV().col(8);
    const Eigen::Map<const Eigen::Matrix3d> ematrix_t(cmatrix_nullspace.data());

    // Enforcing the internal constraint that two singular values must non-zero
    // and one must be zero.
    Eigen::JacobiSVD<Eigen::Matrix3d> fmatrix_svd(ematrix_t.transpose(), Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d singular_values = fmatrix_svd.singularValues();
    singular_values(2) = 0.0;
    const Eigen::Matrix3d F = fmatrix_svd.matrixU() * singular_values.asDiagonal() * fmatrix_svd.matrixV().transpose();

    F_matrix = points2_norm_matrix.transpose()*F*points1_norm_matrix;
}

int FundamentalEstimator::Evaluate(const std::vector<Eigen::Vector2d> &points1,
                                   const std::vector<Eigen::Vector2d> &points2,
                                   const Eigen::Matrix3d &F, std::vector<bool> &inliners)
{
    int inliners_num = 0;
    inliners.resize(points1.size());
    // Note that this code might not be as nice as Eigen expressions,
    // but it is significantly faster in various tests

    const double F_00 = F(0, 0);
    const double F_01 = F(0, 1);
    const double F_02 = F(0, 2);
    const double F_10 = F(1, 0);
    const double F_11 = F(1, 1);
    const double F_12 = F(1, 2);
    const double F_20 = F(2, 0);
    const double F_21 = F(2, 1);
    const double F_22 = F(2, 2);

    for(size_t i = 0; i < points1.size(); ++i)
    {
        const double x1_0 = points1[i](0);
        const double x1_1 = points1[i](1);
        const double x2_0 = points2[i](0);
        const double x2_1 = points2[i](1);

        // Ex1 = E * points1[i].homogeneous();
        const double Fx1_0 = F_00 * x1_0 + F_01 * x1_1 + F_02;
        const double Fx1_1 = F_10 * x1_0 + F_11 * x1_1 + F_12;
        const double Fx1_2 = F_20 * x1_0 + F_21 * x1_1 + F_22;

        // Etx2 = E.transpose() * points2[i].homogeneous();
        const double Ftx2_0 = F_00 * x2_0 + F_10 * x2_1 + F_20;
        const double Ftx2_1 = F_01 * x2_0 + F_11 * x2_1 + F_21;

        // x2tEx1 = points2[i].homogeneous().transpose() * Ex1;
        const double x2tEx1 = x2_0 * Fx1_0 + x2_1 * Fx1_1 + Fx1_2;

        // Sampson distance
        const double err = x2tEx1 * x2tEx1 / (Fx1_0 * Fx1_0 + Fx1_1 * Fx1_1 + Ftx2_0 * Ftx2_0 + Ftx2_1 * Ftx2_1);

        if(err < max_error)
        {
            inliners_num++;
            inliners[i] = true;
        }
        else
            inliners[i] = false;
    }
    return inliners_num;
}

bool FundamentalEstimator::FundamentalRansac(const std::vector<Eigen::Vector2d> &points1,
                                             const std::vector<Eigen::Vector2d> &points2, std::vector<bool> &inliners)
{
    assert(points1.size() == points2.size());
    const int num_samples = points1.size();

    sample_size = 8;

    int res_inliers_num = 0;
    Eigen::Matrix3d res_F;
    std::vector<bool> res_inliner_mask(num_samples, false);

    std::vector<bool> inlier_mask(num_samples);
    int required_iteration = max_iterations;
    for(int i = 0; i < required_iteration; ++i)
    {
        // 1. Select sample
        std::vector<int> minimal_sample;
        random_sampler.ShuffleSample(num_samples, sample_size, minimal_sample);

        std::vector<Eigen::Vector2d> select_points1(sample_size);
        std::vector<Eigen::Vector2d> select_points2(sample_size);
        for(int j = 0; j < sample_size; ++j)
        {
            select_points1[j] = points1[minimal_sample[j]];
            select_points2[j] = points2[minimal_sample[j]];
        }

        // 2. Estimate model
        Eigen::Matrix3d F_;
        SolveFundamental(select_points1, select_points2, F_);

        // 3. Iterate through all estimated models.
        int inliers_num = Evaluate(points1, points2, F_, inlier_mask);

        // Save as best subset if better than all previous subsets.
        if(inliers_num > res_inliers_num)
        {
            res_inliner_mask = inlier_mask;
            res_inliers_num = inliers_num;
            res_F = F_;
        }

        // 4. update required iteration num
        double inlier_ratio = static_cast<double>(res_inliers_num) / static_cast<double>(num_samples);
        required_iteration = NumRequiredIterations(inlier_ratio, confidence, sample_size,
                                                   min_iterations, required_iteration);

        if(i == required_iteration)
            return false;
    }

    double inlier_ratio = static_cast<double>(res_inliers_num) / static_cast<double>(num_samples);

    inliners = res_inliner_mask;

    if(inlier_ratio < min_inlier_ratio)
        return false;

    return true;
}

