#pragma once

#include <iostream>
#include <fstream>
#include <ctime>
#include <sstream>
#include <unistd.h>
#include <list>
#include <algorithm>
#include <vector>
#include <memory>
#include <numeric>
#include <map>
#include <set>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>

#include "preprocessor/imu_processor.h"
#include "camera/camera.h"
#include "ceres_factor.h"
#include "utils/io.h"
#include "utils/lie_group.h"
#include "settings.h"

class StructureFromMotionFactor : public ceres::SizedCostFunction<2, 9, 3, 9, 3, 1>
{
    public:
    StructureFromMotionFactor(const Eigen::Vector2d &pt_i_,
                              const Eigen::Vector2d &pt_j_) : pt_i(pt_i_), pt_j(pt_j_){}
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
    {
        Eigen::Map<const Eigen::Matrix3d> Rwi(parameters[0]);
        Eigen::Map<const Eigen::Vector3d> twi(parameters[1]);
        Eigen::Map<const Eigen::Matrix3d> Rwj(parameters[2]);
        Eigen::Map<const Eigen::Vector3d> twj(parameters[3]);
        const double invdep_i = parameters[4][0];
        const Eigen::Vector3d Pci = pt_i.homogeneous()/invdep_i;
        Eigen::Vector3d Pcj = Rwj.transpose()*(Rwi*Pci + twi - twj);
        const double dep = Pcj(2);
        Eigen::Map<Eigen::Vector2d> residual(residuals);
        residual = Pcj.hnormalized() - pt_j;

        if (jacobians)
        {
            Eigen::Matrix<double, 2, 3> reduce(2, 3);
            reduce << 1.0/dep,       0, - Pcj(0)/(dep*dep),
                            0, 1.0/dep, - Pcj(1)/(dep*dep);
            if (jacobians[0])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[0]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = - Rwj.transpose()*Rwi*Skew(Pci);
                J.leftCols(3) = reduce*jaco;
                J.rightCols(6).setZero();
            }
            if (jacobians[1])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[1]);
                J = reduce*Rwj.transpose();
            }
            if (jacobians[2])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 9, Eigen::RowMajor>> J(jacobians[2]);
                Eigen::Matrix<double, 3, 3> jaco;
                jaco = Skew(Pcj);
                J.leftCols(3) = reduce*jaco;
                J.rightCols(6).setZero();
            }
            if (jacobians[3])
            {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J(jacobians[3]);
                J = -reduce*Rwj.transpose();
            }
            if (jacobians[4])
            {
                Eigen::Map<Eigen::Vector2d> J(jacobians[4]);
                J = -reduce*Rwj.transpose()*Rwi*Pci/invdep_i;
            }
        }
        return true;
    }
    Eigen::Vector2d pt_i;
    Eigen::Vector2d pt_j;
};

class TrackedFeature
{
public:
    TrackedFeature(){}
    TrackedFeature(int feature_id_, int frame_id_, Eigen::Vector2d& point_):
        feature_id(feature_id_), reference_frame_id(frame_id_),
        solve_flag(false)
    {
        reference_point.head<2>() = point_;
        reference_point.tail<2>() = Pixel2Norm(point_);
        tracked_points[frame_id_] = reference_point;
    }
    bool IsObservedInFrame(const int frame_id_);

    float GetParallaxToReference(const int frame_id);
    float GetParallax(const int frame_id1, const int frame_id2);
    bool Triangulate(const Eigen::Matrix3d& R1w, const Eigen::Vector3d& t1w, const int frame_id1,
                     const Eigen::Matrix3d& R2w, const Eigen::Vector3d& t2w, const int frame_id2);
    double &MutableInvDep(){return inv_dep;}
public:
    int feature_id;
    int reference_frame_id;
    Eigen::Vector4d reference_point;
    std::map<int, Eigen::Vector4d> tracked_points;
    bool solve_flag;
    double inv_dep;
    Eigen::Vector3d position;
};


class FeatureManager
{
public:
    FeatureManager(){}

    void Clear();
    void AddFeatures(int frame_id, std::map<int, Eigen::Vector2d>& new_features);
    void DeleteFeatures(int frame_id);
    void DeleteFeatures(int frame_id, const std::vector<Eigen::Matrix3d>& vRcw, const std::vector<Eigen::Vector3d>& vtcw,
                        const std::vector<int>& vframe_ids);
    bool StructureFromMotion(std::vector<int>& frame_ids, std::vector<Eigen::Matrix3d>& R, std::vector<Eigen::Vector3d>& T);
    void ResetSolveFlag();
    std::vector<int> GetCorresponding(int ref_frame_id, int dst_frame_id);
    bool SolvePnPNewFrame(const int& cur_frame_id, Eigen::Matrix3d& cur_Rcw, Eigen::Vector3d& cur_tcw);
    void TriangulateNewFrame(const std::vector<Eigen::Matrix3d>& vRcw, const std::vector<Eigen::Vector3d>& vtcw,
                             const std::vector<int>& vframe_ids, const Eigen::Matrix3d& cur_Rcw,
                             const Eigen::Vector3d& cur_tcw, const int& cur_frame_id);
    void Triangulation(const std::vector<Eigen::Matrix3d>& vRcw, const std::vector<Eigen::Vector3d>& vtcw,
                       const std::vector<int>& vframe_ids);
    Eigen::VectorXd GetInvDepthVector();
    void SetInvDepthVector(const Eigen::VectorXd &dep_vec);
    void ResetOutliers();
    void DeleteOutliers();

public:
    std::map<int, std::shared_ptr<TrackedFeature>> all_features;
    std::set<int> unsolved_feature_ids;
    std::set<int> solved_feature_ids;
    std::set<int> all_frame_ids;

};

