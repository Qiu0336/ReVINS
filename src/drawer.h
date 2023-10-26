
#pragma once

#include <pangolin/pangolin.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include "optimization/estimator.h"
#include "loopclosing/pose_graph.h"

class Drawer
{
    public:
    Drawer();
    void DrawBackground();
    void DrawCamera(bool use_loop);
    void DrawMapPoints();
    void DrawViewingVoxels();
    void DrawViewingPoints();
    void DrawTrajectory(bool use_loop);

    void Run();
    void SetQuit(bool x);
    bool GetQuit();

    float mBackgroundpatchsize;
    int mBackgroundpatchcount;
    float mPointSize;
    float mLineSize;
    float mCameraSize;
    float mCameraLineWidth;

    std::shared_ptr<Estimator> estimator_ptr;
    std::shared_ptr<PoseGraph> posegraph_ptr;

    std::mutex m_quit;
    bool quit_flag;
};

