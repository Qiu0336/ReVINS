#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>

class _ExtractorNode {
public:
    _ExtractorNode() : bNoMore(false) {}

    void DivideNode(_ExtractorNode &n1, _ExtractorNode &n2, _ExtractorNode &n3, _ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<_ExtractorNode>::iterator lit;
    bool bNoMore;
};

class ORB_Extractor {
public:
    ORB_Extractor(int n_features, float scale_factor, int n_levels, int initial_threshold_fast, int min_threshold_fast);

    void Detect(cv::InputArray _image, std::vector<cv::KeyPoint> &_keypoints, cv::OutputArray _descriptors);

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> &allKeypoints);
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint> &vToDistributeKeys,
                                           const int &minX, const int &maxX, const int &minY,
                                           const int &maxY, const int &N);

    std::vector<cv::Mat> mvImagePyramid;

    std::vector<cv::Point> pattern;
    int n_features;
    double scale_factor;
    int n_levels;
    int initial_threshold_fast;
    int min_threshold_fast;
    std::vector<int> mn_featuresPerLevel;

    std::vector<int> umax;

    std::vector<float> mvscale_factor;
    std::vector<float> mvInvscale_factor;

};


namespace cv
{
void detectAndCompute( InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors);
}
