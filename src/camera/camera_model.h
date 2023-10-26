#pragma once

#include <boost/shared_ptr.hpp>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>

namespace CameraModel
{

class Camera
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum ModelType
    {
        PINHOLE,
        KANNALA_BRANDT
    };

    class Parameters
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Parameters(ModelType modelType);

        Parameters(ModelType modelType,
                   int w, int h);

        ModelType& modelType(void);
        int& imageWidth(void);
        int& imageHeight(void);

        ModelType modelType(void) const;
        int imageWidth(void) const;
        int imageHeight(void) const;

        int nIntrinsics(void) const;

        virtual bool readFromYamlFile(const std::string& filename) = 0;
        virtual void writeToYamlFile(const std::string& filename) const = 0;

    protected:
        ModelType m_modelType;
        int m_nIntrinsics;
        int m_imageWidth;
        int m_imageHeight;
    };

    virtual ModelType modelType(void) const = 0;
    virtual int imageWidth(void) const = 0;
    virtual int imageHeight(void) const = 0;

    virtual cv::Mat& mask(void);
    virtual const cv::Mat& mask(void) const;

    virtual void pixel2norm(const Eigen::Vector2d& pixel, Eigen::Vector2d& norm) const = 0;
    virtual void pixel2hnorm(const Eigen::Vector2d& pixel, Eigen::Vector3d& hnorm) const = 0;

    virtual void norm2pixel(const Eigen::Vector2d& norm, Eigen::Vector2d& pixel, bool distort = true) const = 0;
    virtual void mappoint2pixel(const Eigen::Vector3d& mappoint, Eigen::Vector2d& pixel, bool distort = true) const = 0;

    virtual void distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const = 0;

    virtual int parameterCount(void) const = 0;

    virtual void readParameters(const std::vector<double>& parameters) = 0;
    virtual void writeParameters(std::vector<double>& parameters) const = 0;

    virtual void writeParametersToYamlFile(const std::string& filename) const = 0;

    virtual std::string parametersToString(void) const = 0;
protected:
    cv::Mat m_mask;
};

typedef boost::shared_ptr<Camera> CameraPtr;
typedef boost::shared_ptr<const Camera> CameraConstPtr;

}
