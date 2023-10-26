#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include "camera_model.h"

namespace CameraModel
{

class PinholeCamera: public Camera
{
public:
    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();
        Parameters(int w, int h,
                   double k1, double k2, double p1, double p2,
                   double fx, double fy, double cx, double cy);

        double& k1(void);
        double& k2(void);
        double& p1(void);
        double& p2(void);
        double& fx(void);
        double& fy(void);
        double& cx(void);
        double& cy(void);

        double xi(void) const;
        double k1(void) const;
        double k2(void) const;
        double p1(void) const;
        double p2(void) const;
        double fx(void) const;
        double fy(void) const;
        double cx(void) const;
        double cy(void) const;

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<< (std::ostream& out, const Parameters& params);

    private:
        double m_k1;
        double m_k2;
        double m_p1;
        double m_p2;
        double m_fx;
        double m_fy;
        double m_cx;
        double m_cy;
    };

    PinholeCamera();
    PinholeCamera(int imageWidth, int imageHeight,
                  double k1, double k2, double p1, double p2,
                  double fx, double fy, double cx, double cy);
    PinholeCamera(const Parameters& params);

    Camera::ModelType modelType(void) const;
    int imageWidth(void) const;
    int imageHeight(void) const;

    void pixel2norm(const Eigen::Vector2d& pixel, Eigen::Vector2d& norm) const;
    void pixel2hnorm(const Eigen::Vector2d& pixel, Eigen::Vector3d& hnorm) const;

    void norm2pixel(const Eigen::Vector2d& norm, Eigen::Vector2d& pixel, bool distort = true) const;
    void mappoint2pixel(const Eigen::Vector3d& mappoint, Eigen::Vector2d& pixel, bool distort = true) const;

    void distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const;

    int parameterCount(void) const;

    const Parameters& getParameters(void) const;
    void setParameters(const Parameters& parameters);

    void readParameters(const std::vector<double>& parameterVec);
    void writeParameters(std::vector<double>& parameterVec) const;

    void writeParametersToYamlFile(const std::string& filename) const;

    std::string parametersToString(void) const;

private:
    Parameters mParameters;

    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
    bool m_noDistortion;
};

typedef boost::shared_ptr<PinholeCamera> PinholeCameraPtr;
typedef boost::shared_ptr<const PinholeCamera> PinholeCameraConstPtr;

}
