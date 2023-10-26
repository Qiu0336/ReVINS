#pragma once

#include <opencv2/core/core.hpp>
#include <string>
#include "camera_model.h"

namespace CameraModel
{
/**
 * J. Kannala, and S. Brandt, A Generic Camera Model and Calibration Method
 * for Conventional, Wide-Angle, and Fish-Eye Lenses, PAMI 2006
 */

class EquidistantCamera: public Camera
{
public:
    class Parameters: public Camera::Parameters
    {
    public:
        Parameters();
        Parameters(int w, int h,
                   double k2, double k3, double k4, double k5,
                   double mu, double mv,
                   double u0, double v0);

        double& k2(void);
        double& k3(void);
        double& k4(void);
        double& k5(void);
        double& mu(void);
        double& mv(void);
        double& u0(void);
        double& v0(void);

        double k2(void) const;
        double k3(void) const;
        double k4(void) const;
        double k5(void) const;
        double mu(void) const;
        double mv(void) const;
        double u0(void) const;
        double v0(void) const;

        bool readFromYamlFile(const std::string& filename);
        void writeToYamlFile(const std::string& filename) const;

        Parameters& operator=(const Parameters& other);
        friend std::ostream& operator<< (std::ostream& out, const Parameters& params);

    private:
        // projection
        double m_k2;
        double m_k3;
        double m_k4;
        double m_k5;

        double m_mu;
        double m_mv;
        double m_u0;
        double m_v0;
    };

    EquidistantCamera();
    EquidistantCamera(int imageWidth, int imageHeight,
                      double k2, double k3, double k4, double k5,
                      double mu, double mv,
                      double u0, double v0);
    EquidistantCamera(const Parameters& params);

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

    void backprojectSymmetric(const Eigen::Vector2d& p_u,
                              double& theta, double& phi) const;

    Parameters mParameters;

    double m_inv_K11, m_inv_K13, m_inv_K22, m_inv_K23;
};

typedef boost::shared_ptr<EquidistantCamera> EquidistantCameraPtr;
typedef boost::shared_ptr<const EquidistantCamera> EquidistantCameraConstPtr;

}
