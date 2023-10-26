#include "equidistant_camera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

//#include "gpl.h"

namespace CameraModel
{

EquidistantCamera::Parameters::Parameters()
 : Camera::Parameters(KANNALA_BRANDT)
 , m_k2(0.0)
 , m_k3(0.0)
 , m_k4(0.0)
 , m_k5(0.0)
 , m_mu(0.0)
 , m_mv(0.0)
 , m_u0(0.0)
 , m_v0(0.0)
{

}

EquidistantCamera::Parameters::Parameters(int w, int h,
                                          double k2, double k3, double k4, double k5,
                                          double mu, double mv,
                                          double u0, double v0)
 : Camera::Parameters(KANNALA_BRANDT, w, h)
 , m_k2(k2)
 , m_k3(k3)
 , m_k4(k4)
 , m_k5(k5)
 , m_mu(mu)
 , m_mv(mv)
 , m_u0(u0)
 , m_v0(v0)
{

}

double&
EquidistantCamera::Parameters::k2(void)
{
    return m_k2;
}

double&
EquidistantCamera::Parameters::k3(void)
{
    return m_k3;
}

double&
EquidistantCamera::Parameters::k4(void)
{
    return m_k4;
}

double&
EquidistantCamera::Parameters::k5(void)
{
    return m_k5;
}

double&
EquidistantCamera::Parameters::mu(void)
{
    return m_mu;
}

double&
EquidistantCamera::Parameters::mv(void)
{
    return m_mv;
}

double&
EquidistantCamera::Parameters::u0(void)
{
    return m_u0;
}

double&
EquidistantCamera::Parameters::v0(void)
{
    return m_v0;
}

double
EquidistantCamera::Parameters::k2(void) const
{
    return m_k2;
}

double
EquidistantCamera::Parameters::k3(void) const
{
    return m_k3;
}

double
EquidistantCamera::Parameters::k4(void) const
{
    return m_k4;
}

double
EquidistantCamera::Parameters::k5(void) const
{
    return m_k5;
}

double
EquidistantCamera::Parameters::mu(void) const
{
    return m_mu;
}

double
EquidistantCamera::Parameters::mv(void) const
{
    return m_mv;
}

double
EquidistantCamera::Parameters::u0(void) const
{
    return m_u0;
}

double
EquidistantCamera::Parameters::v0(void) const
{
    return m_v0;
}

bool
EquidistantCamera::Parameters::readFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if (!fs.isOpened())
    {
        return false;
    }

    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (sModelType.compare("KANNALA_BRANDT") != 0)
        {
            return false;
        }
    }

    m_modelType = KANNALA_BRANDT;

    std::vector<int> resolution;
    std::vector<double> intrinsic;
    std::vector<double> dist_coef;
    fs["resolution"] >> resolution;
    fs["intrinsics"] >> intrinsic;
    fs["distortion_coefficients"] >> dist_coef;

    m_imageWidth = resolution[0];
    m_imageHeight = resolution[1];
    m_mu = intrinsic[0];
    m_mv = intrinsic[1];
    m_u0 = intrinsic[2];
    m_v0 = intrinsic[3];
    m_k2 = dist_coef[0];
    m_k3 = dist_coef[1];
    m_k4 = dist_coef[2];
    m_k5 = dist_coef[3];
    return true;
}

void
EquidistantCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "KANNALA_BRANDT";
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    // projection: k2, k3, k4, k5, mu, mv, u0, v0
    fs << "projection_parameters";
    fs << "{" << "k2" << m_k2
              << "k3" << m_k3
              << "k4" << m_k4
              << "k5" << m_k5
              << "mu" << m_mu
              << "mv" << m_mv
              << "u0" << m_u0
              << "v0" << m_v0 << "}";

    fs.release();
}

EquidistantCamera::Parameters&
EquidistantCamera::Parameters::operator=(const EquidistantCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_k2 = other.m_k2;
        m_k3 = other.m_k3;
        m_k4 = other.m_k4;
        m_k5 = other.m_k5;
        m_mu = other.m_mu;
        m_mv = other.m_mv;
        m_u0 = other.m_u0;
        m_v0 = other.m_v0;
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const EquidistantCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "KANNALA_BRANDT" << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;

    // projection: k2, k3, k4, k5, mu, mv, u0, v0
    out << "Projection Parameters" << std::endl;
    out << "            k2 " << params.m_k2 << std::endl
        << "            k3 " << params.m_k3 << std::endl
        << "            k4 " << params.m_k4 << std::endl
        << "            k5 " << params.m_k5 << std::endl
        << "            mu " << params.m_mu << std::endl
        << "            mv " << params.m_mv << std::endl
        << "            u0 " << params.m_u0 << std::endl
        << "            v0 " << params.m_v0 << std::endl;

    return out;
}

EquidistantCamera::EquidistantCamera()
 : m_inv_K11(1.0)
 , m_inv_K13(0.0)
 , m_inv_K22(1.0)
 , m_inv_K23(0.0)
{

}

EquidistantCamera::EquidistantCamera(int imageWidth, int imageHeight,
                                     double k2, double k3, double k4, double k5,
                                     double mu, double mv,
                                     double u0, double v0)
 : mParameters(imageWidth, imageHeight,
               k2, k3, k4, k5, mu, mv, u0, v0)
{
    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

EquidistantCamera::EquidistantCamera(const EquidistantCamera::Parameters& params)
 : mParameters(params)
{
    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

Camera::ModelType
EquidistantCamera::modelType(void) const
{
    return mParameters.modelType();
}

int
EquidistantCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
EquidistantCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}

void EquidistantCamera::pixel2norm(const Eigen::Vector2d& pixel, Eigen::Vector2d& norm) const
{
    // Lift points to normalised plane
    Eigen::Vector2d p_u;
    p_u << m_inv_K11 * pixel(0) + m_inv_K13,
           m_inv_K22 * pixel(1) + m_inv_K23;

    // Obtain a projective ray
    double theta, phi;
    backprojectSymmetric(p_u, theta, phi);

    double p_x = sin(theta) * cos(phi);
    double p_y = sin(theta) * sin(phi);
    double p_z = cos(theta);

    norm(0) = p_x/p_z;
    norm(1) = p_y/p_z;
}
void EquidistantCamera::pixel2hnorm(const Eigen::Vector2d& pixel, Eigen::Vector3d& hnorm) const
{
    Eigen::Vector2d norm;
    pixel2norm(pixel, norm);
    hnorm = norm.homogeneous();
}

void EquidistantCamera::norm2pixel(const Eigen::Vector2d& norm, Eigen::Vector2d& pixel, bool distort) const
{
    Eigen::Vector2d p_u;
    if(!distort)
    {
        p_u = norm;
    }
    else
    {
        distortpoint(norm, p_u);
    }
    // Apply generalised projection matrix
    pixel << mParameters.mu() * p_u(0) + mParameters.u0(),
             mParameters.mv() * p_u(1) + mParameters.v0();
}

void EquidistantCamera::mappoint2pixel(const Eigen::Vector3d& mappoint, Eigen::Vector2d& pixel, bool distort) const
{
    const Eigen::Vector2d norm = mappoint.hnormalized();
    norm2pixel(norm, pixel, distort);
}

void EquidistantCamera::distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const
{
    const Eigen::Vector3d hp = p.homogeneous();
    const double theta = acos(hp(2) / hp.norm());
    const double theta2 = theta*theta;
    const double dtheta = theta*(1 + mParameters.k2()*theta2
                                 + mParameters.k3()*theta2*theta2
                                 + mParameters.k4()*theta2*theta2*theta2
                                 + mParameters.k5()*theta2*theta2*theta2*theta2);
    const double phi = atan2(hp(1), hp(0));
    dp << dtheta*cos(phi), dtheta*sin(phi);
}

int EquidistantCamera::parameterCount(void) const
{
    return 8;
}

const EquidistantCamera::Parameters&
EquidistantCamera::getParameters(void) const
{
    return mParameters;
}

void
EquidistantCamera::setParameters(const EquidistantCamera::Parameters& parameters)
{
    mParameters = parameters;

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.mu();
    m_inv_K13 = -mParameters.u0() / mParameters.mu();
    m_inv_K22 = 1.0 / mParameters.mv();
    m_inv_K23 = -mParameters.v0() / mParameters.mv();
}

void
EquidistantCamera::readParameters(const std::vector<double>& parameterVec)
{
    if (parameterVec.size() != parameterCount())
    {
        return;
    }

    Parameters params = getParameters();

    params.k2() = parameterVec.at(0);
    params.k3() = parameterVec.at(1);
    params.k4() = parameterVec.at(2);
    params.k5() = parameterVec.at(3);
    params.mu() = parameterVec.at(4);
    params.mv() = parameterVec.at(5);
    params.u0() = parameterVec.at(6);
    params.v0() = parameterVec.at(7);

    setParameters(params);
}

void
EquidistantCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());
    parameterVec.at(0) = mParameters.k2();
    parameterVec.at(1) = mParameters.k3();
    parameterVec.at(2) = mParameters.k4();
    parameterVec.at(3) = mParameters.k5();
    parameterVec.at(4) = mParameters.mu();
    parameterVec.at(5) = mParameters.mv();
    parameterVec.at(6) = mParameters.u0();
    parameterVec.at(7) = mParameters.v0();
}

void
EquidistantCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
EquidistantCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

void EquidistantCamera::backprojectSymmetric(const Eigen::Vector2d& p_u,
                                             double& theta, double& phi) const
{
    double tol = 1e-10;
    double p_u_norm = p_u.norm();

    if (p_u_norm < 1e-10)
    {
        phi = 0.0;
    }
    else
    {
        phi = atan2(p_u(1), p_u(0));
    }

    int npow = 9;
    if (mParameters.k5() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k4() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k3() == 0.0)
    {
        npow -= 2;
    }
    if (mParameters.k2() == 0.0)
    {
        npow -= 2;
    }

    Eigen::MatrixXd coeffs(npow + 1, 1);
    coeffs.setZero();
    coeffs(0) = -p_u_norm;
    coeffs(1) = 1.0;

    if (npow >= 3)
    {
        coeffs(3) = mParameters.k2();
    }
    if (npow >= 5)
    {
        coeffs(5) = mParameters.k3();
    }
    if (npow >= 7)
    {
        coeffs(7) = mParameters.k4();
    }
    if (npow >= 9)
    {
        coeffs(9) = mParameters.k5();
    }

    if (npow == 1)
    {
        theta = p_u_norm;
    }
    else
    {
        // Get eigenvalues of companion matrix corresponding to polynomial.
        // Eigenvalues correspond to roots of polynomial.
        Eigen::MatrixXd A(npow, npow);
        A.setZero();
        A.block(1, 0, npow - 1, npow - 1).setIdentity();
        A.col(npow - 1) = - coeffs.block(0, 0, npow, 1) / coeffs(npow);

        Eigen::EigenSolver<Eigen::MatrixXd> es(A);
        Eigen::MatrixXcd eigval = es.eigenvalues();

        std::vector<double> thetas;
        for (int i = 0; i < eigval.rows(); ++i)
        {
            if (fabs(eigval(i).imag()) > tol)
            {
                continue;
            }

            double t = eigval(i).real();

            if (t < -tol)
            {
                continue;
            }
            else if (t < 0.0)
            {
                t = 0.0;
            }

            thetas.push_back(t);
        }

        if (thetas.empty())
        {
            theta = p_u_norm;
        }
        else
        {
            theta = *std::min_element(thetas.begin(), thetas.end());
        }
    }
}

}
