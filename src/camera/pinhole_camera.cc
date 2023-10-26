#include "pinhole_camera.h"

#include <cmath>
#include <cstdio>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>

namespace CameraModel
{

PinholeCamera::Parameters::Parameters()
 : Camera::Parameters(PINHOLE)
 , m_k1(0.0)
 , m_k2(0.0)
 , m_p1(0.0)
 , m_p2(0.0)
 , m_fx(0.0)
 , m_fy(0.0)
 , m_cx(0.0)
 , m_cy(0.0)
{

}

PinholeCamera::Parameters::Parameters(int w, int h,
                                      double k1, double k2,
                                      double p1, double p2,
                                      double fx, double fy,
                                      double cx, double cy)
 : Camera::Parameters(PINHOLE, w, h)
 , m_k1(k1)
 , m_k2(k2)
 , m_p1(p1)
 , m_p2(p2)
 , m_fx(fx)
 , m_fy(fy)
 , m_cx(cx)
 , m_cy(cy)
{
}

double&
PinholeCamera::Parameters::k1(void)
{
    return m_k1;
}

double&
PinholeCamera::Parameters::k2(void)
{
    return m_k2;
}

double&
PinholeCamera::Parameters::p1(void)
{
    return m_p1;
}

double&
PinholeCamera::Parameters::p2(void)
{
    return m_p2;
}

double&
PinholeCamera::Parameters::fx(void)
{
    return m_fx;
}

double&
PinholeCamera::Parameters::fy(void)
{
    return m_fy;
}

double&
PinholeCamera::Parameters::cx(void)
{
    return m_cx;
}

double&
PinholeCamera::Parameters::cy(void)
{
    return m_cy;
}

double
PinholeCamera::Parameters::k1(void) const
{
    return m_k1;
}

double
PinholeCamera::Parameters::k2(void) const
{
    return m_k2;
}

double
PinholeCamera::Parameters::p1(void) const
{
    return m_p1;
}

double
PinholeCamera::Parameters::p2(void) const
{
    return m_p2;
}

double
PinholeCamera::Parameters::fx(void) const
{
    return m_fx;
}

double
PinholeCamera::Parameters::fy(void) const
{
    return m_fy;
}

double
PinholeCamera::Parameters::cx(void) const
{
    return m_cx;
}

double
PinholeCamera::Parameters::cy(void) const
{
    return m_cy;
}

bool
PinholeCamera::Parameters::readFromYamlFile(const std::string& filename)
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

        if (sModelType.compare("PINHOLE") != 0)
        {
            return false;
        }
    }

    m_modelType = PINHOLE;

    std::vector<int> resolution;
    std::vector<double> intrinsic;
    std::vector<double> dist_coef;
    fs["resolution"] >> resolution;
    fs["intrinsics"] >> intrinsic;
    fs["distortion_coefficients"] >> dist_coef;

    m_imageWidth = resolution[0];
    m_imageHeight = resolution[1];
    m_fx = intrinsic[0];
    m_fy = intrinsic[1];
    m_cx = intrinsic[2];
    m_cy = intrinsic[3];
    m_k1 = dist_coef[0];
    m_k2 = dist_coef[1];
    m_p1 = dist_coef[2];
    m_p2 = dist_coef[3];
    return true;
}

void
PinholeCamera::Parameters::writeToYamlFile(const std::string& filename) const
{
    cv::FileStorage fs(filename, cv::FileStorage::WRITE);

    fs << "model_type" << "PINHOLE";
    fs << "image_width" << m_imageWidth;
    fs << "image_height" << m_imageHeight;

    // radial distortion: k1, k2
    // tangential distortion: p1, p2
    fs << "distortion_parameters";
    fs << "{" << "k1" << m_k1
              << "k2" << m_k2
              << "p1" << m_p1
              << "p2" << m_p2 << "}";

    // projection: fx, fy, cx, cy
    fs << "projection_parameters";
    fs << "{" << "fx" << m_fx
              << "fy" << m_fy
              << "cx" << m_cx
              << "cy" << m_cy << "}";

    fs.release();
}

PinholeCamera::Parameters&
PinholeCamera::Parameters::operator=(const PinholeCamera::Parameters& other)
{
    if (this != &other)
    {
        m_modelType = other.m_modelType;
        m_imageWidth = other.m_imageWidth;
        m_imageHeight = other.m_imageHeight;
        m_k1 = other.m_k1;
        m_k2 = other.m_k2;
        m_p1 = other.m_p1;
        m_p2 = other.m_p2;
        m_fx = other.m_fx;
        m_fy = other.m_fy;
        m_cx = other.m_cx;
        m_cy = other.m_cy;
    }

    return *this;
}

std::ostream&
operator<< (std::ostream& out, const PinholeCamera::Parameters& params)
{
    out << "Camera Parameters:" << std::endl;
    out << "    model_type " << "PINHOLE" << std::endl;
    out << "   image_width " << params.m_imageWidth << std::endl;
    out << "  image_height " << params.m_imageHeight << std::endl;

    // radial distortion: k1, k2
    // tangential distortion: p1, p2
    out << "Distortion Parameters" << std::endl;
    out << "            k1 " << params.m_k1 << std::endl
        << "            k2 " << params.m_k2 << std::endl
        << "            p1 " << params.m_p1 << std::endl
        << "            p2 " << params.m_p2 << std::endl;

    // projection: fx, fy, cx, cy
    out << "Projection Parameters" << std::endl;
    out << "            fx " << params.m_fx << std::endl
        << "            fy " << params.m_fy << std::endl
        << "            cx " << params.m_cx << std::endl
        << "            cy " << params.m_cy << std::endl;

    return out;
}

PinholeCamera::PinholeCamera()
 : m_inv_K11(1.0)
 , m_inv_K13(0.0)
 , m_inv_K22(1.0)
 , m_inv_K23(0.0)
 , m_noDistortion(true)
{

}

PinholeCamera::PinholeCamera(int imageWidth, int imageHeight,
                             double k1, double k2, double p1, double p2,
                             double fx, double fy, double cx, double cy)
 : mParameters(imageWidth, imageHeight,
               k1, k2, p1, p2, fx, fy, cx, cy)
{
    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

PinholeCamera::PinholeCamera(const PinholeCamera::Parameters& params)
 : mParameters(params)
{
    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    // Inverse camera projection matrix parameters
    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

Camera::ModelType
PinholeCamera::modelType(void) const
{
    return mParameters.modelType();
}

int
PinholeCamera::imageWidth(void) const
{
    return mParameters.imageWidth();
}

int
PinholeCamera::imageHeight(void) const
{
    return mParameters.imageHeight();
}


void PinholeCamera::pixel2norm(const Eigen::Vector2d& pixel, Eigen::Vector2d& norm) const
{
    double mx_d, my_d,mx2_d, mxy_d, my2_d, mx_u, my_u;
    double rho2_d, rho4_d, radDist_d, Dx_d, Dy_d, inv_denom_d;
    // Lift points to normalized plane
    mx_d = m_inv_K11 * pixel(0) + m_inv_K13;
    my_d = m_inv_K22 * pixel(1) + m_inv_K23;

    if (m_noDistortion)
    {
        mx_u = mx_d;
        my_u = my_d;
    }
    else
    {
        if (0)
        {
            double k1 = mParameters.k1();
            double k2 = mParameters.k2();
            double p1 = mParameters.p1();
            double p2 = mParameters.p2();

            // Apply inverse distortion model
            // proposed by Heikkila
            mx2_d = mx_d*mx_d;
            my2_d = my_d*my_d;
            mxy_d = mx_d*my_d;
            rho2_d = mx2_d+my2_d;
            rho4_d = rho2_d*rho2_d;
            radDist_d = k1*rho2_d+k2*rho4_d;
            Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
            Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
            inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

            mx_u = mx_d - inv_denom_d*Dx_d;
            my_u = my_d - inv_denom_d*Dy_d;
        }
        else
        {
            // Recursive distortion model
            int n = 8;
            Eigen::Vector2d d_u;
            distortpoint(Eigen::Vector2d(mx_d, my_d), d_u);
            // Approximate value
            mx_u = 2*mx_d - d_u(0);
            my_u = 2*my_d - d_u(1);

            for (int i = 1; i < n; ++i)
            {
                distortpoint(Eigen::Vector2d(mx_u, my_u), d_u);
                mx_u = mx_d - (d_u(0) - mx_u);
                my_u = my_d - (d_u(1) - my_u);
            }
        }
    }
    // Obtain a projective ray
    norm << mx_u, my_u;
}

void PinholeCamera::pixel2hnorm(const Eigen::Vector2d& pixel, Eigen::Vector3d& hnorm) const
{
    Eigen::Vector2d norm;
    pixel2norm(pixel, norm);
    hnorm = norm.homogeneous();
}

void PinholeCamera::norm2pixel(const Eigen::Vector2d& norm, Eigen::Vector2d& pixel, bool distort) const
{
    Eigen::Vector2d p_u;
    if (m_noDistortion || !distort)
    {
        p_u = norm;
    }
    else
    {
        distortpoint(norm, p_u);
    }

    // Apply generalised projection matrix
    pixel << mParameters.fx() * p_u(0) + mParameters.cx(),
            mParameters.fy() * p_u(1) + mParameters.cy();
}

void PinholeCamera::mappoint2pixel(const Eigen::Vector3d& mappoint, Eigen::Vector2d& pixel, bool distort) const
{
    const Eigen::Vector2d norm = mappoint.hnormalized();
    norm2pixel(norm, pixel, distort);
}

void PinholeCamera::distortpoint(const Eigen::Vector2d& p, Eigen::Vector2d& dp) const
{
    double k1 = mParameters.k1();
    double k2 = mParameters.k2();
    double p1 = mParameters.p1();
    double p2 = mParameters.p2();

    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p(0) * p(0);
    my2_u = p(1) * p(1);
    mxy_u = p(0) * p(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    dp << p(0) + p(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
          p(1) + p(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

int PinholeCamera::parameterCount(void) const
{
    return 8;
}

const PinholeCamera::Parameters&
PinholeCamera::getParameters(void) const
{
    return mParameters;
}

void
PinholeCamera::setParameters(const PinholeCamera::Parameters& parameters)
{
    mParameters = parameters;

    if ((mParameters.k1() == 0.0) &&
        (mParameters.k2() == 0.0) &&
        (mParameters.p1() == 0.0) &&
        (mParameters.p2() == 0.0))
    {
        m_noDistortion = true;
    }
    else
    {
        m_noDistortion = false;
    }

    m_inv_K11 = 1.0 / mParameters.fx();
    m_inv_K13 = -mParameters.cx() / mParameters.fx();
    m_inv_K22 = 1.0 / mParameters.fy();
    m_inv_K23 = -mParameters.cy() / mParameters.fy();
}

void
PinholeCamera::readParameters(const std::vector<double>& parameterVec)
{
    if ((int)parameterVec.size() != parameterCount())
    {
        return;
    }

    Parameters params = getParameters();

    params.k1() = parameterVec.at(0);
    params.k2() = parameterVec.at(1);
    params.p1() = parameterVec.at(2);
    params.p2() = parameterVec.at(3);
    params.fx() = parameterVec.at(4);
    params.fy() = parameterVec.at(5);
    params.cx() = parameterVec.at(6);
    params.cy() = parameterVec.at(7);

    setParameters(params);
}

void
PinholeCamera::writeParameters(std::vector<double>& parameterVec) const
{
    parameterVec.resize(parameterCount());
    parameterVec.at(0) = mParameters.k1();
    parameterVec.at(1) = mParameters.k2();
    parameterVec.at(2) = mParameters.p1();
    parameterVec.at(3) = mParameters.p2();
    parameterVec.at(4) = mParameters.fx();
    parameterVec.at(5) = mParameters.fy();
    parameterVec.at(6) = mParameters.cx();
    parameterVec.at(7) = mParameters.cy();
}

void
PinholeCamera::writeParametersToYamlFile(const std::string& filename) const
{
    mParameters.writeToYamlFile(filename);
}

std::string
PinholeCamera::parametersToString(void) const
{
    std::ostringstream oss;
    oss << mParameters;

    return oss.str();
}

}
