#include "camera_model.h"

namespace CameraModel
{

Camera::Parameters::Parameters(ModelType modelType)
 : m_modelType(modelType)
 , m_imageWidth(0)
 , m_imageHeight(0)
{
    switch (modelType)
    {
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
    default:
        m_nIntrinsics = 9;
    }
}

Camera::Parameters::Parameters(ModelType modelType, int w, int h)
 : m_modelType(modelType)
 , m_imageWidth(w)
 , m_imageHeight(h)
{
    switch (modelType)
    {
    case KANNALA_BRANDT:
        m_nIntrinsics = 8;
        break;
    case PINHOLE:
        m_nIntrinsics = 8;
        break;
        break;
    default:
        m_nIntrinsics = 9;
    }
}

Camera::ModelType&
Camera::Parameters::modelType(void)
{
    return m_modelType;
}

int&
Camera::Parameters::imageWidth(void)
{
    return m_imageWidth;
}

int&
Camera::Parameters::imageHeight(void)
{
    return m_imageHeight;
}

Camera::ModelType
Camera::Parameters::modelType(void) const
{
    return m_modelType;
}

int
Camera::Parameters::imageWidth(void) const
{
    return m_imageWidth;
}

int
Camera::Parameters::imageHeight(void) const
{
    return m_imageHeight;
}

int
Camera::Parameters::nIntrinsics(void) const
{
    return m_nIntrinsics;
}

cv::Mat&
Camera::mask(void)
{
    return m_mask;
}

const cv::Mat&
Camera::mask(void) const
{
    return m_mask;
}

}
