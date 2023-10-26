#pragma once

#include <boost/shared_ptr.hpp>
#include <opencv2/core/core.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>

#include "camera_model.h"
#include "pinhole_camera.h"
#include "equidistant_camera.h"


namespace CameraModel
{

class CameraFactory
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    CameraFactory();

    static boost::shared_ptr<CameraFactory> instance(void);

    CameraPtr generateCamera(Camera::ModelType modelType,
                             cv::Size imageSize) const;

    CameraPtr generateCameraFromYamlFile(const std::string& filename);

private:
    static boost::shared_ptr<CameraFactory> m_instance;
};

}

