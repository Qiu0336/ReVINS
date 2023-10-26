
#include "camera.h"

namespace CameraModel
{

boost::shared_ptr<CameraFactory> CameraFactory::m_instance;

CameraFactory::CameraFactory()
{
}

boost::shared_ptr<CameraFactory>
CameraFactory::instance(void)
{
    if (m_instance.get() == 0)
    {
        m_instance.reset(new CameraFactory);
    }

    return m_instance;
}

CameraPtr CameraFactory::generateCamera(Camera::ModelType modelType,
                              cv::Size imageSize) const
{
    PinholeCameraPtr camera(new PinholeCamera);

    PinholeCamera::Parameters params = camera->getParameters();
    params.imageWidth() = imageSize.width;
    params.imageHeight() = imageSize.height;
    camera->setParameters(params);
    return camera;
}

CameraPtr CameraFactory::generateCameraFromYamlFile(const std::string& filename)
{
    cv::FileStorage fs(filename, cv::FileStorage::READ);

    if(!fs.isOpened())
    {
        return CameraPtr();
    }


    Camera::ModelType modelType = Camera::PINHOLE;
    if (!fs["model_type"].isNone())
    {
        std::string sModelType;
        fs["model_type"] >> sModelType;

        if (boost::iequals(sModelType, "pinhole"))
        {
            modelType = Camera::PINHOLE;
        }
        else if (boost::iequals(sModelType, "kannala_brandt"))
        {
            modelType = Camera::KANNALA_BRANDT;
        }
        else
        {
            std::cerr << "# ERROR: Unknown camera model: " << sModelType << std::endl;
            return CameraPtr();
        }
    }

    switch (modelType)
    {
        case Camera::PINHOLE:
        {
            PinholeCameraPtr camera(new PinholeCamera);

            PinholeCamera::Parameters params = camera->getParameters();
            params.readFromYamlFile(filename);
            camera->setParameters(params);
            return camera;
        }
        case Camera::KANNALA_BRANDT:
        {
            EquidistantCameraPtr camera(new EquidistantCamera);

            EquidistantCamera::Parameters params = camera->getParameters();
            params.readFromYamlFile(filename);
            camera->setParameters(params);
            return camera;
        }
    }

    return CameraPtr();
}

}
