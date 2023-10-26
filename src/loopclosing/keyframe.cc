
#include "keyframe.h"

KeyFrame::KeyFrame(io::timestamp_t _time_stamp, int _index, int _sequence_id, Eigen::Isometry3d &_Twb, cv::Mat &_image)
{
    time_stamp = _time_stamp;
    index = _index;
    sequence_id = _sequence_id;
    Twb = _Twb;
    Eigen::Isometry3d Tbc(RbcG);
    Tbc.pretranslate(tbcG);
    Twc = _Twb*Tbc;

    image = _image.clone();
    loop_index = -1;
    ComputeBRIEFPoint();
    image.release();
}

void KeyFrame::ComputeBRIEFPoint()
{
    ORB_Extractor ext(500, 1.2f, 8, 20, 7);
    ext.Detect(image, keypoints, descriptors);

    keypoints_norm.resize(keypoints.size());
    for (int i = 0; i < (int)keypoints.size(); ++i)
    {
        Eigen::Vector3d tmp_p = Pixel2Hnorm(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y));
        cv::KeyPoint tmp_norm;
        keypoints_norm[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
    }
    mappoint_ids.resize(keypoints.size(), -1);
}

void KeyFrame::SetPose(const Eigen::Isometry3d& _Twb)
{
    Twb = _Twb;
    Eigen::Isometry3d Tbc(RbcG);
    Tbc.pretranslate(tbcG);
    Twc = _Twb*Tbc;
}

void KeyFrame::SetUpdatePose(const Eigen::Isometry3d& _Twb_update)
{
    Twb_update = _Twb_update;
    Eigen::Isometry3d Tbc(RbcG);
    Tbc.pretranslate(tbcG);
    Twc_update = _Twb_update*Tbc;
}


void KeyFrame::SetLoopMessage(const Eigen::Isometry3d &_Tij, const int loop_id)
{
    loop_Tij = _Tij;
    loop_index = loop_id;
}

void KeyFrame::SetLoopMessageGt(const Eigen::Isometry3d &_Tij_gt, const int loop_id)
{
    loop_Tij_gt = _Tij_gt;
    loop_index = loop_id;
}
