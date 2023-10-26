#include "image_processor.h"


template<typename T1, typename T2>
void ReduceVector(std::vector<T1> &v, std::vector<T2> s)
{
    int j = 0;
    for(int i = 0, iend = int(v.size()); i < iend; ++i)
        if(s[i])
            v[j++] = v[i];
    v.resize(j);
}


struct Block
{
    Block(){}
    Block(int top_, int bottom_, int left_, int right_):
        top(top_), bottom(bottom_), left(left_), right(right_)
    {}
    std::vector<std::pair<int, cv::Point2f>> points;
    int top, bottom, left, right;
    bool InBlock(cv::Point2f &p)
    {
        return (p.x >= left && p.x < right && p.y >= top && p.y < bottom);
    }
    int StrongestPointID(cv::Mat &eigen)
    {
        if(points.empty())
            return -1;
        int cur_id = 0, max_id = 0;
        float max_res = 0;
        for(auto &p : points)
        {
            if(eigen.at<float>(p.second) > max_res)
            {
                max_res = eigen.at<float>(p.second);
                max_id = cur_id;
            }
            ++cur_id;
        }
        return max_id;
    }
};

void GoodFeaturesToTrackProShortWithTrackedPoints(const cv::Mat &grad_x, const cv::Mat &grad_y, cv::OutputArray _new_corners,
                                                  int max_corners, float density_th, float upper_th, const std::vector<cv::Point2f>& tracked_points,
                                                  std::vector<uchar>& inliners)
{
    inliners.resize(tracked_points.size(), 0);
    const float FLT_SCALE10 = 1.f/(1 << 10);
    const int height = grad_x.rows;
    const int width = grad_x.cols;

    cv::Mat cov_dxdx(grad_x.size(), CV_32F);
    cv::Mat cov_dxdy(grad_x.size(), CV_32F);
    cv::Mat cov_dydy(grad_x.size(), CV_32F);

    for(int y = 0; y < height; ++y)
    {
        float* cov_data1 = cov_dxdx.ptr<float>(y);
        float* cov_data2 = cov_dxdy.ptr<float>(y);
        float* cov_data3 = cov_dydy.ptr<float>(y);
        const short* dxdata = grad_x.ptr<short>(y);
        const short* dydata = grad_y.ptr<short>(y);
        int x = 0;
        for(; x < width; ++x)
        {
            short dx = dxdata[x];
            short dy = dydata[x];
            cov_data1[x] = (dx*dx)*FLT_SCALE10;
            cov_data2[x] = (dx*dy)*FLT_SCALE10;
            cov_data3[x] = (dy*dy)*FLT_SCALE10;
        }
    }

    cv::blur(cov_dxdx, cov_dxdx, cv::Size(3, 3));
    cv::blur(cov_dxdy, cov_dxdy, cv::Size(3, 3));
    cv::blur(cov_dydy, cov_dydy, cv::Size(3, 3));

    cv::Mat eig(grad_x.size(), CV_32F);

    const __m128 c_5 = _mm_set_ps(0.5, 0.5, 0.5, 0.5);

    for(int y = 0; y < height; ++y)
    {
        const float* cov_data1 = cov_dxdx.ptr<float>(y);
        const float* cov_data2 = cov_dxdy.ptr<float>(y);
        const float* cov_data3 = cov_dydy.ptr<float>(y);
        float* dst_ptr = eig.ptr<float>(y);
        int x = 0;
        for(; x <= width - 4; x += 4)
        {
            __m128 v_a = _mm_mul_ps(c_5, _mm_loadu_ps(cov_data1 + x));
            __m128 v_b = _mm_loadu_ps(cov_data2 + x);
            __m128 v_c = _mm_mul_ps(c_5, _mm_loadu_ps(cov_data3 + x));
            __m128 a_c = _mm_sub_ps(v_a, v_c);
            _mm_storeu_ps(dst_ptr + x, _mm_sub_ps(_mm_add_ps(v_a, v_c), _mm_sqrt_ps(_mm_add_ps(_mm_mul_ps(a_c, a_c), _mm_mul_ps(v_b, v_b)))));
        }
        for(; x < width; ++x)
        {
            float a = cov_data1[x]/2;
            float b = cov_data2[x];
            float c = cov_data3[x]/2;
            dst_ptr[x] = float((a + c) - std::sqrt((a - c)*(a - c) + b*b));
        }
    }

    const int grid_size = 64;
    const int threshold_nth = density_th*grid_size*grid_size;
    const int grad_width = width/grid_size;
    const int grad_height = height/grid_size;
    cv::Mat thresholds(cv::Size(grad_width, grad_height), CV_32F);
    for(int gy = 0; gy < grad_height; ++gy)
    {
        int s_y = gy*grid_size;
        int e_y = s_y + grid_size;
        float* dst_threshold = thresholds.ptr<float>(gy);
        for(int gx = 0; gx < grad_width; ++gx)
        {
            int s_x = gx*grid_size;
            int e_x = s_x + grid_size;
            std::vector<float> eig_grid;
            for(int y = s_y; y < e_y; ++y)
            {
                float* eig_data = eig.ptr<float>(y);
                for(int x = s_x; x < e_x; ++x)
                {
                    eig_grid.emplace_back(eig_data[x]);
                }
            }
            std::nth_element(eig_grid.begin(), eig_grid.begin() + threshold_nth, eig_grid.end(),
                             [](const float &a, const float &b){return a > b;});
            dst_threshold[gx] = eig_grid[threshold_nth] + upper_th;
        }
    }
    cv::GaussianBlur(thresholds, thresholds, cv::Size(3, 3), 0, 0, cv::BORDER_REPLICATE);

    Block bG(0, grad_x.rows, 0, grad_x.cols);

    {
        for(int i = 0; i < tracked_points.size(); ++i)
        {
            bG.points.emplace_back(i, tracked_points[i]);
        }

        int s_y, e_y;
        for(int gy = 0; gy < grad_height; ++gy)
        {
            s_y = gy*grid_size;
            e_y = s_y + grid_size;
            int s_x, e_x;
            float th;
            for(int gx = 0; gx < grad_width; ++gx)
            {
                s_x = gx*grid_size;
                e_x = s_x + grid_size;
                th = thresholds.at<float>(cv::Point2f(gx, gy));
                for(int y = s_y; y < e_y; ++y)
                {
                    const float* eig_data = (const float*)eig.ptr<float>(y);
                    for(int x = s_x; x < e_x; ++x)
                    {
                        if(eig_data[x] > th)
                        {
                            bG.points.emplace_back(-1, cv::Point2f((float)x, (float)y));
                        }
                    }
                }
            }
            for(int y = s_y; y < e_y; ++y)
            {
                const float* eig_data = (const float*)eig.ptr<float>(y);
                for(int x = e_x; x < width; ++x)
                {
                    if(eig_data[x] > th)
                        bG.points.emplace_back(-1, cv::Point2f((float)x, (float)y));
                }
            }
        }
        int s_x, e_x;
        float th;
        for(int gx = 0; gx < grad_width; ++gx)
        {
            s_x = gx*grid_size;
            e_x = s_x + grid_size;
            th = thresholds.at<float>(cv::Point2f(gx, grad_height - 1));
            for(int y = e_y; y < height; ++y)
            {
                const float* eig_data = (const float*)eig.ptr<float>(y);
                for(int x = s_x; x < e_x; ++x)
                {
                    if(eig_data[x] > th)
                        bG.points.emplace_back(-1, cv::Point2f((float)x, (float)y));
                }
            }
        }

        for(int y = e_y; y < height; ++y)
        {
            const float* eig_data = (const float*)eig.ptr<float>(y);
            for(int x = e_x; x < width; ++x)
            {
                if(eig_data[x] > th)
                    bG.points.emplace_back(-1, cv::Point2f((float)x, (float)y));
            }
        }
    }



    std::vector<cv::Point2f> new_corners;
    int corners_num = 0;
    {
        if(0 == bG.points.size())
        {
            _new_corners.release();
            return;
        }

        std::queue<Block> blocks;
        blocks.push(bG);
        while(!blocks.empty() && (int)blocks.size() < max_corners - corners_num)
        {
            Block b = blocks.front();
            blocks.pop();
            if(b.points.empty())
                continue;

            int mid_y = (b.top + b.bottom)/2;
            int mid_x = (b.left + b.right)/2;
            Block sub_b[4];
            sub_b[0] = Block(b.top, mid_y, b.left, mid_x);
            sub_b[1] = Block(b.top, mid_y, mid_x, b.right);
            sub_b[2] = Block(mid_y, b.bottom, b.left, mid_x);
            sub_b[3] = Block(mid_y, b.bottom, mid_x, b.right);
            for(auto &p : b.points)
            {
                for(int i = 0; i < 4; ++i)
                {
                    if(sub_b[i].InBlock(p.second))
                    {
                        sub_b[i].points.push_back(p);
                        break;
                    }
                }
            }
            for(int i = 0; i < 4; ++i)
            {
                if(!sub_b[i].points.empty())
                {
                    if(sub_b[i].points.size() == 1
                     || ((sub_b[i].bottom - sub_b[i].top) < 2 || (sub_b[i].right - sub_b[i].left) < 2) // 如果小于2个pixel，则不再细分
                            )
                    {
                        ++corners_num;
                        if(sub_b[i].points[0].first >= 0)
                        {
                            inliners[sub_b[i].points[0].first] = 1;
                        }
                        else
                        {
                            new_corners.push_back(sub_b[i].points[0].second);
                        }
                    }
                    else
                        blocks.push(sub_b[i]);
                }
            }
        }
        while(!blocks.empty())
        {
            Block &b = blocks.front();
            if(b.points[0].first >= 0)
            {
                inliners[b.points[0].first] = 1;
            }
            else
            {
                int id = b.StrongestPointID(eig);
                new_corners.push_back(b.points[id].second);
            }
            blocks.pop();
        }
    }

    cv::Mat(new_corners).convertTo(_new_corners, _new_corners.fixedType() ? _new_corners.type() : CV_32F);
}



void PointFeature::TrackPatch(const std::vector<cv::Mat>& dst_img_pyr, const std::vector<cv::Mat>& derivX_pyr, const std::vector<cv::Mat>& derivY_pyr,
                              cv::Point2f& cur_pt, uchar& status, float& error, const Eigen::Isometry3d& Tji,
                              bool use_penalty)
{
    #define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

    const int W_BITS = 14, W_BITS1 = 14;
    const float FLT_SCALE20 = 1.f/(1 << 20);
    cv::Size winSize = cv::Size(2*half_patch_size+1, 2*half_patch_size+1);
    int patch_area = winSize.area();
    cv::Point2f halfWin(half_patch_size, half_patch_size);

    auto OutBorder = [&](const cv::Point2f& point_, int level){
        cv::Point2f pt_corner = point_ - halfWin;
        cv::Point2i ipt_corner;
        ipt_corner.x = cvFloor(pt_corner.x);
        ipt_corner.y = cvFloor(pt_corner.y);
        if(ipt_corner.x < -winSize.width || ipt_corner.x >= dst_img_pyr[level].cols ||
           ipt_corner.y < -winSize.height || ipt_corner.y >= dst_img_pyr[level].rows)
            return true;
        return false;
    };

    auto OutBorderPatch = [&](const cv::Point2f& point_, int level){
        cv::Point2i ipt_corner;
        ipt_corner.x = cvFloor(point_.x);
        ipt_corner.y = cvFloor(point_.y);
        if(ipt_corner.x < -winSize.width || ipt_corner.x >= dst_img_pyr[level].cols + winSize.width ||
           ipt_corner.y < -winSize.height || ipt_corner.y >= dst_img_pyr[level].rows + winSize.height)
            return true;
        return false;
    };

    auto GetCurImgValue = [&](const cv::Point2f& point_, int level, int stepI){
        cv::Point2i ipoint_;
        ipoint_.x = cvFloor(point_.x);
        ipoint_.y = cvFloor(point_.y);
        float a_ = point_.x - ipoint_.x;
        float b_ = point_.y - ipoint_.y;

        int iw00_ = cvRound((1.f - a_)*(1.f - b_)*(1 << W_BITS));
        int iw01_ = cvRound(a_*(1.f - b_)*(1 << W_BITS));
        int iw10_ = cvRound((1.f - a_)*b_*(1 << W_BITS));
        int iw11_ = (1 << W_BITS) - iw00_ - iw01_ - iw10_;
        const uchar* src_ = dst_img_pyr[level].ptr(ipoint_.y) + ipoint_.x;
        int ival_ = CV_DESCALE(src_[0]*iw00_ + src_[1]*iw01_ +
                               src_[stepI]*iw10_ + src_[stepI+1]*iw11_, W_BITS1-5);
        return ival_;
    };


    cur_pt.x = src_corner.x;
    cur_pt.y = src_corner.y;
    std::vector<cv::Point2f> affine_warp;
    {
        Eigen::Matrix3d Rji = Tji.rotation();
        Eigen::Vector3d tji = Tji.translation();
        std::vector<cv::Point2f> AffineMatrix;
        Eigen::Vector2d point_src(src_corner.x, src_corner.y);
        Eigen::Vector2d point_dst, point_dst2;
        const Eigen::Vector3d npi = Pixel2Hnorm(point_src);
        {
            if(solve_flag)
            {
                double lambda = 1.0 / (Rji.row(2)*npi + tji(2)*inv_dep);
                Eigen::Vector3d npj = (Rji*npi + tji*inv_dep)*lambda;
                point_dst = Mappoint2Pixel(npj);
            }
            else
            {
                double lambda = 1.0 / (Rji.row(2)*npi);
                Eigen::Vector3d npj = (Rji*npi)*lambda;
                point_dst = Mappoint2Pixel(npj);
            }
        }

        cur_pt.x = point_dst.x();
        cur_pt.y = point_dst.y();

        if(consider_affine)
        {
            Eigen::Matrix<double, 2, 4> affine_left_multiplyer;
            for(int i = 0; i < 4; ++i)
            {
                Eigen::Vector2d pjc;
                const Eigen::Vector2d pic = point_dst + patch_four_corners.col(i);
                const Eigen::Vector3d npic = Pixel2Hnorm(pic);
                if(solve_flag)
                {
                    double lambda = 1.0 / (Rji.row(2)*npic + tji(2)*inv_dep);
                    Eigen::Vector3d npjc = (Rji*npic + tji*inv_dep)*lambda;
                    pjc = Mappoint2Pixel(npjc);
                }
                else
                {
                    double lambda = 1.0 / (Rji.row(2)*npic);
                    Eigen::Vector3d npjc = (Rji*npic)*lambda;
                    pjc = Mappoint2Pixel(npjc);
                }
                affine_left_multiplyer.col(i) = pjc - point_dst;
            }

            Eigen::Matrix2d affine = affine_left_multiplyer*affine_right_multiplyer;
            int idx = 0;
            for(int y = -half_patch_size; y <= half_patch_size; ++y)
            {
                for(int x = -half_patch_size; x <= half_patch_size; ++x)
                {
                    float wx = affine(0, 0)*x + affine(0, 1)*y;
                    float wy = affine(1, 0)*x + affine(1, 1)*y;
                    affine_warp.emplace_back(wx, wy);
                    ++idx;
                }
            }
        }
    }

    cv::Point2f pred_pos = cur_pt;

    for(int level = pyramid_levels - 1; level >= 0; --level)
    {
        int stepI = (int)(dst_img_pyr[level].step/dst_img_pyr[level].elemSize1());
        int dstep = (int)(derivX_pyr[level].step/derivX_pyr[level].elemSize1());
        cv::Point2f prevPt = src_corner*(float)(1./(1 << level));
        cv::Point2f nextPt;
        if(level == pyramid_levels - 1)
            nextPt = cur_pt*(float)(1./(1 << level));
        else
            nextPt = cur_pt*2.f;
        cur_pt = nextPt;

        cv::Point2f pred_pos_level = pred_pos*(float)(1./(1 << level));
        float level_factor = (float)(1./(1 << level));

        if(OutBorder(prevPt, level))
        {
            if(level == 0)
            {
                status = 0;
                error = 0;
            }
            return;
        }
        short* IWinBuf = src_patch[level].ptr<short>();

        int iw00, iw01, iw10, iw11, idx;
        Eigen::VectorXf prevDelta;
        float db = 0.f;
        for(int j = 0; j < 30; ++j)
        {
            cv::Point2i inextPt;
            inextPt.x = cvFloor(nextPt.x);
            inextPt.y = cvFloor(nextPt.y);

            if(OutBorder(nextPt, level))
            {
                if(level == 0)
                {
                    status = 0;
                    error = 0;
                }
                break;
            }

            float a = nextPt.x - inextPt.x;
            float b = nextPt.y - inextPt.y;
            iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
            iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
            iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;
            idx = 0;
            Eigen::VectorXi iresidual;
            iresidual.resize(patch_area);

            Eigen::MatrixXi ijacobian;
            if(consider_illumination)
                ijacobian.resize(patch_area, 3);
            else
                ijacobian.resize(patch_area, 2);

            bool patch_invalid = false;
            for(int y = -half_patch_size; y <= half_patch_size; ++y)
            {
                const uchar* Jptr = dst_img_pyr[level].ptr(inextPt.y + y) + inextPt.x;
                const short* ddstX = derivX_pyr[level].ptr<short>(inextPt.y + y) + inextPt.x;
                const short* ddstY = derivY_pyr[level].ptr<short>(inextPt.y + y) + inextPt.x;
                for(int x = -half_patch_size; x <= half_patch_size; ++x)
                {
                    int cur_ival;
                    if(consider_affine)
                    {
                        cv::Point2f warp_pt = nextPt + affine_warp[idx];
                        if(OutBorderPatch(warp_pt, level))
                        {
                            patch_invalid = true;
                            break;
                        }
                        cur_ival = GetCurImgValue(warp_pt, level, stepI);
                    }
                    else
                    {
                        cur_ival = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+1]*iw01 +
                                              Jptr[x+stepI]*iw10 + Jptr[x+stepI+1]*iw11,
                                              W_BITS1-5);
                    }


                    int jxval = CV_DESCALE(ddstX[x]*iw00 + ddstX[x+1]*iw01 +
                                           ddstX[x+dstep]*iw10 + ddstX[x+dstep+1]*iw11, W_BITS1);
                    int jyval = CV_DESCALE(ddstY[x]*iw00 + ddstY[x+1]*iw01 +
                                           ddstY[x+dstep]*iw10 + ddstY[x+dstep+1]*iw11, W_BITS1);
                    if(consider_illumination)
                        ijacobian.row(idx) << jxval, jyval, 32;
                    else
                        ijacobian.row(idx) << jxval, jyval;
                    int diff = cvRound(cur_ival + 32*db - IWinBuf[idx]);
                    iresidual(idx) = diff;
                    ++idx;
                }

                if(patch_invalid)
                {
                    break;
                }
            }

            if(patch_invalid)
            {
                if(level == 0)
                {
                    status = 0;
                    error = 0;
                }
                break;
            }

            Eigen::VectorXf residual = iresidual.cast<float>();

            Eigen::MatrixXf jacobian = ijacobian.cast<float>()*FLT_SCALE20;
            Eigen::MatrixXi ihessian = ijacobian.transpose()*ijacobian;
            Eigen::MatrixXf hessian = ihessian.cast<float>()*FLT_SCALE20;
            Eigen::MatrixXf hessian_inv = hessian.inverse();

            float A11 = hessian(0, 0);
            float A12 = hessian(0, 1);
            float A22 = hessian(1, 1);

            float D = A11*A22 - A12*A12;
            float minEig = (A22 + A11 - std::sqrt((A11-A22)*(A11-A22) +
                            4.f*A12*A12))/(2*patch_area);
            if( minEig < 1e-4 || D < FLT_EPSILON ||
                    hessian_inv.array().isInf().any() || hessian_inv.array().isNaN().any())
            {
                if(level == 0)
                {
                    status = false;
                    error = -1;
                }
                return;
            }
            Eigen::MatrixXf hessian_inv_jacobian_t = hessian_inv*jacobian.transpose();


            Eigen::VectorXf delta;

            if(use_penalty && solve_flag)
            {
                float mLambda = 100.f * level_factor;
                float mAlpha = 0.5f;
                float mMaxDistance = 25;
                float mInvLogMaxDist = 1.0 / (std::log(mAlpha * mMaxDistance + 1));
                cv::Point2f dp = nextPt - pred_pos_level;
                float d = std::sqrt(dp.x * dp.x + dp.y * dp.y);
                float e_penalty = mLambda * mInvLogMaxDist * std::log(mAlpha * d + 1);

                Eigen::VectorXf residual2(jacobian.rows() + 1);
                residual2.head(jacobian.rows()) = residual;
                residual2[jacobian.rows()] = 32.0*e_penalty;

                Eigen::VectorXi JPenalty(jacobian.cols());
                JPenalty.setZero();
                JPenalty.x() = cvRound(32*mLambda * mInvLogMaxDist * mAlpha / (mAlpha * d + 1) * (dp.x / d));
                JPenalty.y() = cvRound(32*mLambda * mInvLogMaxDist * mAlpha / (mAlpha * d + 1) * (dp.y / d));

                Eigen::MatrixXi ijacobian2(ijacobian.rows() + 1, ijacobian.cols());
                ijacobian2.setZero();
                ijacobian2.topRows(ijacobian.rows()) = ijacobian;
                ijacobian2.bottomRows<1>() = JPenalty.transpose();

                Eigen::MatrixXf jacobian2 = ijacobian2.cast<float>()*FLT_SCALE20;
                Eigen::MatrixXi ihessian2 = ijacobian2.transpose()*ijacobian2;
                Eigen::MatrixXf hessian2 = ihessian2.cast<float>()*FLT_SCALE20;
                Eigen::MatrixXf hessian_inv2 = hessian2.inverse();
                Eigen::MatrixXf hessian_inv_jacobian_t2 = hessian_inv2*jacobian2.transpose();

                delta = - hessian_inv_jacobian_t2*residual2;
            }
            else
            {
                delta = - hessian_inv_jacobian_t*residual;
            }

            nextPt.x += delta.x();
            nextPt.y += delta.y();
            cur_pt = nextPt;
            if(consider_illumination)
            {
                db += delta(2);
            }

            if(delta.head<2>().squaredNorm() <= 1e-4)
                break;

            if(j > 0 && std::abs(delta.x() + prevDelta.x()) < 0.01 &&
               std::abs(delta.y() + prevDelta.y()) < 0.01)
            {
                cur_pt.x -= delta.x()*0.5f;
                cur_pt.y -= delta.y()*0.5f;
                break;
            }
            prevDelta = delta;
        }

        if(status && level == 0)
        {
            if(OutBorder(cur_pt, level))
            {
                status = 0;
                error = 0;
                return;
            }
            cv::Point2i icur_pt;
            icur_pt.x = cvFloor(cur_pt.x);
            icur_pt.y = cvFloor(cur_pt.y);
            float aa = cur_pt.x - icur_pt.x;
            float bb = cur_pt.y - icur_pt.y;
            iw00 = cvRound((1.f - aa)*(1.f - bb)*(1 << W_BITS));
            iw01 = cvRound(aa*(1.f - bb)*(1 << W_BITS));
            iw10 = cvRound((1.f - aa)*bb*(1 << W_BITS));
            iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

            float errval = 0.f;
            idx = 0;
            for(int y = -half_patch_size; y <= half_patch_size; ++y)
            {
                const uchar* Jptr = dst_img_pyr[level].ptr(icur_pt.y + y) + icur_pt.x;
                for(int x = -half_patch_size; x <= half_patch_size; ++x)
                {
                    int cur_ival;
                    if(consider_affine)
                    {
                        cv::Point2f warp_pt = cur_pt + affine_warp[idx];
                        cur_ival = GetCurImgValue(warp_pt, level, stepI);
                    }
                    else
                    {
                        cur_ival = CV_DESCALE(Jptr[x]*iw00 + Jptr[x+1]*iw01 +
                                              Jptr[x+stepI]*iw10 + Jptr[x+stepI+1]*iw11,
                                              W_BITS1-5);
                    }
                    int diff = cvRound(cur_ival + 32*db - IWinBuf[idx]);
                    errval += std::abs((float)diff);
                    ++idx;
                }
            }
            error = errval * 1.f/(32*patch_area);
        }
    }
}

void PointFeature::AddNewestPoint(const int frame_id, const std::vector<cv::Mat>& img_pyr, cv::Point2f& newest_pt)
{
    last_src_frame_id = src_frame_id;
    last_src_corner = src_corner;
    last_src_undist_corner = src_undist_corner;

    src_frame_id = frame_id;
    src_corner = newest_pt;
    Eigen::Vector2d p(src_corner.x, src_corner.y);
    Eigen::Vector2d np = Norm2Pixel(Pixel2Norm(p), false);
    src_undist_corner = cv::Point2f(np.x(), np.y());

    history_corners[frame_id] = src_corner;
    history_undist_corners[frame_id] = src_undist_corner;

    const int W_BITS = 14, W_BITS1 = 14;

    for(int level = 0; level < pyramid_levels; ++level)
    {
        short* last_IWinBuf = last_src_patch[level].ptr<short>();
        short* IWinBuf = src_patch[level].ptr<short>();
        int idx = 0;
        int stepI = (int)(img_pyr[level].step/img_pyr[level].elemSize1());

        cv::Point2f newest_pt_level = newest_pt*(float)(1./(1 << level));
        cv::Point2i newest_pti_level;
        newest_pti_level.x = cvFloor(newest_pt_level.x);
        newest_pti_level.y = cvFloor(newest_pt_level.y);
        float a = newest_pt_level.x - newest_pti_level.x;
        float b = newest_pt_level.y - newest_pti_level.y;

        int iw00 = cvRound((1.f - a)*(1.f - b)*(1 << W_BITS));
        int iw01 = cvRound(a*(1.f - b)*(1 << W_BITS));
        int iw10 = cvRound((1.f - a)*b*(1 << W_BITS));
        int iw11 = (1 << W_BITS) - iw00 - iw01 - iw10;

        for(int y = -half_patch_size; y <= half_patch_size; ++y)
        {
            const uchar* src = img_pyr[level].ptr(newest_pti_level.y + y) + newest_pti_level.x;

            for(int x = -half_patch_size; x <= half_patch_size; ++x)
            {
                int ival = CV_DESCALE(src[x]*iw00 + src[x+1]*iw01 +
                                      src[x+stepI]*iw10 + src[x+stepI+1]*iw11, W_BITS1-5);
                last_IWinBuf[idx] = IWinBuf[idx];
                IWinBuf[idx++] = (short)ival;
            }
        }
    }

}

ImageProcessor::ImageProcessor(const std::string &yaml_path)
{
    cv::FileStorage fSettings(yaml_path, cv::FileStorage::READ);
    if(!fSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
        return;
    }

    image_data_path = std::string(fSettings["ImagePath"]);
    image_timestamp_path = std::string(fSettings["TimestampPath"]);
    std::vector<int> resolution;
    fSettings["resolution"] >> resolution;
    width = resolution[0];
    height = resolution[1];

    std::ifstream f;
    f.open(image_timestamp_path.c_str());
    image_file_names.reserve(5000);
    image_timestamps.reserve(5000);
    while(!f.eof())
    {
        std::string s;
        getline(f, s);
        if(!s.empty())
        {
            if(s.front() == '#') continue;
            std::stringstream ss;
            ss << s;
            std::string s2;
            getline(ss, s2, ',');
            image_file_names.push_back(image_data_path + "/" + s2 + ".png");
            ss << s2;
            io::timestamp_t t;
            ss >> t;
            image_timestamps.push_back(t);
        }
    }
    image_data_size = image_file_names.size();
    std::cout << " read " << image_data_size << " images !!!" << std::endl;

//    int apply_mask = fSettings["apply_mask"];
//    if(apply_mask)
//        mask_template = cv::imread(std::string(DATA_DIR) + "mask.jpg", 0);
//    else
//        mask_template = cv::Mat(height, width, CV_8UC1, cv::Scalar(255));

    cur_img_pyr.resize(pyramid_levels);
    cur_img_dx_pyr.resize(pyramid_levels);
    cur_img_dy_pyr.resize(pyramid_levels);

    last_keyframe_id = -1;
    global_feature_id = 0;
}

cv::Mat ImageProcessor::GetImage(int id, bool use_clahe)
{
    if(id < 0 || id >= image_data_size)
        std::cerr << "GetImage error !!!" << std::endl;
    cv::Mat imgOrg = cv::imread(image_file_names[id], CV_LOAD_IMAGE_GRAYSCALE);

    if(use_clahe)
    {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        clahe->apply(imgOrg, imgOrg);
    }
    return imgOrg;
}


std::map<int, Eigen::Vector2d> ImageProcessor::GetTrackedFeatures(
        const int id, const cv::Mat& origin_image, bool& is_keyframe, std::map<int, Eigen::Isometry3d>& poses_list, std::map<int, Eigen::Vector3f>& mappoint_list)
{
    std::map<int, Eigen::Vector2d> frame_tracked_points;
    cur_img_pyr.clear();
    cv::Size sz = origin_image.size();
    int winsize = 2*half_patch_size+1;
    for(int level = 0; level < pyramid_levels; ++level)
    {
        cv::Mat temp;
        temp.create(sz.height + winsize*2, sz.width + winsize*2, origin_image.type());
        if(level == 0)
            cv::copyMakeBorder(origin_image, temp, winsize, winsize, winsize, winsize, cv::BORDER_REFLECT_101);
        else
        {
            cv::Mat thisLevel = temp(cv::Rect(winsize, winsize, sz.width, sz.height));
            cv::pyrDown(cur_img_pyr[level-1], thisLevel, sz);
            cv::copyMakeBorder(thisLevel, temp, winsize, winsize, winsize, winsize, cv::BORDER_REFLECT_101|cv::BORDER_ISOLATED);
        }
        temp.adjustROI(-winsize, -winsize, -winsize, -winsize);
        cur_img_pyr.push_back(temp);
        sz = cv::Size((sz.width+1)/2, (sz.height+1)/2);
    }

    cur_img_dx_pyr.clear();
    cur_img_dy_pyr.clear();
    for(int i = 0; i < pyramid_levels; ++i)
    {
        cv::Mat derivX, derivY;
        cv::Scharr(cur_img_pyr[i], derivX, CV_16S, 1, 0);
        cv::Scharr(cur_img_pyr[i], derivY, CV_16S, 0, 1);
        cv::copyMakeBorder(derivX, derivX, winsize, winsize, winsize, winsize, cv::BORDER_CONSTANT);
        cv::copyMakeBorder(derivY, derivY, winsize, winsize, winsize, winsize, cv::BORDER_CONSTANT);

        derivX.adjustROI(-winsize, -winsize, -winsize, -winsize);
        derivY.adjustROI(-winsize, -winsize, -winsize, -winsize);

        cur_img_dx_pyr.push_back(derivX);
        cur_img_dy_pyr.push_back(derivY);
    }

    std::vector<cv::Point2f> src_tracked_points;
    std::vector<cv::Point2f> dst_tracked_points;
    std::vector<int> tracked_feature_ids;
    std::vector<uchar> track_inliers;
    if(!tracking_points.empty())
    {
        if(!imu_aided_track)
        {
            src_tracked_points.resize(tracking_points.size());
            dst_tracked_points.resize(tracking_points.size());
            tracked_feature_ids.resize(tracking_points.size());
            track_inliers.resize(tracking_points.size(), 1);

            std::vector<float> track_errors(tracking_points.size(), 0);
            std::transform(tracking_points.begin(), tracking_points.end(), src_tracked_points.begin(),
                           [](std::shared_ptr<PointFeature>& tracking_point){return tracking_point->src_corner;});
            dst_tracked_points = src_tracked_points;

            std::vector<float> err;
            cv::calcOpticalFlowPyrLK(pre_img_pyr[0], cur_img_pyr[0], src_tracked_points, dst_tracked_points, track_inliers, err);
            for(int i = 0, iend = dst_tracked_points.size(); i < iend; ++i)// 追踪的点赋值
            {
                tracked_feature_ids[i] = tracking_points[i]->feature_id;
            }
        }
        else
        {
            src_tracked_points.resize(tracking_points.size());
            dst_tracked_points.resize(tracking_points.size());
            tracked_feature_ids.resize(tracking_points.size());
            track_inliers.resize(tracking_points.size(), 1);

            std::vector<float> track_errors(tracking_points.size(), 0);
            std::transform(tracking_points.begin(), tracking_points.end(), src_tracked_points.begin(),
                           [](std::shared_ptr<PointFeature>& tracking_point){return tracking_point->src_corner;});
            dst_tracked_points = src_tracked_points;
            cv::parallel_for_(cv::Range(0, tracking_points.size()), [&](const cv::Range& range)
            {
                for(auto i = range.start; i < range.end; ++i)
                {
                    int frame_i_id = tracking_points[i]->src_frame_id;
                    Eigen::Isometry3d Tji;
                    if(poses_list.count(id) && poses_list.count(frame_i_id))
                    {
                        Tji = poses_list[id].inverse()*poses_list[frame_i_id];
                        int feature_id = tracking_points[i]->feature_id;
                        if(mappoint_list.count(feature_id)&&false)
                        {
                            Eigen::Vector3f position = mappoint_list[feature_id];
                            Eigen::Matrix3f Rwi = poses_list[frame_i_id].linear().cast<float>();
                            Eigen::Vector3f twi = poses_list[frame_i_id].translation().cast<float>();
                            float inv_dep = 1.0f / (Rwi.transpose()*(position - twi)).z();
                            if(inv_dep > 0.01 && inv_dep < 10)
                            {
                                tracking_points[i]->inv_dep = inv_dep;
                                tracking_points[i]->solve_flag = true;
                            }
                            else
                                tracking_points[i]->solve_flag = false;
                        }
                    }
                    else
                    {
                        Tji = Eigen::Isometry3d::Identity();
                        tracking_points[i]->solve_flag = false;
                    }

                    tracking_points[i]->TrackPatch(cur_img_pyr, cur_img_dx_pyr, cur_img_dy_pyr,
                                                   dst_tracked_points[i], track_inliers[i], track_errors[i], Tji,
                                                   false);

                    tracked_feature_ids[i] = tracking_points[i]->feature_id;
                }
            });
        }
    }

    if(!dst_tracked_points.empty())
    {
        std::vector<cv::Point2f> ref_pts, cur_pts;
        std::vector<int> src_frame_ids;

        std::map<int, int> fundamental_to_track;
        for(int i = 0; i < dst_tracked_points.size(); ++i)
        {
            if(track_inliers[i])
            {
                ref_pts.push_back(tracking_points[i]->src_undist_corner);
                Eigen::Vector2d p(dst_tracked_points[i].x, dst_tracked_points[i].y);
                Eigen::Vector2d np = Norm2Pixel(Pixel2Norm(p), false);
                cur_pts.push_back(cv::Point2f(np.x(), np.y()));
                fundamental_to_track[int(ref_pts.size() - 1)] = i;
                src_frame_ids.push_back(tracking_points[i]->src_frame_id);
            }
        }
        if(ref_pts.size() > 15)
        {
            std::vector<uchar> fundamental_inliers;
            cv::findFundamentalMat(ref_pts, cur_pts, cv::FM_RANSAC, fundamental_ransac_threshold, 0.99, fundamental_inliers);
            for(int i = 0; i < fundamental_inliers.size(); ++i)
            {
                if(!fundamental_inliers[i])
                {
                    track_inliers[fundamental_to_track[i]] = 0;
                }
            }
        }
    }

    int success_num = 0;
    for(int i = 0; i < dst_tracked_points.size(); ++i)
    {
        if(track_inliers[i])
        {
            ++success_num;
            Eigen::Vector2d p;
            p << dst_tracked_points[i].x, dst_tracked_points[i].y;
            frame_tracked_points[tracked_feature_ids[i]] = p;
        }
    }

    is_keyframe = true;
    if(!dst_tracked_points.empty())
    {
        float ave_parallax = 0.f;
        int inliers_size = 0;
        for(int i = 0; i < dst_tracked_points.size(); ++i)
        {
            if(!track_inliers[i]) continue;
            if(tracking_points[i]->src_frame_id != keyframe_ids.back()) continue;

            cv::Point2f dp = dst_tracked_points[i] - src_tracked_points[i];
            ave_parallax += std::sqrt(dp.x*dp.x + dp.y*dp.y);
            ++inliers_size;
        }
        ave_parallax /= float(inliers_size);
        if(ave_parallax > keyframe_ave_parallax)
            is_keyframe = true;
        else if(frame_tracked_points.size() < keyframe_tracked_num)
            is_keyframe = true;
        else
            is_keyframe = false;
    }


    if(is_keyframe)
    {
        if(keyframe_ids.size() < 1)
            keyframe_ids.push_back(id);
        else
        {
            int mov_id = keyframe_ids[0];
            for(auto &tracking_point : tracking_points)
            {
                if(tracking_point->history_corners.count(mov_id))
                {
                    tracking_point->history_corners.erase(mov_id);
                    tracking_point->history_undist_corners.erase(mov_id);
                }
            }

            for(int i = 0; i < keyframe_ids.size() - 1; ++i)
            {
                keyframe_ids[i] = keyframe_ids[i + 1];
            }
            keyframe_ids.back() = id;
        }

        {
            std::vector<cv::Point2f> mask_points;
            std::map<int, int> mask_to_track;
            std::vector<uchar> mask_inliners;
            for(size_t i = 0; i < dst_tracked_points.size(); ++i)
            {
                if(track_inliers[i] == 1)
                {
                    mask_points.push_back(dst_tracked_points[i]);
                    mask_to_track[int(mask_points.size() - 1)] = i;
                }
            }

            std::vector<cv::Point2f> new_points;

            GoodFeaturesToTrackProShortWithTrackedPoints(cur_img_dx_pyr[0], cur_img_dy_pyr[0], new_points,
                    max_feature_num, 0.01, 5, mask_points, mask_inliners);

            for(int i = 0; i < mask_inliners.size(); ++i)
            {
                if(!mask_inliners[i])
                {
                    track_inliers[mask_to_track[i]] = 0;
                }
            }

            cv::parallel_for_(cv::Range(0, tracking_points.size()), [&](const cv::Range& range)
            {
                for(auto i = range.start; i < range.end; ++i)
                {
                    if(track_inliers[i] == 1)
                    {
                        tracking_points[i]->AddNewestPoint(id, cur_img_pyr, dst_tracked_points[i]);
                    }
                }
            });
            ReduceVector(tracking_points, track_inliers);

            for(size_t i = 0; i < new_points.size(); ++i)
            {
                std::shared_ptr<PointFeature> new_point = std::make_shared<PointFeature>(global_feature_id, id, new_points[i], cur_img_pyr);

                tracking_points.push_back(new_point);
                Eigen::Vector2d p;
                p << new_points[i].x, new_points[i].y;
                frame_tracked_points[global_feature_id] = p;

                ++global_feature_id;
            }
        }
        pre_img_pyr = cur_img_pyr;
    }
    return frame_tracked_points;
}

io::timestamp_t ImageProcessor::GetTimestamp(int id)
{
    if(id < 0 || id >= image_data_size)
        std::cerr << "GetImageTimestamp error !!!" << std::endl;
    return image_timestamps[id];
}

