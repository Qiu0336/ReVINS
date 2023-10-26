
#include "drawer.h"

Drawer::Drawer()
{
    SetQuit(false);

    mBackgroundpatchsize = 2;
    mBackgroundpatchcount = 5;
    mPointSize = 8;
    mLineSize = 3;
    mCameraSize = 0.5;
    mCameraLineWidth = 3;
}

void Drawer::DrawBackground()
{
    int z = mBackgroundpatchcount;
    float w = mBackgroundpatchsize;
    float edge = z*w;
    float x;
    glLineWidth(mCameraLineWidth);
    glColor3f(0.0f, 0.0f, 0.0f);
    glBegin(GL_LINES);
    for(int i = 0; i <= z; i++)
    {
        x = w*i;
        glVertex3f(x, edge, 0);
        glVertex3f(x, -edge, 0);

        glVertex3f(-x, edge, 0);
        glVertex3f(-x, -edge, 0);

        glVertex3f(edge, x, 0);
        glVertex3f(-edge, x, 0);

        glVertex3f(edge, -x, 0);
        glVertex3f(-edge, -x, 0);
    }
    glEnd();
}


void Drawer::DrawCamera(bool use_loop)
{
    pangolin::OpenGlMatrix Twc;
    Twc.SetIdentity();
    Eigen::Isometry3d camera_pose = estimator_ptr->GetCameraPose();
    if(use_loop)
        camera_pose = posegraph_ptr->DriftRemove(camera_pose);
    Eigen::Matrix3d R = camera_pose.rotation();
    Eigen::Vector3d t = camera_pose.translation();
    Twc.m[0] = R(0, 0);
    Twc.m[1] = R(1, 0);
    Twc.m[2] = R(2, 0);
    Twc.m[3] = 0.0;

    Twc.m[4] = R(0, 1);
    Twc.m[5] = R(1, 1);
    Twc.m[6] = R(2, 1);
    Twc.m[7] = 0.0;

    Twc.m[8] = R(0, 2);
    Twc.m[9] = R(1, 2);
    Twc.m[10] = R(2, 2);
    Twc.m[11] = 0.0;

    Twc.m[12] = t(0);
    Twc.m[13] = t(1);
    Twc.m[14] = t(2);
    Twc.m[15] = 1.0;

    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();
    glMultMatrixd(Twc.m);

    glLineWidth(mCameraLineWidth);
    if(use_loop)
        glColor3f(1.0f, 0.0f, 0.0f);
    else
        glColor3f(1.0f, 0.0f, 1.0f);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(w, h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, -h, z);
    glVertex3f(0, 0, 0);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();

    glPopMatrix();
}

void Drawer::DrawTrajectory(bool use_loop)
{
    const auto& keyframe_list = posegraph_ptr->GetKeyframelist();
    if(keyframe_list.size() == 0)
        return;
    const int ptsize = mPointSize;
    glPointSize(ptsize/2);//点的大小
    glLineWidth(mLineSize);
    glBegin(GL_LINES);
    int trajsize = keyframe_list.size();
    Eigen::Vector3d pos1;
    Eigen::Vector3d pos2;
    Eigen::Isometry3d camera_pose = estimator_ptr->GetCameraPose();

    if(use_loop)
    {
        glColor3f(0.0, 0.0, 1.0);
        pos1 = keyframe_list[0]->Twc_update.translation();
        for(int i = 1; i < trajsize; ++i)
        {
            pos2 = keyframe_list[i]->Twc_update.translation();
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            if(keyframe_list[i]->loop_index >= 0)
            {
                glColor3f(1.0, 0.0, 0.0);
                int lp = keyframe_list[i]->loop_index;
                Eigen::Vector3d posloop = keyframe_list[lp]->Twc_update.translation();
                glVertex3f(posloop(0), posloop(1), posloop(2));
                glVertex3f(pos2(0), pos2(1), pos2(2));

                glColor3f(0.0, 0.0, 1.0);
            }
            pos1 = pos2;
        }

        glVertex3f(pos2(0), pos2(1), pos2(2));
        camera_pose = posegraph_ptr->DriftRemove(camera_pose);
        glVertex3f(camera_pose.translation().x(), camera_pose.translation().y(), camera_pose.translation().z());

    }

    else
    {
        glColor3f(1.0, 0.0, 0.0);
        pos1 = keyframe_list[0]->Twc.translation();
        for(int i = 1; i < trajsize; ++i)
        {
            pos2 = keyframe_list[i]->Twc.translation();
            glVertex3f(pos1(0), pos1(1), pos1(2));
            glVertex3f(pos2(0), pos2(1), pos2(2));
            pos1 = pos2;
        }
        glVertex3f(pos2(0), pos2(1), pos2(2));
        camera_pose = estimator_ptr->GetCameraPose();
        glVertex3f(camera_pose.translation().x(), camera_pose.translation().y(), camera_pose.translation().z());

    }
    glEnd();
}

void Drawer::Run()
{
    pangolin::CreateWindowAndBind("Main", 640, 480);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(640, 480, 420, 420, 320, 320, 0.2, 500),
                pangolin::ModelViewLookAt(-2,0,2, 0,0,0, pangolin::AxisZ)
                );

    pangolin::View &d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    while(!pangolin::ShouldQuit() && !GetQuit())
    {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

//        DrawBackground();
        DrawCamera(true);
//        DrawCamera(false);
        DrawTrajectory(true);
//        DrawTrajectory(false);

        pangolin::FinishFrame();
    }
}

void Drawer::SetQuit(bool x)
{
    std::unique_lock<std::mutex> lock(m_quit);
    quit_flag = x;
}

bool Drawer::GetQuit()
{
    std::unique_lock<std::mutex> lock(m_quit);
    return quit_flag;
}

