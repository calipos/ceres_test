#include <tuple> 
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include "opencv2/opencv.hpp"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#define RAND_UNIT (rand() % 10000 / 5000. - 1.)
#define RAND_SCALE (0)
// Read a Bundle Adjustment in the Large dataset.
//7 parameter about the cameraExtra [r1,r2,r3 angle t1,t2,t3]
//5 parameter about the cameraInner [f,l1,l2  cx,cy]
//the rotationAxis has 3 parameters [r1_,r2_,r3_ ]
class BALProblem {
public:
    ~BALProblem() 
    { 
        //delete[] observations_;
        //delete[] parameters_;
        //delete[] observations_comeFrom_;
    }
    int initial_camera_init_r(double a, double b, double c)
    {
        parameters_[0] = a;
        parameters_[1] = b;
        parameters_[2] = c;
        return 0;
    }
    int initial_camera_init_r_angle(double a)
    {
        parameters_[3] = a;
        return 0;
    }
    int initial_camera_init_t(double a, double b, double c)
    {
        parameters_[4] = a;
        parameters_[5] = b;
        parameters_[6] = c;
        return 0;
    }
    int initial_F(double f)
    {
        parameters_[7] = f;
        return 0;
    }
    int initial_l1l2(double l1, double l2)
    {
        parameters_[8] = l1;
        parameters_[9] = l2;
        return 0;
    }
    int initial_cxcy(double cx, double cy)
    {
        parameters_[10] = cx;
        parameters_[11] = cy;
        return 0;
    }
    int initial_axis_r(double a, double b, double c)
    {
        parameters_[12] = a;
        parameters_[13] = b;
        parameters_[14] = c;
        return 0;
    }
    void showParam()
    {
        LOG(INFO) << "camera init dir = [" << parameters_[0] << " " << parameters_[1] << " " << parameters_[2] << "]";
        LOG(INFO) << "camera init r_angle = [" << parameters_[3] << "]";
        LOG(INFO) << "camera init pos = [" << parameters_[4] << " " << parameters_[5] << " " << parameters_[6] << "]";
        LOG(INFO) << "camera [f l1 l2] = [" << parameters_[7] << " " << parameters_[8] << " " << parameters_[9] << "]";
        LOG(INFO) << "camera [cx cy] = [" << parameters_[10] << " " << parameters_[11] << "]";
        LOG(INFO) << "axis = [" << parameters_[12] << " " << parameters_[13] << " " << parameters_[14] << "]";
    }
    int num_observations() const { return pairPts.size()*2; }
    int* observations_comeFrom() const{ return observations_comeFrom_; }
    const double* observations() const { return observations_; }
    double* mutable_cameraExtra() { return parameters_; }
    double* mutable_cameraInner() { return parameters_+7; }
    double* mutable_axis() { return parameters_+12; } 
    double* mutable_point_for_observation(int i) {
        return parameters_ + 15 + i * 3;
    }

    bool LoadFile(const std::string&dirPath,const int K)
    {
        if (dirPath[dirPath.length() - 1] == '\\'|| dirPath[dirPath.length() - 1] == '/')
        {
            dirPath_ = dirPath;
        }
        else
        {
            dirPath_ = dirPath + "/";
        }
        for (int i = 0; i < K; i++)
        {
            int j = i == K - 1 ? 0 : i + 1;
            std::fstream fin(dirPath_ +std::to_string(i)+".txt",std::ios::in);
            std::string aline;
            int ptsIdx = 0;
            double x1, y1, x2, y2;
            double gtX, gtY, gtZ;
            int thisFilePairPtsNum = 0;
            while (std::getline(fin, aline))
            {
                thisFilePairPtsNum++;
                std::stringstream ss(aline);
                ss >> ptsIdx >> x1 >> y1 >> x2 >> y2>> gtX>> gtY>> gtZ;
                pairPts.emplace_back(std::make_tuple(i, j, x1, y1, x2, y2,cv::Point3d(gtX, gtY, gtZ)));
            }
            fin.close();
            CHECK(thisFilePairPtsNum>0)<<i<<" and "<<j<<" has no shared points! ";
        }
        //assume a certain point appears only in  i and j , and another img can see the point!!!
        // there is  one camera but capture K imgs around.
        //11 parameter about the camera [r1,r2,r3  t1,t2,t3  f,l1,l2  cx,cy]
        //the rotationAxis has 3 parameters [r1_,r2_,r3_ ]
        num_parameters_ = 11 +  3 + 3 * pairPts.size();
        parameters_ = new double[num_parameters_];
        for (int i = 0; i < num_parameters_; i++)
        {
            parameters_[i] = rand() % 10000 / 5000. - 1.;
        }
        for (int i = 0; i < pairPts.size(); i++)
        {

            parameters_[15 + 3 * i] = std::get<6>(pairPts[i]).x+ RAND_UNIT* RAND_SCALE;
            parameters_[15 + 3 * i + 1] = std::get<6>(pairPts[i]).y + RAND_UNIT * RAND_SCALE;
            parameters_[15 + 3 * i + 2] = std::get<6>(pairPts[i]).z + RAND_UNIT * RAND_SCALE;
        } 
        observations_ = new double[2 *2* pairPts.size()];
        observations_comeFrom_ = new int[2 * pairPts.size()];
        for (int i = 0; i < pairPts.size(); i++)
        {
            const int& from_i = std::get<0>(pairPts[i]);
            const int& from_j = std::get<1>(pairPts[i]);
            double obsserved_i_x = std::get<2>(pairPts[i]);
            double obsserved_i_y = std::get<3>(pairPts[i]);
            double obsserved_j_x = std::get<4>(pairPts[i]);
            double obsserved_j_y = std::get<5>(pairPts[i]);
            observations_comeFrom_[2 * i] = from_i;
            observations_comeFrom_[2 * i + 1] = from_j;
            observations_[4 * i] = obsserved_i_x;
            observations_[4 * i + 1] = obsserved_i_y;
            observations_[4 * i + 2] = obsserved_j_x;
            observations_[4 * i + 3] = obsserved_j_y;
        }
        return true;
    }
    BALProblem()
    {
        pairPts.clear();
        num_parameters_ = 0;
        observations_comeFrom_ = NULL;
        observations_ = NULL;
        parameters_ = NULL;
    }
//private:
    std::vector<std::tuple<int, int, double, double, double, double,cv::Point3d>> pairPts;
    std::string dirPath_;      
    int num_parameters_;     
    int* observations_comeFrom_;
    double* observations_;
    double* parameters_;
};
 
struct SnavelyReprojectionError 
{
    SnavelyReprojectionError(double rotationAngle,double observed_x, double observed_y)
        :rotationAngle(rotationAngle), observed_x(observed_x), observed_y(observed_y) {}

    template <typename T>
    bool operator()(
        const T* const cameraExtra,
        const T* const cameraInner,
        const T* const axis,
        const T* const point,
        T* residuals) const 
    { 
        //cameraExtra[0:3] initial camera dir
        //cameraExtra[3:6] initial camera pos
        //cameraInner[0:5] f,l1,l2  cx,cy
        //axis[0:3] r1_,r2_,r3_ 
        T thisCameraPos[3];
        T thisAxis_r[3];   
        T axis_normal = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        thisAxis_r[0] = rotationAngle * axis[0] / (axis_normal);
        thisAxis_r[1] = rotationAngle * axis[1] / (axis_normal);
        thisAxis_r[2] = rotationAngle * axis[2] / (axis_normal);
        ceres::AngleAxisRotatePoint(thisAxis_r, cameraExtra +4, thisCameraPos);
        T camera_x[3] = { static_cast<T>(1),static_cast<T>(0),static_cast<T>(0) };
        T camera_y[3] = { static_cast<T>(0),static_cast<T>(1),static_cast<T>(0) };
        T camera_z[3] = { static_cast<T>(0),static_cast<T>(0),static_cast<T>(1) };
        T camera_x2[3];
        T camera_y2[3];
        T camera_z2[3]; 

        T camera_r_normal = cameraExtra[0] * cameraExtra[0] + cameraExtra[1] * cameraExtra[1] + cameraExtra[2] * cameraExtra[2];
        T thisCameraR[3];
        thisCameraR[0] = cameraExtra[3] * cameraExtra[0] / sqrt(camera_r_normal);
        thisCameraR[1] = cameraExtra[3] * cameraExtra[1] / sqrt(camera_r_normal);
        thisCameraR[2] = cameraExtra[3] * cameraExtra[2] / sqrt(camera_r_normal);
        ceres::AngleAxisRotatePoint(thisCameraR, camera_x, camera_x2);
        ceres::AngleAxisRotatePoint(thisCameraR, camera_y, camera_y2);
        ceres::AngleAxisRotatePoint(thisCameraR, camera_z, camera_z2);
        T camera_x3[3];
        T camera_y3[3];
        T camera_z3[3];
        ceres::AngleAxisRotatePoint(thisAxis_r, camera_x2, camera_x3);
        ceres::AngleAxisRotatePoint(thisAxis_r, camera_y2, camera_y3);
        ceres::AngleAxisRotatePoint(thisAxis_r, camera_z2, camera_z3);
        T p0[3];
        p0[0] = point[0] - thisCameraPos[0];
        p0[1] = point[1] - thisCameraPos[1];
        p0[2] = point[2] - thisCameraPos[2];
        T p1[3];
        p1[0] = p0[0] * camera_x3[0] + p0[1] * camera_x3[1] + p0[2] * camera_x3[2];
        p1[1] = p0[0] * camera_y3[0] + p0[1] * camera_y3[1] + p0[2] * camera_y3[2];
        p1[2] = p0[0] * camera_z3[0] + p0[1] * camera_z3[1] + p0[2] * camera_z3[2];
        T xp = p1[0] / p1[1];
        T yp = p1[2] / p1[1];
        const T& l1 = cameraInner[1];
        const T& l2 = cameraInner[2];
        T r2 = xp * xp + yp * yp;
        T distortion = 1.0 + r2 * (l1 + l2 * r2);         
        const T& focal = cameraInner[0];
        T predicted_x = focal * distortion * xp + cameraInner[3];
        T predicted_y = cameraInner[4]-focal * distortion * yp  ;
        residuals[0] = predicted_x - observed_x;
        residuals[1] = predicted_y - observed_y;
        LOG(INFO) << "("<<residuals[0] << "  " << residuals[1]<<")";
        return true;
    } 

    static ceres::CostFunction* Create(const double angle,
        const double observed_x,
        const double observed_y)
    {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 7, 5, 3, 3>(
            new SnavelyReprojectionError(angle, observed_x, observed_y)));
    };

    double rotationAngle;
    double observed_x;
    double observed_y;
};

int rotationCamera()
{
    double axis_test[3] = { 1, 2, 3 };
    double p_test[3] = { 4, 5, 6 };
    double p2_test[3] = { 1, 2, 3 };
    ceres::AngleAxisRotatePoint(axis_test, p_test, p2_test);


    srand((unsigned)time(NULL));
    int K = 90;
    double eachKangle = -CV_2PI / K;
    BALProblem bal_problem;
    bal_problem.LoadFile("D:/ucl360/UCL360Calib/MultiStation", K);
    const int* observations_comeFrom = bal_problem.observations_comeFrom();
    const double* observations = bal_problem.observations();
#define LITTLE  (RAND_UNIT * RAND_SCALE)
    bal_problem.initial_camera_init_r(-0.02+ LITTLE,    0.01 + LITTLE,    1 + LITTLE);
    bal_problem.initial_camera_init_r_angle(CV_PI/12 + LITTLE*10);
    bal_problem.initial_camera_init_t(0.08 + LITTLE, 0.001 + LITTLE, 0.008 + LITTLE);
    bal_problem.initial_cxcy(240 + LITTLE*10, 160 + LITTLE*10);
    bal_problem.initial_F(250 + LITTLE*10);
    bal_problem.initial_l1l2( 0 + LITTLE, 0 + LITTLE);
    bal_problem.initial_axis_r(0.01 + LITTLE, 0.02 + LITTLE, 1 + LITTLE);
    //bal_problem.showParam();
    LOG(INFO) << "bal_problem.pairPts.size() = "<<bal_problem.pairPts.size();
        ceres::Problem problem;
        for (int i = 0; i < bal_problem.pairPts.size(); ++i)
        {
            ceres::CostFunction* cost_function1 = SnavelyReprojectionError::Create(
                eachKangle * std::get<0>(bal_problem.pairPts[i]), std::get<2>(bal_problem.pairPts[i]), std::get<3>(bal_problem.pairPts[i])); 
            problem.AddResidualBlock(cost_function1,
                NULL ,
                bal_problem.mutable_cameraExtra(),
                bal_problem.mutable_cameraInner(),
                bal_problem.mutable_axis(),
                bal_problem.mutable_point_for_observation(i));
            ceres::CostFunction* cost_function2 = SnavelyReprojectionError::Create(
                eachKangle * std::get<1>(bal_problem.pairPts[i]), std::get<4>(bal_problem.pairPts[i]), std::get<5>(bal_problem.pairPts[i]));
            problem.AddResidualBlock(cost_function2,
                NULL,
                bal_problem.mutable_cameraExtra(),
                bal_problem.mutable_cameraInner(),
                bal_problem.mutable_axis(),
                bal_problem.mutable_point_for_observation(i));
            break;
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << "\n";
        bal_problem.showParam();
    
    return 0;
}
