#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <fstream>
#include "ceres/ceres.h"
#include "glog/logging.h"
     
#define RandUnit (rand()%10000/5000.-1)
#define UseAutoDiff 0

#define  PI (3.14159265359)
struct dataGenterator
{
    dataGenterator(const double& centerX, const double& centerY, const double& angle1, const double& angle2, const double& angle3)
    {
        centerX_ = centerX;
        centerY_ = centerY;
        angle1_ = angle1;
        angle2_ = angle2;
        angle3_ = angle3;
    }
    std::vector<double> gener(const int&cnt)
    {
        std::vector<double>ret(6*cnt);
        for (size_t i = 0; i < cnt; i++)
        {
            double length1 = abs(RandUnit) + 0.5;
            double length2 = abs(RandUnit) + 0.5;
            double length3 = abs(RandUnit) + 0.5;
            ret[6 * i + 0] = centerX_ + length1 * cos(angle1_);
            ret[6 * i + 1] = centerY_ + length1 * sin(angle1_);
            ret[6 * i + 2] = centerX_ + length2 * cos(angle2_);
            ret[6 * i + 3] = centerY_ + length2 * sin(angle2_);
            ret[6 * i + 4] = centerX_ + length3 * cos(angle3_);
            ret[6 * i + 5] = centerY_ + length3 * sin(angle3_);
        }
        return ret;
    };
    double centerX_; 
    double centerY_;
    double  angle1_, angle2_, angle3_;
};

void writeData(const std::string& path, const std::vector<double>& tripleData)
{
    std::fstream fout(path,std::ios::out);
    for (size_t i = 0; i < tripleData.size()/2; i++)
    {
        fout << tripleData[2 * i] << " " << tripleData[2 * i + 1] << " 0" << std::endl;
    }
    fout . close();
}


struct Residual {
    Residual(double *xyz) : xyz_(xyz) {}

    template <typename T>
    bool operator()(const T* const center, const T* const angles, T* residual) const
    {
        const double& x1 = xyz_[0];
        const double& y1 = xyz_[1];
        const double& x2 = xyz_[2];
        const double& y2 = xyz_[3];
        const double& x3 = xyz_[4];
        const double& y3 = xyz_[5];
        T centerX = (center[0]);
        T centerY = (center[1]); 
        T anglePt_x1 = cos(angles[0])+centerX;
        T anglePt_y1 = sin(angles[0])+centerY;
        T anglePt_x2 = cos(angles[1])+centerX;
        T anglePt_y2 = sin(angles[1])+centerY;
        T anglePt_x3 = cos(angles[2])+centerX;
        T anglePt_y3 = sin(angles[2])+centerY;
        T dist11 = x1 - centerX;
        T dist12 = y1 - centerY;
        T length1 = sqrt(dist11 * dist11 + dist12 * dist12);
        T dist13 = x2 - centerX;
        T dist14 = y2 - centerY;
        T length2 = sqrt(dist13 * dist13 + dist14 * dist14);
        T dist15 = x3 - centerX;
        T dist16 = y3 - centerY;
        T length3 = sqrt(dist15 * dist15 + dist16 * dist16);
        T dist21 = centerX + (x1 - centerX) / length1 - anglePt_x1;
        T dist22 = centerY + (y1 - centerY) / length1 - anglePt_y1;
        T dist23 = centerX + (x2 - centerX) / length2 - anglePt_x2;
        T dist24 = centerY + (y2 - centerY) / length2 - anglePt_y2;
        T dist25 = centerX + (x3 - centerX) / length3 - anglePt_x3;
        T dist26 = centerY + (y3 - centerY) / length3 - anglePt_y3;

        residual[0] = dist21 * dist21 + dist22 * dist22;
        residual[1] = dist23 * dist23 + dist24 * dist24;
        residual[2] = dist25 * dist25 + dist26 * dist26;
        return true;
    } 
private:
    const double* xyz_;
};
struct Residual2 :public ceres::SizedCostFunction<3, 2,3 >
{
    Residual2(double* xyz) : xyz_(xyz) {}
    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const 
    { 
        const double& x1 = xyz_[0];
        const double& y1 = xyz_[1];
        const double& x2 = xyz_[2];
        const double& y2 = xyz_[3];
        const double& x3 = xyz_[4];
        const double& y3 = xyz_[5];
        double centerX = (parameters[0][0]);
        double centerY = (parameters[0][1]);
        double anglePt_x1 = cos(parameters[1][0]);
        double anglePt_y1 = sin(parameters[1][0]);
        double anglePt_x2 = cos(parameters[1][1]);
        double anglePt_y2 = sin(parameters[1][1]);
        double anglePt_x3 = cos(parameters[1][2]);
        double anglePt_y3 = sin(parameters[1][2]);
        double x1_xc = x1 - centerX;
        double y1_yc = y1 - centerY;
        double length1 = sqrt(x1_xc * x1_xc + y1_yc * y1_yc);
        double x2_xc = x2 - centerX;
        double y2_yc = y2 - centerY;
        double length2 = sqrt(x2_xc * x2_xc + y2_yc * y2_yc);
        double x3_xc = x3 - centerX;
        double y3_yc = y3 - centerY;
        double length3 = sqrt(x3_xc * x3_xc + y3_yc * y3_yc);
        double dist21 =  (x1_xc) / length1 - anglePt_x1;
        double dist22 =  (y1_yc) / length1 - anglePt_y1;
        double dist23 =  (x2_xc) / length2 - anglePt_x2;
        double dist24 =  (y2_yc) / length2 - anglePt_y2;
        double dist25 =  (x3_xc) / length3 - anglePt_x3;
        double dist26 =  (y3_yc) / length3 - anglePt_y3;
#define USE_SECOND_RASID 1
#if USE_SECOND_RASID>0
#define SECOND_WEIGHT (0.1)
        residuals[0] = dist21 * dist21 + dist22 * dist22 + SECOND_WEIGHT*(x1_xc * x1_xc + y1_yc * y1_yc);
        residuals[1] = dist23 * dist23 + dist24 * dist24 + SECOND_WEIGHT*(x2_xc * x2_xc + y2_yc * y2_yc);
        residuals[2] = dist25 * dist25 + dist26 * dist26 + SECOND_WEIGHT*(x3_xc * x3_xc + y3_yc * y3_yc);
#else
        residuals[0] = dist21 * dist21 + dist22 * dist22; ;
        residuals[1] = dist23 * dist23 + dist24 * dist24;;
        residuals[2] = dist25 * dist25 + dist26 * dist26;;
#endif // USE_SECOND_RASID>0


        if (jacobians != NULL && jacobians[0] != NULL)
        {
            //---------------------------------------------------------------------------------------------- 
#if USE_SECOND_RASID>0
            jacobians[0][0] = 2. * dist21 * (x1_xc * x1_xc - length1 * length1) / length1 / length1 / length1
                + 2. * dist22 * x1_xc * y1_yc / length1 / length1 / length1 + SECOND_WEIGHT * 2. * x1_xc;
            jacobians[0][1] = 2. * dist22 * (y1_yc * y1_yc - length1 * length1) / length1 / length1 / length1
                + 2. * dist21 * x1_xc * y1_yc / length1 / length1 / length1 + SECOND_WEIGHT * 2. * y1_yc;
#else
            jacobians[0][0] = 2. * dist21 * (x1_xc * x1_xc - length1 * length1) / length1 / length1 / length1
                + 2. * dist22 * x1_xc * y1_yc / length1 / length1 / length1;
            jacobians[0][1] = 2. * dist22 * (y1_yc * y1_yc - length1 * length1) / length1 / length1 / length1
                + 2. * dist21 * x1_xc * y1_yc / length1 / length1 / length1;
#endif
            jacobians[1][0] = 2. * dist21 * (sin(parameters[1][0])) + 2. * dist22 * (-cos(parameters[1][0]));
            jacobians[1][1] = 0.;
            jacobians[1][2] = 0.;
            //----------------------------------------------------------------------------------------------
#if USE_SECOND_RASID>0
            jacobians[0][2] = 2. * dist23 * (x2_xc * x2_xc - length2 * length2) / length2 / length2 / length2
                + 2. * dist24 * x2_xc * y2_yc / length2 / length2 / length2 + SECOND_WEIGHT * 2.* x2_xc;
            jacobians[0][3] = 2. * dist24 * (y2_yc * y2_yc - length2 * length2) / length2 / length2 / length2
                + 2. * dist23 * x2_xc * y2_yc / length2 / length2 / length2+ SECOND_WEIGHT * 2.* y2_yc;
#else
            jacobians[0][2] = 2. * dist23 * (x2_xc * x2_xc - length2 * length2) / length2 / length2 / length2
                + 2. * dist24 * x2_xc * y2_yc / length2 / length2 / length2;
            jacobians[0][3] = 2. * dist24 * (y2_yc * y2_yc - length2 * length2) / length2 / length2 / length2
                + 2. * dist23 * x2_xc * y2_yc / length2 / length2 / length2;
#endif
            jacobians[1][3] = 0.;
            jacobians[1][4] = 2. * dist23 * (sin(parameters[1][1])) + 2. * dist24 * (-cos(parameters[1][1]));
            jacobians[1][5] = 0.;
            //----------------------------------------------------------------------------------------------
#if USE_SECOND_RASID>0
            jacobians[0][4] = 2. * dist25 * (x3_xc * x3_xc - length3 * length3) / length3 / length3 / length3
                + 2. * dist26 * x3_xc * y3_yc / length3 / length3 / length3+ SECOND_WEIGHT * 2.*x3_xc;
            jacobians[0][5] = 2. * dist26 * (y3_yc * y3_yc - length3 * length3) / length3 / length3 / length3
                + 2. * dist25 * x3_xc * y3_yc / length3 / length3 / length3+ SECOND_WEIGHT * 2. * y3_yc;
#else
            jacobians[0][4] = 2. * dist25 * (x3_xc * x3_xc - length3 * length3) / length3 / length3 / length3
                + 2. * dist26 * x3_xc * y3_yc / length3 / length3 / length3;
            jacobians[0][5] = 2. * dist26 * (y3_yc * y3_yc - length3 * length3) / length3 / length3 / length3
                + 2. * dist25 * x3_xc * y3_yc / length3 / length3 / length3;
#endif
            jacobians[1][6] = 0.;
            jacobians[1][7] = 0.;
            jacobians[1][8] = 2. * dist25 * (sin(parameters[1][2])) + 2. * dist26 * (-cos(parameters[1][2]));
        }
        return true;
    }
private:
    const double* xyz_;
}; 
struct Residual3 {
    Residual3(double* xyz) : xyz_(xyz) {}

    template <typename T>
    bool operator()(const T* const center, const T* const angles, T* residual) const
    {
        const double& x1 = xyz_[0];
        const double& y1 = xyz_[1];
        const double& x2 = xyz_[2];
        const double& y2 = xyz_[3];
        const double& x3 = xyz_[4];
        const double& y3 = xyz_[5]; 
        T centerX = (center[0]);
        T centerY = (center[1]);
        //    line:tan(theta1)*(x-xc)-y+yc=0
        //    line:tan(theta2)*(x-xc)-y+yc=0
        //    line:tan(theta3)*(x-xc)-y+yc=0
        T tan1 = tan(angles[0]);
        T tan2 = tan(angles[1]);
        T tan3 = tan(angles[2]);
        T x1_xc = x1 - centerX;
        T y1_yc = y1 - centerY;
        T x2_xc = x2 - centerX;
        T y2_yc = y2 - centerY;
        T x3_xc = x3 - centerX;
        T y3_yc = y3 - centerY;
        T d1 = tan1 * x1_xc - y1_yc;
        T d2 = tan2 * x2_xc - y2_yc;
        T d3 = tan3 * x3_xc - y3_yc;
        residual[0] = static_cast<T>(d1 * d1) / (static_cast<T>(1.) + tan1 * tan1);
        residual[1] = static_cast<T>(d2 * d2) / (static_cast<T>(1.) + tan2 * tan2);
        residual[2] = static_cast<T>(d3 * d3) / (static_cast<T>(1.) + tan3 * tan3);
        return true;
    }
private:
    const double* xyz_;
};
struct Residual4 :public ceres::SizedCostFunction<3, 2, 3 >
{
    Residual4(double* xyz) : xyz_(xyz) {}
    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const
    {
        const double& x1 = xyz_[0];
        const double& y1 = xyz_[1];
        const double& x2 = xyz_[2];
        const double& y2 = xyz_[3];
        const double& x3 = xyz_[4];
        const double& y3 = xyz_[5];
        double centerX = (parameters[0][0]);
        double centerY = (parameters[0][1]); 
        //    line:tan(theta1)*(x-xc)-y+yc=0
        //    line:tan(theta2)*(x-xc)-y+yc=0
        //    line:tan(theta3)*(x-xc)-y+yc=0
        double tan1 = tan(parameters[1][0]);
        double tan2 = tan(parameters[1][1]);
        double tan3 = tan(parameters[1][2]);
        double tan1_2 = tan1 * tan1;
        double tan2_2 = tan2 * tan2;
        double tan3_2 = tan3 * tan3;
        double tan1_2_1 = tan1_2+1.;
        double tan2_2_1 = tan2_2+1.;
        double tan3_2_1 = tan3_2+1.;
        double cos1 = cos(parameters[1][0]);
        double cos2 = cos(parameters[1][1]);
        double cos3 = cos(parameters[1][2]);
        double x1_xc = x1 - centerX;
        double y1_yc = y1 - centerY; 
        double x2_xc = x2 - centerX;
        double y2_yc = y2 - centerY; 
        double x3_xc = x3 - centerX;
        double y3_yc = y3 - centerY; 
        double d1 = tan1 * x1_xc - y1_yc;
        double d2 = tan2 * x2_xc - y2_yc;
        double d3 = tan3 * x3_xc - y3_yc;
        double d1_2 = d1 * d1;
        double d2_2 = d2 * d2;
        double d3_2 = d3 * d3;

        residuals[0] = d1_2 / tan1_2_1;
        residuals[1] = d2_2 / tan2_2_1;
        residuals[2] = d3_2 / tan3_2_1;


        if (jacobians != NULL && jacobians[0] != NULL)
        {
            //---------------------------------------------------------------------------------------------- 
 
            jacobians[0][0] = 2. * d1 / tan1_2_1 * (-tan1);
            jacobians[0][1] = 2. * d1 / tan1_2_1;
            jacobians[1][0] = (2. * d1 * x1_xc / cos1 / cos1 * tan1_2_1 - d1_2 * 2. * tan1 / cos1 / cos1) / tan1_2_1 / tan1_2_1;
            jacobians[1][1] = 0.;
            jacobians[1][2] = 0.;
            //----------------------------------------------------------------------------------------------
 
            jacobians[0][2] = 2. * d2 / (1 + tan2 * tan2) * (-tan2);
            jacobians[0][3] = 2. * d2 / (1 + tan2 * tan2);
            jacobians[1][3] = 0.;
            jacobians[1][4] = (2. * d2 * x2_xc / cos2 / cos2 * tan2_2_1 - d2_2 * 2. * tan2 / cos2 / cos2) / tan2_2_1 / tan2_2_1;
            jacobians[1][5] = 0.;
            //----------------------------------------------------------------------------------------------
 
            jacobians[0][4] = 2. * d3 / (1 + tan3 * tan3) * (-tan3);
            jacobians[0][5] = 2. * d3 / (1 + tan3 * tan3);
            jacobians[1][6] = 0.;
            jacobians[1][7] = 0.;
            jacobians[1][8] = (2. * d3 * x3_xc / cos3 / cos3* tan3_2_1 - d3_2 * 2. * tan3 / cos3 / cos3) / tan3_2_1 / tan3_2_1;
        }
        return true;
    }
private:
    const double* xyz_;
};

void checkDists(const std::vector<double>&data, const double&centerX, const double& centerY, const double& angle1, const double& angle2, const double& angle3)
{
    CHECK(data.size() > 0 && data.size() % 6 == 0);
    int tripleCnt = data.size() / 6;
    double theta1 = angle1 ;
    double theta2 = angle2 ;
    double theta3 = angle3 ;
    std::vector<double>dists(data.size()/2);
    double  x1_theta = cos(theta1) + centerX;
    double  y1_theta = sin(theta1) + centerY;
    double  x2_theta = cos(theta2) + centerX;
    double  y2_theta = sin(theta2) + centerY;
    double  x3_theta = cos(theta3) + centerX;
    double  y3_theta = sin(theta3) + centerY;
    for (int i = 0; i < tripleCnt; i++)
    {
        const double& x1 = data[6 * i + 0];
        const double& y1 = data[6 * i + 1];
        const double& x2 = data[6 * i + 2];
        const double& y2 = data[6 * i + 3];
        const double& x3 = data[6 * i + 4];
        const double& y3 = data[6 * i + 5];
        double length1 = sqrt((centerX - x1) * (centerX - x1) + (centerY - y1) * (centerY - y1));
        double length2 = sqrt((centerX - x2) * (centerX - x2) + (centerY - y2) * (centerY - y2));
        double length3 = sqrt((centerX - x3) * (centerX - x3) + (centerY - y3) * (centerY - y3));
        double new_x1 = (x1 - centerX) / length1 + centerX;
        double new_y1 = (y1 - centerY) / length1 + centerY;
        double new_x2 = (x2 - centerX) / length2 + centerX;
        double new_y2 = (y2 - centerY) / length2 + centerY;
        double new_x3 = (x3 - centerX) / length3 + centerX;
        double new_y3 = (y3 - centerY) / length3 + centerY;
        dists[3 * i + 0] = sqrt((new_x1 - x1_theta) * (new_x1 - x1_theta) + (new_y1 - y1_theta) * (new_y1 - y1_theta));
        dists[3 * i + 1] = sqrt((new_x2 - x2_theta) * (new_x2 - x2_theta) + (new_y2 - y2_theta) * (new_y2 - y2_theta));
        dists[3 * i + 2] = sqrt((new_x3 - x3_theta) * (new_x3 - x3_theta) + (new_y3 - y3_theta) * (new_y3 - y3_theta));
    }
    double avg = std::accumulate(dists.begin(), dists.end(), 1.) / dists.size();
    double minV = *std::min_element(dists.begin(), dists.end());
    double maxV = *std::max_element(dists.begin(), dists.end());
    LOG(INFO) << minV << "\t" << avg << "\t" << maxV;
}

int TEST_threeLaserAndOneCenter()
{
    int dataTripleCnt = 10;
    double centerX_gt = 5.123;
    double centerY_gt = 4.789;
    double angle1_gt = 121.1/180.*PI;
    double angle2_gt = 0.5 / 180. * PI;
    double angle3_gt = -120.6 / 180. * PI;
    dataGenterator ins(centerX_gt, centerY_gt, angle1_gt, angle2_gt, angle3_gt);

    std::vector<double>tripleData = ins.gener(dataTripleCnt);
    //writeData("data.txt", tripleData);
    double initCenter[2] = { RandUnit ,RandUnit };
    double initAngles[3] = { RandUnit ,RandUnit,-RandUnit };

    ceres::Problem problem;
    for (int i = 0; i < dataTripleCnt; ++i)
    {
#if UseAutoDiff>0
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual3, 3, 2,3>(
                new Residual3(&tripleData[6 * i])),
            NULL,
            initCenter,
            initAngles);
#else
        ceres::CostFunction* cost_function = new Residual4(&tripleData[6 * i]);
        problem.AddResidualBlock(cost_function, NULL, initCenter, initAngles);
#endif
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "GT     : " << centerX_gt << "  " << centerY_gt << "  " << angle1_gt * 180. / PI << "  " << angle2_gt * 180. / PI << "  " << angle3_gt * 180. / PI << "\n";
    while (initAngles[0] > PI)initAngles[0] -= PI;
    while (initAngles[1] > PI)initAngles[1] -= PI;
    while (initAngles[2] > PI)initAngles[2] -= PI;
    while (initAngles[0] <- PI)initAngles[0] += PI;
    while (initAngles[1] <- PI)initAngles[1] += PI;
    while (initAngles[2] <- PI)initAngles[2] += PI;
    std::cout << "Final  : " << initCenter[0] << "  " << initCenter[1] << "  " << initAngles[0] * 180. / PI << "  " << initAngles[1] * 180. / PI << "  " << initAngles[2] * 180. / PI << "\n";
    //checkDists(tripleData, initCenter[0], initCenter[1], initAngles[0], initAngles[1] , initAngles[2]);
    //checkDists(tripleData, centerX_gt, centerY_gt, angle1_gt, angle2_gt, angle3_gt);
    return 0;
}
