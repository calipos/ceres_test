//g++ TEST_optimize_rotation.cpp -I/home/tx2/repo/ceres-solver/build/install/usr/local/include -I/usr/local/include/eigen3 -I/usr/include -L/home/tx2/repo/ceres-solver/build/install/usr/local/lib -lceres -lglog -lpthread -lopenblas


#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#if (defined WIN32 || defined _WIN32) 
#ifdef NDEBUG
#pragma comment(lib,"shlwapi.lib") 
#pragma comment(lib,"libjpeg-turbo.lib")
#pragma comment(lib,"libjasper.lib")
#pragma comment(lib,"IlmImf.lib")
#pragma comment(lib,"libtiff.lib")
#pragma comment(lib,"libwebp.lib")
#pragma comment(lib,"libpng.lib")
#pragma comment(lib,"zlib.lib")
#pragma comment(lib,"opencv_world412.lib") 
#endif
#ifdef _DEBUG
#pragma comment(lib,"shlwapi.lib") 
#pragma comment(lib,"libjpeg-turbod.lib")
#pragma comment(lib,"libjasperd.lib")
#pragma comment(lib,"IlmImfd.lib")
#pragma comment(lib,"libtiffd.lib")
#pragma comment(lib,"libwebpd.lib")
#pragma comment(lib,"libpngd.lib")
#pragma comment(lib,"zlibd.lib")
#pragma comment(lib,"opencv_world412d.lib") 
#endif
#endif
#define RandUnit (rand()%10000/5000.-1)
#define UseAutoDiff 0
struct RotationError {
	RotationError(double before_x_, double before_y_, double before_z_, double after_x_, double after_y_, double after_z_)
	{
		before_x = before_x_;
		before_y = before_y_;
		before_z = before_z_;
		after_x = after_x_;
		after_y = after_y_;
		after_z = after_z_;
	}

	template <typename T>
	bool operator()(const T* const rVec,
		T* residuals) const
	{
		T p[3];
		T beforeXYZ[3] = { static_cast<T>(before_x),static_cast<T>(before_y),static_cast<T>(before_z) };
		ceres::AngleAxisRotatePoint(rVec, beforeXYZ, p);
		residuals[0] = after_x - p[0];
		residuals[1] = after_y - p[1];
		residuals[2] = after_z - p[2];
		return true;
	}
	double before_x, before_y, before_z;
	double after_x, after_y, after_z;
};
struct RotationError2 :public ceres::SizedCostFunction<3, 3>
{
	RotationError2(double x1, double x2, double x3, double y1, double y2, double y3) : x1_(x1), x2_(x2), x3_(x3), y1_(y1), y2_(y2), y3_(y3)
	{
		dR = cv::Mat::zeros(3, 9, CV_64FC1);
		dR.ptr<double>(0)[0] = x1_;
		dR.ptr<double>(0)[1] = x2_;
		dR.ptr<double>(0)[2] = x3_;
		dR.ptr<double>(1)[3] = x1_;
		dR.ptr<double>(1)[4] = x2_;
		dR.ptr<double>(1)[5] = x3_;
		dR.ptr<double>(2)[6] = x1_;
		dR.ptr<double>(2)[7] = x2_;
		dR.ptr<double>(2)[8] = x3_;
		x = cv::Mat(3, 1, CV_64FC1);
		x.ptr<double>(0)[0] = x1_;
		x.ptr<double>(1)[0] = x2_;
		x.ptr<double>(2)[0] = x3_;
	}
	virtual bool Evaluate(double const* const* parameters,
		double* residuals,
		double** jacobians) const
	{
		cv::Mat R,jacob,y;		
		cv::Mat rAxis = cv::Mat(3, 1, CV_64FC1);
		rAxis.ptr<double>(0)[0] = parameters[0][0];
		rAxis.ptr<double>(1)[0] = parameters[0][1];
		rAxis.ptr<double>(2)[0] = parameters[0][2];
		cv::Rodrigues(rAxis, R, jacob);
		y = R * x;
		residuals[0] = y.ptr<double>(0)[0] - y1_;
		residuals[1] = y.ptr<double>(1)[0] - y2_;
		residuals[2] = y.ptr<double>(2)[0] - y3_;
		if (jacobians != NULL && jacobians[0] != NULL) {
			cv::Mat residualMat = dR* jacob.t();
			jacobians[0][0] = residualMat.ptr<double>(0)[0];
			jacobians[0][1] = residualMat.ptr<double>(0)[1];
			jacobians[0][2] = residualMat.ptr<double>(0)[2];
			jacobians[0][3] = residualMat.ptr<double>(1)[0];
			jacobians[0][4] = residualMat.ptr<double>(1)[1];
			jacobians[0][5] = residualMat.ptr<double>(1)[2];
			jacobians[0][6] = residualMat.ptr<double>(2)[0];
			jacobians[0][7] = residualMat.ptr<double>(2)[1];
			jacobians[0][8] = residualMat.ptr<double>(2)[2];
		}
		return true;
	}
private:	
	const double x1_; const double x2_; const double x3_;	
	const double y1_; const double y2_; const double y3_;
	cv::Mat dR;
	cv::Mat x;
};

cv::Point3f r1( )
{ 
	int sampleNum = 1000;
	std::vector<double>before(3 * sampleNum);
	std::vector<double>after(3 * sampleNum);
	double axis_gt[3] = { 1,2,-2 };
	for (int i = 0; i < sampleNum; i++)
	{
		before[3 * i] = RandUnit*20;
		before[3 * i + 1] = RandUnit*20;
		before[3 * i + 2] = RandUnit*20;
		double ax[3];
		double p0[3];
		double p[3];
		ceres::AngleAxisRotatePoint(axis_gt, &before[3 * i], p);
		after[3 * i] = p[0] + 0.01 * RandUnit;
		after[3 * i + 1] = p[1] + 0.01 * RandUnit;
		after[3 * i + 2] = p[2] + 0.01 * RandUnit;
	}
	//double rVec[3]={0.1,1,-0.5};
	double* rVec = new double[3];
	rVec[0] = 0.1;
	rVec[1] = 0.2;
	rVec[2] = 3;

	ceres::Problem problem;
	for (int i = 0; i < sampleNum; ++i)
	{
#if UseAutoDiff>0
		ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RotationError, 3, 3>(
			new RotationError(
				before[3 * i], before[3 * i + 1], before[3 * i + 2],
				after[3 * i], after[3 * i + 1], after[3 * i + 2])
			);
		problem.AddResidualBlock(cost_function,
			NULL,// squared loss 
			rVec);
#else
		ceres::CostFunction* cost_function = new RotationError2(before[3 * i], before[3 * i + 1], before[3 * i + 2], after[3 * i], after[3 * i + 1], after[3 * i + 2]);
		problem.AddResidualBlock(cost_function, NULL, rVec);
#endif
 
		
	}

	ceres::Solver::Options options;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n"; 
	cv::Point3f ret;
	ret.x = rVec[0];
	ret.y = rVec[1];
	ret.z = rVec[2];
	return ret;
}
