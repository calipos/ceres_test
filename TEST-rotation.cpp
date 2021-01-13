//g++ TEST_optimize_rotation.cpp -I/home/tx2/repo/ceres-solver/build/install/usr/local/include -I/usr/local/include/eigen3 -I/usr/include -L/home/tx2/repo/ceres-solver/build/install/usr/local/lib -lceres -lglog -lpthread -lopenblas


#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "glog/logging.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"

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
template<class Dtype>
cv::Mat rotationVectorToMatrix(const double& a, const double& b, const double& c)
{
	cv::Mat r(3, 1, sizeof(Dtype) / 4 + 4);
	r.ptr<Dtype>(0)[0] = a;
	r.ptr<Dtype>(1)[0] = b;
	r.ptr<Dtype>(2)[0] = c;
	Dtype theta = cv::norm(r);
	cv::Mat unitAxis = r / theta;
	cv::Mat K=cv::Mat::zeros(3, 3, sizeof(Dtype) / 4 + 4);
	K.ptr<Dtype>(0)[0] = 0;
	K.ptr<Dtype>(0)[1] = -unitAxis.ptr<Dtype>(2)[0];
	K.ptr<Dtype>(0)[2] = unitAxis.ptr<Dtype>(1)[0];
	K.ptr<Dtype>(1)[0] = unitAxis.ptr<Dtype>(2)[0];;
	K.ptr<Dtype>(1)[1] = 0;
	K.ptr<Dtype>(1)[2] = -unitAxis.ptr<Dtype>(0)[0];
	K.ptr<Dtype>(2)[0] = -unitAxis.ptr<Dtype>(1)[0];
	K.ptr<Dtype>(2)[1] = unitAxis.ptr<Dtype>(0)[0];
	K.ptr<Dtype>(2)[2] =0 ;
	cv::Mat R = cv::Mat::eye(3, 3, sizeof(Dtype) / 4 + 4) + (1 - cos(theta)) * K * K + sin(theta) * K;
	return R;
}
template<class Dtype>
cv::Mat zyxToToMatrix(const double& z, const double& y, const double& x)
{
	cv::Mat Rx = cv::Mat::eye(3, 3, sizeof(Dtype) / 4 + 4);
	cv::Mat Ry = cv::Mat::eye(3, 3, sizeof(Dtype) / 4 + 4);
	cv::Mat Rz = cv::Mat::eye(3, 3, sizeof(Dtype) / 4 + 4);
	Rx.ptr<Dtype>(1)[1] = cos(x);
	Rx.ptr<Dtype>(1)[2] = -sin(x);
	Rx.ptr<Dtype>(2)[1] = sin(x);
	Rx.ptr<Dtype>(2)[2] = cos(x);
	Ry.ptr<Dtype>(0)[0] = cos(y);
	Ry.ptr<Dtype>(0)[2] = sin(y);
	Ry.ptr<Dtype>(2)[0] = -sin(y);
	Ry.ptr<Dtype>(2)[2] = cos(y);
	Rz.ptr<Dtype>(0)[0] = cos(z);
	Rz.ptr<Dtype>(0)[1] = -sin(z);
	Rz.ptr<Dtype>(1)[0] = sin(z);
	Rz.ptr<Dtype>(1)[1] = cos(z);
	return Rz * Ry * Rx;
}
template<class Dtype>
cv::Point3_<Dtype> matrixToZyx(const cv::Mat& m)
{
	if (m.cols != 3 || m.rows != 3)return cv::Point3_<Dtype>(0, 0, 0);
	cv::Point3_<Dtype> zyx;
	zyx.x = atan2(m.ptr<Dtype>(2)[1], m.ptr<Dtype>(2)[2]);
	zyx.y = atan2(-m.ptr<Dtype>(2)[0], sqrt(m.ptr<Dtype>(2)[2]* m.ptr<Dtype>(2)[2]+ m.ptr<Dtype>(2)[1]* m.ptr<Dtype>(2)[1]));
	zyx.z = atan2(m.ptr<Dtype>(1)[0], m.ptr<Dtype>(0)[0]);
	return zyx;
}
void TEST_matrictDev()
{
	cv::Mat R, jacob, y;
	cv::Mat rAxis = cv::Mat(3, 1, CV_64FC1);
	rAxis.ptr<double>(0)[0] = 1;
	rAxis.ptr<double>(1)[0] = 2;
	rAxis.ptr<double>(2)[0] = 3;
	cv::Rodrigues(rAxis, R, jacob);
	cv::Mat R2 = rotationVectorToMatrix<double>(1, 2, 3);
	return;

}
cv::Point3f TESTrotation( )
{
	cv::Mat newR = zyxToToMatrix<float>(CV_PI / 6, CV_PI / 4, CV_PI / 3);
	cv::Point3d zyx = matrixToZyx<float>(newR);
	{
		cv::Mat cameraRgt, cameraR;
		cv::Mat r = cv::Mat::zeros(3, 1, CV_32FC1);
		r.ptr<float>(0)[0] = -0.02;
		r.ptr<float>(1)[0] = 0.01;
		r.ptr<float>(2)[0] = 1;
		r /= cv::norm(r);
		r *= 3.1415926535 / 12;
		cv::Rodrigues(r, cameraRgt);
		r.ptr<float>(0)[0] = 0;
		r.ptr<float>(1)[0] = 0;
		r.ptr<float>(2)[0] = 1;
		r /= cv::norm(r);
		r *= 3.1415926535 / 12;
		cv::Rodrigues(r, cameraR);
		cv::Mat R_delta = cameraRgt * cameraR.t();
		cv::Point3d zyx = matrixToZyx<float>(R_delta);
	}
	TEST_matrictDev();
	int sampleNum = 1000;
	std::vector<double>before(3 * sampleNum);
	std::vector<double>after(3 * sampleNum);
	cv::Point3f axisDir = {1,2,6};
	double angle = 0.57;
	axisDir = axisDir / cv::norm(axisDir) * angle;
	double axis_gt[3] = { axisDir.x,axisDir.y,axisDir.z };
	for (int i = 0; i < sampleNum; i++)
	{
		double length = 2 + RandUnit;
		double theta = atan2(RandUnit, RandUnit);
		before[3 * i] = length*cos(theta);
		before[3 * i + 1] = length * sin(theta);;
		before[3 * i + 2] = RandUnit;
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
	options.max_num_iterations=100;
	options.update_state_every_iteration = true;
	options.linear_solver_type = ceres::DENSE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	ceres::Solver::Summary summary;
	ceres::Solve(options, &problem, &summary);
	std::cout << summary.FullReport() << "\n"; 
	cv::Point3f ret;
	ret.x = rVec[0];
	ret.y = rVec[1];
	ret.z = rVec[2];

	LOG(INFO) << axisDir;
	return ret;
}

