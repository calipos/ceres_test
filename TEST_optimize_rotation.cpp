//g++ TEST_optimize_rotation.cpp -I/home/tx2/repo/ceres-solver/build/install/usr/local/include -I/usr/local/include/eigen3 -I/usr/include -L/home/tx2/repo/ceres-solver/build/install/usr/local/lib -lceres -lglog -lpthread -lopenblas


#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "glog/logging.h"
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#define RandUnit (rand()%10000/5000.-1)

struct RotationError {
  RotationError(double before_x_,double before_y_,double before_z_,double after_x_,double after_y_,double after_z_)
  {
	before_x=before_x_;
	before_y=before_y_;
	before_z=before_z_;
	after_x=after_x_;
	after_y=after_y_;
	after_z=after_z_;
  }

  template <typename T>
  bool operator()(const T* const rVec,
                  T* residuals) const
 {
    T p[3]; 
    T beforeXYZ[3]={static_cast<T>(before_x),static_cast<T>(before_y),static_cast<T>(before_z)};
    ceres::AngleAxisRotatePoint(rVec, beforeXYZ, p);
    residuals[0] = after_x - p[0];
    residuals[1] = after_y - p[1];
    residuals[2] = after_z - p[2];
    return true;
  }

  

  double before_x, before_y, before_z;
  double after_x, after_y, after_z;
};
//*****************************************************************************************************************************************************************
int main(int argc, char** argv) {
FLAGS_logtostderr = true;
google::InitGoogleLogging(argv[0]);
 
int sampleNum=10000;
std::vector<double>before(3*sampleNum);
std::vector<double>after(3*sampleNum);
double axis_gt[3]={1,2,-2};
for(int i=0;i<sampleNum;i++)
{
	before[3*i]=RandUnit;
	before[3*i+1]=RandUnit;
	before[3*i+2]=RandUnit;
	double ax[3];
	double p0[3];
	double p[3];
	ceres::AngleAxisRotatePoint(axis_gt, &before[3*i], p);
	after[3*i]=p[0]+0.01*RandUnit;
	after[3*i+1]=p[1]+0.01*RandUnit;
	after[3*i+2]=p[2]+0.01*RandUnit;
}
//double rVec[3]={0.1,1,-0.5};
	double *rVec=new double[3];
	rVec[0]=0.1;
	rVec[1]=0.2;
	rVec[2]=3;
    
  ceres::Problem problem;
  for (int i = 0; i < sampleNum; ++i) 
  {
    ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<RotationError, 3, 3>(
								new RotationError(
								before[3 * i],before[3 * i+1],before[3 * i+2],
								after[3 * i],after[3 * i+1],after[3 * i+2])
								);
    problem.AddResidualBlock(cost_function,
                             NULL ,// squared loss 
                             rVec);
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::DENSE_SCHUR;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";
LOG(INFO)<<rVec[0]<<" "<<rVec[1]<<" "<<rVec[2];
  return 0;
}
