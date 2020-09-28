//g++ TEST_optimize_rotation.cpp -I/home/tx2/repo/ceres-solver/build/install/usr/local/include -I/usr/local/include/eigen3 -I/usr/include -L/home/tx2/repo/ceres-solver/build/install/usr/local/lib -lceres -lglog -lpthread -lopenblas

#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "glog/logging.h"

 
#include <algorithm>
#include <cstddef>
#include <vector>  
#include "ceres/internal/fixed_array.h" 
#define RandUnit (rand()%10000/5000.-1)
#define UseAutoDiff 0
#define F1(X1,X2,X3,A,B,C,D,E,F)  (A*A*X1+B*X2+2*sqrt(C*C)*X3)
#define F2(X1,X2,X3,A,B,C,D,E,F)  (D*D*D*X1+E*E*X2-F*X3)
#define dF1dA(X1,X2,X3,A,B,C,D,E,F) (2*A*X1)
#define dF1dB(X1,X2,X3,A,B,C,D,E,F) (X2)
#define dF1dC(X1,X2,X3,A,B,C,D,E,F) (2*C/sqrt(C*C)*X3)
#define dF1dD(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF1dE(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF1dF(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF2dA(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF2dB(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF2dC(X1,X2,X3,A,B,C,D,E,F) (0)
#define dF2dD(X1,X2,X3,A,B,C,D,E,F) (3*D*D*X1)
#define dF2dE(X1,X2,X3,A,B,C,D,E,F) (2*E*X2)
#define dF2dF(X1,X2,X3,A,B,C,D,E,F) (-X3)
struct Residual {
    Residual(double x0, double x1, double x2, double y0, double y1) : x0_(x0), x1_(x1), x2_(x2), y0_(y0), y1_(y1) {}

    template <typename T>
    bool operator()(const T* const matrix,  T* residual) const
    {
        residual[0] = y0_ - matrix[0] * matrix[0] * x0_ - matrix[1] * x1_ - sin(matrix[2]) * x2_;
        residual[1] = y1_ - matrix[3]* matrix[3]* matrix[3] * x0_ - matrix[4] * matrix[4] * x1_ - matrix[5] * x2_;
        return true;
    }
private:
    const double x0_;
    const double x1_;
    const double x2_;
    const double y0_;
    const double y1_;
};
struct Residual2 :public ceres::SizedCostFunction<2,6>
{
    Residual2(double x0, double x1, double x2, double y0, double y1) : x0_(x0), x1_(x1), x2_(x2), y0_(y0), y1_(y1) {}
    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const
    {
        residuals[0] = F1(x0_,x1_,x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5])-y0_;
        residuals[1] = F2(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]) - y1_;
 
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = dF1dA(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][1] = dF1dB(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][2] = dF1dC(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][3] = dF1dD(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][4] = dF1dE(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][5] = dF1dF(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 0] = dF2dA(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 1] = dF2dB(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 2] = dF2dC(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 3] = dF2dD(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 4] = dF2dE(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
            jacobians[0][6 + 5] = dF2dF(x0_, x1_, x2_, parameters[0][0], parameters[0][1], parameters[0][2], parameters[0][3], parameters[0][4], parameters[0][5]);
        }
        return true;
    }
private:
    const double x0_;
    const double x1_;
    const double x2_;
    const double y0_;
    const double y1_;
};


int r4()
{
    //ceres::internal::FixedArray<double*, 8> global_jacobians(1);
    double gt_mc[6] = { 2,3,4,-1,2,-3 };
    double* mc = new double[6];
    mc[0] = 1;    mc[1] = 1;    mc[2] = 1;
    mc[3] = -2;    mc[4] = 2;    mc[5] = -2;
    int kNumObservations = 1000;
    std::vector<double>data(kNumObservations * 5);
    {
        for (int i = 0; i < kNumObservations; i++)
        {
            data[5 * i] = RandUnit * 5;
            data[5 * i + 1] = RandUnit * 5;
            data[5 * i + 2] = RandUnit * 5;
            data[5 * i + 3] = F1(data[5 * i], data[5 * i+1], data[5 * i+2], gt_mc[0], gt_mc[1], gt_mc[2], gt_mc[3], gt_mc[4], gt_mc[5]);
            data[5 * i + 4] = F2(data[5 * i], data[5 * i + 1], data[5 * i + 2], gt_mc[0], gt_mc[1], gt_mc[2], gt_mc[3], gt_mc[4], gt_mc[5]);
        }
    }
    ceres::Problem problem;
    for (int i = 0; i < kNumObservations; ++i)
    {
#if UseAutoDiff>0
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<Residual, 2, 6>(
                new Residual(data[5 * i], data[5 * i + 1], data[5 * i + 2], data[5 * i + 3], data[5 * i + 4])),
            NULL,
            mc);
#else
        ceres::CostFunction* cost_function = new Residual2(data[5 * i], data[5 * i + 1], data[5 * i + 2], data[5 * i + 3], data[5 * i + 4]);
        problem.AddResidualBlock(cost_function, NULL, mc);
#endif



    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n"; 
    std::cout << mc[0] << "   " << mc[1] << "   " << mc[2] << "\n";
    std::cout << mc[3] << "   " << mc[4] << "   " << mc[5] << "\n";
    return 0;
}
