#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "glog/logging.h"
     
#define RandUnit (rand()%10000/5000.-1)
#define UseAutoDiff 0
struct ExponentialResidual {
    ExponentialResidual(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const m, const T* const c, T* residual) const
    {
        residual[0] = y_ - exp(m[0] * x_ + c[0]);
        return true;
    } 
private:
    const double x_;
    const double y_;
};
struct ExponentialResidual2    :public ceres::SizedCostFunction<1, 2 >
{
    ExponentialResidual2(double x, double y) : x_(x), y_(y) {} 
    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const 
    { 
        residuals[0] = y_ - exp(parameters[0][0] * x_ + parameters[0][1]);
        if (jacobians != NULL && jacobians[0] != NULL) {
            jacobians[0][0] = -x_ * exp(parameters[0][0] * x_ + parameters[0][1]);
            jacobians[0][1] =   -exp(parameters[0][0] * x_ + parameters[0][1]);
        } 
        return true;
    }
private:
    const double x_;
    const double y_;
}; 


int y-exp(mx+c)()
{  
    double* mc = new double[2];
    mc[0] = 2;
    mc[1] = 3;
    int kNumObservations = 1000;
    std::vector<double>data(kNumObservations * 2);
    { 
        for (int i = 0; i < kNumObservations; i++)
        {
            data[2 * i] = RandUnit * 5;
            data[2 * i+1] = exp(0.3* data[2 * i] + 0.1)+0.01* RandUnit;
        }
    }
    ceres::Problem problem;
    for (int i = 0; i < kNumObservations; ++i) 
    {
#if UseAutoDiff>0
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                new ExponentialResidual(data[2 * i], data[2 * i + 1])),
            NULL,
            &mc[0],
            &mc[1]);
#else
        ceres::CostFunction* cost_function = new ExponentialResidual2(data[2 * i], data[2 * i + 1]);
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
    std::cout << "Initial m: " << 0.0 << " c: " << 0.0 << "\n";
    std::cout << "Final   m: " << mc[0] << " c: " << mc[1] << "\n";
    return 0;
}
