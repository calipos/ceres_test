#include <vector>
#include <cmath>
#include <cstdio>
#include <iostream>
#include "ceres/ceres.h"
#include "glog/logging.h"
     
#define RandUnit (rand()%10000/5000.-1)
#define UseAutoDiff 0
struct ExponentialResidual {
    ExponentialResidual(double *x, double *y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const A, const T* const B, T* residual) const
    {
        T y1 = A[0] *abs( x_[0])+ A[1] * abs(x_[2]) + A[2] * abs(x_[4]) + B[0];
        T y2 = A[0] * abs(x_[1])+ A[1] * abs(x_[3]) + A[2] * abs(x_[5]) + B[1];
        T y3 = A[3] * abs(x_[0])+ A[4] * abs(x_[2]) + A[5] * abs(x_[4]) + B[2];
        T y4 = A[3] * abs(x_[1])+ A[4] * abs(x_[3]) + A[5] * abs(x_[5]) + B[3];
        residual[0] = y1-y_[0];
        residual[1] = y2-y_[1];
        residual[2] = y3-y_[2];
        residual[3] = y4-y_[3]; 
        return true;
    } 
private:
    const double *x_;
    const double *y_;
};
struct ExponentialResidual2    :public ceres::SizedCostFunction<4, 6,4 >
{
    ExponentialResidual2(double*x, double*y) : x_(x), y_(y) {} 
    virtual bool Evaluate(double const* const* parameters,
        double* residuals,
        double** jacobians) const 
    { 
        double y1 = parameters[0][0] * abs(x_[0]) + parameters[0][1] * abs(x_[2]) + parameters[0][2] * abs(x_[4]) + parameters[1][0];
        double y2 = parameters[0][0] * abs(x_[1]) + parameters[0][1] * abs(x_[3]) + parameters[0][2] * abs(x_[5]) + parameters[1][1];
        double y3 = parameters[0][3] * abs(x_[0]) + parameters[0][4] * abs(x_[2]) + parameters[0][5] * abs(x_[4]) + parameters[1][2];
        double y4 = parameters[0][3] * abs(x_[1]) + parameters[0][4] * abs(x_[3]) + parameters[0][5] * abs(x_[5]) + parameters[1][3];
        residuals[0] = y1 - y_[0];
        residuals[1] = y2 - y_[1];
        residuals[2] = y3 - y_[2];
        residuals[3] = y4 - y_[3];
        if (jacobians != NULL && jacobians[0] != NULL) {
            std::vector<int> eachParamSize{6,4};
            for (size_t paramId = 0; paramId < 2; paramId++)
            {
                for (size_t residualId = 0; residualId < 4; residualId++)
                {
                    for (size_t subParamId = 0; subParamId < eachParamSize[paramId]; subParamId++)
                    {
                        jacobians[paramId][residualId * eachParamSize[paramId] + subParamId];
                    }
                }

            }
            jacobians[0][0] = abs(x_[0]);
            jacobians[0][1] = abs(x_[2]);
            jacobians[0][2] = abs(x_[4]);
            jacobians[0][3] = 0;
            jacobians[0][4] = 0;
            jacobians[0][5] = 0;
            jacobians[0][6 + 0] = abs(x_[1]);
            jacobians[0][6 + 1] = abs(x_[3]);
            jacobians[0][6 + 2] = abs(x_[5]);
            jacobians[0][6 + 3] = 0;
            jacobians[0][6 + 4] = 0;
            jacobians[0][6 + 5] = 0; 
            jacobians[0][12 + 0] = 0;;
            jacobians[0][12 + 1] = 0;
            jacobians[0][12 + 2] = 0;
            jacobians[0][12 + 3] = abs(x_[0]);
            jacobians[0][12 + 4] = abs(x_[2]);
            jacobians[0][12 + 5] = abs(x_[4]);
            jacobians[0][18 + 0] = 0;
            jacobians[0][18 + 1] = 0;
            jacobians[0][18 + 2] = 0;
            jacobians[0][18 + 3] = abs(x_[1]);
            jacobians[0][18 + 4] = abs(x_[3]);
            jacobians[0][18 + 5] = abs(x_[5]);

            jacobians[1][0] = 1;
            jacobians[1][1] = 0;
            jacobians[1][2] = 0;
            jacobians[1][3] = 0;
            jacobians[1][4+0] = 0;
            jacobians[1][4+1] = 1;
            jacobians[1][4+2] = 0;
            jacobians[1][4+3] = 0;
            jacobians[1][8+0] = 0;
            jacobians[1][8+1] = 0;
            jacobians[1][8+2] = 1;
            jacobians[1][8+3] = 0;
            jacobians[1][12+0] = 0;
            jacobians[1][12+1] = 0;
            jacobians[1][12+2] = 0;
            jacobians[1][12+3] = 1;

            //jacobians[0][0] = -x_ * exp(parameters[0][0] * x_ + parameters[0][1]);
            //jacobians[0][1] =   -exp(parameters[0][0] * x_ + parameters[0][1]);
        } 
        return true;
    }
private:
    const double* x_;
    const double* y_;
}; 


int TEST_Y_absX_B()
{
    double* A = new double[6];
    double* B = new double[4];
    A[0] = 1;    A[1] = 2;    A[2] = 3;    A[3] = 4;    A[4] = 5;    A[5] = 6;
    B[0] = 7;    B[1] = 8;    B[2] = 9;    B[3] = 10;
    double* initA = new double[6];
    double* initB = new double[4];
    {
        for (int i = 0; i < 6; i++)initA[i] = RandUnit * 5;
        for (int i = 0; i < 4; i++)initB[i] = RandUnit * 5;
    }
    int kNumObservations = 1000;
    std::vector<double>data(kNumObservations * 10);
    { 
        for (int i = 0; i < kNumObservations; i++)
        {
            data[10 * i] = RandUnit * 5; 
            data[10 * i + 1] = RandUnit * 5;
            data[10 * i + 2] = RandUnit * 5;
            data[10 * i + 3] = RandUnit * 5; 
            data[10 * i + 4] = RandUnit * 5; 
            data[10 * i + 5] = RandUnit * 5; 
            data[10 * i + 6] = A[0] * abs(data[10 * i]) + A[1] * abs(data[10 * i + 2]) + A[2] * abs(data[10 * i + 4]) + B[0];
            data[10 * i + 7] = A[0] * abs(data[10 * i+ 1] ) + A[1] * abs(data[10 * i + 3]) + A[2] * abs(data[10 * i + 5]) + B[1];
            data[10 * i + 8] = A[3] * abs(data[10 * i]) + A[4] * abs(data[10 * i + 2]) + A[5] * abs(data[10 * i + 4]) + B[2];
            data[10 * i + 9] = A[3] * abs(data[10 * i + 1]) + A[4] * abs(data[10 * i + 3]) + A[5] * abs(data[10 * i + 5]) + B[3];
        }
    }
    ceres::Problem problem;
    for (int i = 0; i < kNumObservations; ++i) 
    {
#if UseAutoDiff>0
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialResidual, 4, 6,4>(
                new ExponentialResidual(&data[10 * i], &data[10 * i + 6])),
            NULL,
            &(initA[0]),
            &(initB[0]));
#else
        ceres::CostFunction* cost_function = new ExponentialResidual2(&data[10 * i], &data[10 * i + 6]);
        problem.AddResidualBlock(cost_function, NULL, &initA[0], &initB[0]);
#endif
        


    }

    ceres::Solver::Options options;
    options.max_num_iterations = 25;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.BriefReport() << "\n";
    std::cout << "GT  : " << A[0] << "  " << A[1] << "  " << A[2] << "  " << A[3] << "  " << A[4] << "  " << A[5] << "  " << B[0] << "  " << B[1] << "  " << B[2] << "  " << B[3] << "\n";
    std::cout << "Final  : " << initA[0] << "  " << initA[1] << "  " << initA[2] << "  " << initA[3] << "  " << initA[4] << "  " << initA[5] << "  " << initB[0] << "  " << initB[1] << "  " << initB[2] << "  " << initB[3] << "\n";
    return 0;
}
