#include <cmath>
#include <iostream>
#include <Eigen/Dense>

#include "functions.hpp"

using Eigen::MatrixXd;


class Matmul
{
public:
    Matmul(MatrixXd _W){
        W = _W;
    }

    MatrixXd forward(MatrixXd _x) {
        auto out = _x * W.transpose();
        x = _x;
        return out;
    }
    MatrixXd x;
    MatrixXd W;
};


class SoftmaxWithLoss
{
public:
    SoftmaxWithLoss(){}

    double forward(MatrixXd _x, MatrixXd _t) {
        t = _t;
        y = _x.unaryExpr([](double e){return 1.0/(1.0+std::exp(-e));});

        MatrixXd tmp(y.rows(), y.cols()+y.cols());
        tmp << y.unaryExpr([](double e){return 1.0-e;}), y;
        loss = cross_entropy_error(y, t);

        return loss;
    }

    double loss;
    MatrixXd y;
    MatrixXd t;
};


int main()
{
    SoftmaxWithLoss softmax = SoftmaxWithLoss();
    MatrixXd y(4, 1);
    MatrixXd t(4, 1);
    
    y << 1, 2, 3, 4;
    t << 1, 2, 3, 4;

    std::cout << y << std::endl;
    std::cout << t << std::endl;

    auto result = softmax.forward(y, t);
    std::cout << result << std::endl;


    MatrixXd W(2, 4);
    MatrixXd x(2, 4);
    W << 1, 2, 3, 4, 5, 6, 7, 8;
    x << 1, 2, 3, 4, 5, 6, 7, 8;

    Matmul mat = Matmul(W);
    auto result2 = mat.forward(x);
    std::cout << result2 << std::endl;

    return 0;
}
