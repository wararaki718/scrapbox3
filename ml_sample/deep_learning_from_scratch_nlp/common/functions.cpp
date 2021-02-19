#include <cmath>
#include <iostream>
#include <Eigen/Dense>


using Eigen::MatrixXd;


double cross_entropy_error(MatrixXd y, MatrixXd t)
{
    auto _y = y;
    auto _t = t;
    if(y.cols() == 1) {
        _t.resize(1, t.rows()*t.cols());
        _y.resize(1, y.rows()*y.cols());
    }

    /*
    if(t.rows()*t.cols() == y.rows()*y.cols()) {
        MatrixXd tmp(1, t.rows());
        for(int i = 0; i < t.rows(); i++) {
            for(int j = 0; j < t.cols(); j++) {
                if(t(i, j) > 0){
                    tmp << j;
                }
            }
        }
        t = tmp;
    }
    */

    //int batch_size = y.rows();
    std::cout << _y << std::endl;
    std::cout << _t << std::endl;
    return -1 * (_t.transpose() * _y.unaryExpr([](double e){return std::log(e);})).sum();
}


int main()
{
    MatrixXd y(4, 1);
    MatrixXd t(4, 1);
    
    y << 1, 2, 3, 4;
    t << 1, 2, 3, 4;

    std::cout << y << std::endl;
    std::cout << t << std::endl;

    std::cout << cross_entropy_error(y, t) << std::endl;
    std::cout << std::endl;

    return 0;
}