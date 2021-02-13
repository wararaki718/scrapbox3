#include <cmath>
#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(x));
}

int main()
{
    MatrixXd m(2, 2);
    m << 3, 2.5, -1, 0;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    std::cout << m.unaryExpr(&sigmoid) << std::endl;

    return 0;
}