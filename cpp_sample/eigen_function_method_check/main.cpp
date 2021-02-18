#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;


MatrixXd calc(MatrixXd m, VectorXd v) {
    return (m.transpose().colwise() + v).transpose();
}


MatrixXd calc2(MatrixXd m, VectorXd v) {
    return m.rowwise() + v.transpose();
}


int main()
{
    MatrixXd m(3, 2);
    m << 1, 2, 3, 4, 5, 6;
    std::cout << m << std::endl;
    std::cout << std::endl;

    VectorXd v(2);
    v << 10, 20;
    std::cout << v << std::endl;
    std::cout << std::endl;

    std::cout << (m.transpose().colwise() + v).transpose() << std::endl;
    std::cout << std::endl;

    std::cout << calc(m, v) << std::endl;
    std::cout << std::endl;

    std::cout << calc2(m ,v) << std::endl;
    std::cout << std::endl;

    return 0;
}