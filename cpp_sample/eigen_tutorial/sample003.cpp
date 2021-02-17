#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

int main()
{
    MatrixXd m(2, 2);
    m << 1, 2, 3, 4;
    VectorXd v(2);
    v << 5, 10;
    std::cout << "Matrix:" << std::endl;
    std::cout << m << std::endl;
    std::cout << "Vector:" << std::endl;
    std::cout << v << std::endl;
    std::cout << "m + v =" << std::endl;
    std::cout << m.colwise() + v << std::endl;

    return 0;
}