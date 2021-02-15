#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;

class Layer{};

class Sigmoid : public Layer
{
public:
    Sigmoid(){}
    MatrixXd forward(MatrixXd x) {
        return x.unaryExpr([](double e){return 1.0/(1.0+std::exp(e));});
    }
    std::vector<MatrixXd> params;
};


class Affine: public Layer
{
public:
    Affine(MatrixXd W, MatrixXd b){
        params.push_back(W);
        params.push_back(b);
    }
    MatrixXd forward(MatrixXd x) {
        auto W = params[0];
        auto b = params[1];
        auto out = x.adjoint() * W + b;
        return out;
    }
    std::vector<MatrixXd> params;
};


/*
class TwoLayerNet
{
public:
    TwoLayerNet(int input_size, int hidden_size, int output_size) {
        auto W1 = MatrixXd::Random(input_size, hidden_size);
        auto b1 = MatrixXd::Random(hidden_size, hidden_size);
        auto W2 = MatirxXd::Random(hidden_size, output_size);
        auto b2 = MatrixXd::Random(hidden_size, output_size);
    }
};
*/

int main()
{
    MatrixXd m(2, 2);
    m << 3, 2.5, -1, 0;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
    std::cout << std::endl;

    auto sigmoid = Sigmoid();
    std::cout << sigmoid.forward(m) << std::endl;
    std::cout << std::endl;

    auto W = MatrixXd::Random(2, 2);
    auto b = MatrixXd::Random(2, 2);
    auto affine = Affine(W, b);
    std::cout << affine.forward(m) << std::endl;

    return 0;
}