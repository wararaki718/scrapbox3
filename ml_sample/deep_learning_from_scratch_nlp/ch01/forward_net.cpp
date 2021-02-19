#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class Sigmoid
{
public:
    Sigmoid(){}
    MatrixXd forward(MatrixXd x) {
        return x.unaryExpr([](double e){return 1.0/(1.0+std::exp(e));});
    }
    std::vector<MatrixXd> params;
};


class Affine
{
public:
    Affine(){}
    Affine(MatrixXd _W, VectorXd _b){
        W = _W;
        b = _b;
    }
    MatrixXd forward(MatrixXd x) {
        return (x* W).rowwise() + b.transpose();
    }
    MatrixXd W;
    VectorXd b;
};


class TwoLayerNet
{
public:
    TwoLayerNet(int input_size, int hidden_size, int output_size) {
        auto W1 = MatrixXd::Random(input_size, hidden_size);
        auto b1 = VectorXd::Random(hidden_size);
        auto W2 = MatrixXd::Random(hidden_size, output_size);
        auto b2 = VectorXd::Random(output_size);

        layer1 = Affine(W1, b1);
        sigmoid = Sigmoid();
        layer2 = Affine(W2, b2);
    }

    MatrixXd predict(MatrixXd x) {
        auto out = layer1.forward(x);
        out = sigmoid.forward(out);
        out = layer2.forward(out);
        return out;
    }

    Affine layer1;
    Affine layer2;
    Sigmoid sigmoid;
};


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
    auto b = VectorXd::Random(2);
    auto affine = Affine(W, b);
    std::cout << affine.forward(m) << std::endl;
    std::cout << std::endl;

    int input_size = 2;
    int hidden_size = 2;
    int output_size = 1;
    auto model = TwoLayerNet(input_size, hidden_size, output_size);
    std::cout << model.predict(m) << std::endl;

    return 0;
}