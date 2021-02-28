#include <iostream>
#include <string>

template <typename T>
T plus(const T& a, const T& b)
{
    return a + b;
}


int main()
{
    int a = 1;
    int b = 2;
    double c = 1.5;
    double d = 2.;
    std::string e = "1";
    std::string f = "2";

    std::cout << plus(a, b) << std::endl;
    std::cout << plus(c, d) << std::endl;
    std::cout << plus(e, f) << std::endl;

    return 0;
}
