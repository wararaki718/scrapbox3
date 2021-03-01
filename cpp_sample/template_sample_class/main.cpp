#include <iostream>
#include <string>
#include <vector>
// #include "custom_stack.hpp"


template<int a>
class Sample
{
    public:
    Sample() {
        vec.assign(a, a);
    }
    std::vector<int> vec;

    void show() {
        for(const auto& e: vec) {
            std::cout << e << std::endl;
        }
    }
};


class Sample2
{
    public:
    Sample2() {}

    template<typename T>
    void plus(T a, T b) {
        std::cout << a + b << std::endl;
    }
};



int main()
{
    Sample<5> s1;
    s1.show();
    std::cout << "DONE" << std::endl;

    Sample2 s2;
    std::string a = "1";
    std::string b = "2";
    s2.plus(1, 2);
    s2.plus(a, b);
    std::cout << "DONE" << std::endl;

    /*
    MyStack<int, 10> mystack;
    mystack.push(1);
    mystack.push(2);
    
    std::cout << mystack.pop() << ", " << mystack.pop() << std::endl;
    std::cout << std::endl;
    */

    return 0;
}
