#ifndef CUSTOM_STACK_HPP
#define CUSTOM_STACK_HPP

template<class T, int i>
class MyStack
{
private:
    T* pStack;
    T StackBuffer[i];
    static const int cItems = i * sizeof(T);
public:
    MyStack(void);
    void push(const T item);
    T& pop(void);
};

#endif