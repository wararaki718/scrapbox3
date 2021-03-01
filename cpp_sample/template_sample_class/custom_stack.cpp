#ifndef CUSTOM_STACK_CPP
#define CUSTOM_STACK_CPP

#include "custom_stack.hpp"

template<class T, int i>
MyStack<T, i>::MyStack(void){
    pStack = StackBuffer;
};

template<class T, int i>
void MyStack<T, i>::push(const T item) {
    *pStack = item;
    pStack++;
};

template<class T, int i>
T& MyStack<T, i>::pop(void) {
    --pStack;
    return *pStack;
};

#endif
