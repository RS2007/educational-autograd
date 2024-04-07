- [ ] functions like relu and dropout does not work on tensor outputs
    - POSSIBLE FIX: tensor abstraction?
- [ ] Getting mnist to run
- [ ] Implementing cross entropy loss for classifiers


# Base assumptions

- always pass in python lists to tensor init(if there are other cases, handle later )
- Karpathy's value abstraction and the tensor abstraction are not one to one, hence there is a need to have some other abstraction that handles Ops for them
- Function abstraction in tinygrad?  
    - Add inherits a Function(function defined to be something with a forward and backward pass)
    - function applied on tensors while invoking op  

- Tensors should hold a backward method and functions should have a backward method?
- backward should be applicable for results of an op, what if we store the backward method within the op
- Problem with calling backward on op is that it doesnt have current nodes grad information, pass that in as an argument, therefore the backward prototype would look something like `backward(parents,grad)`


# Running mnist

- [x] First get the pytorch version working
- [ ] Tensor abstraction
    - [ ] Forward
        - [ ] Add
        - [ ] Mul
        - [ ] Pow
        - [ ] Neg
        - [ ] Relu
        - [ ] Dropout
    - [ ] Backward
        - [ ] Add
        - [ ] Mul
        - [ ] Pow
        - [ ] Neg
        - [ ] Relu
        - [ ] Dropout
