## A Tiny tensor library + an autograd engine

### TODO:
- [ ] A functional tensor library
- [ ] Passing Backward passes with tests
- [ ] MLP working
- [ ] Basic CNN
- [ ] CIFAR-10/MNIST working with the lib
- [ ] Add a different backend than numpy
    - Current plan is to convert the autograd graph into a set of basic ops and compile that to C/Metal/CUDA(not decided yet) kernels
    - Yes this is inspired from tinygrad

> [!WARNING]
> Solely for teaching myself how pytorch works
