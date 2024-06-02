## A Tiny tensor library + an autograd engine

- To run examples:
```bash
    PYTHONPATH="." python3 ./examples/or_gate.py
```
- For MNIST: check `./mnist.ipynb`

### TODO:
- [x] A functional tensor library
- [x] Passing Backward passes with tests
- [x] MLP working
- [ ] Basic CNN
- [x] CIFAR-10/MNIST working with the lib (MNIST working)
- [ ] Add a different backend than numpy
    - Current plan is to convert the autograd graph into a set of basic ops and compile that to C/Metal/CUDA(not decided yet) kernels
    - Yes this is inspired from tinygrad

> [!WARNING]
> Solely for teaching myself how pytorch works
