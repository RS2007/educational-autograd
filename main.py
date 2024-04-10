#!/usr/bin/env python3.11
import subprocess
from tensor.tensor import Tensor
import numpy as np

if __name__ == "__main__":
    # a = Tensor(np.random.rand(2, 2).tolist())
    # b = Tensor(np.random.rand(2, 2).tolist())
    a = Tensor(np.random.rand(8, 8).tolist())
    b = Tensor(np.random.rand(8, 8).tolist())
    c = a @ b.transpose()
    d = c + b
    e = d @ a
    g = e + b
    h = (g - b) ** 2
    f = g.sum()
    f.backward()
    # c = a + b
    # d = c.max()
    # e = Tensor(np.random.rand(1).tolist())
    # f = d + e
    print("NUMPY EVAL:")
    print(f)
    print("AST: ")
    print(f.print_ops())
    print("Generated code: ")
    # C_PREFIX = "#include<stdio.h>\n#define max(x,y) x > y?x:y\nint main(){\n"
    # code = C_PREFIX
    # for line in f.codegen().split("\n"):
    #     code += "\t" + line + "\n"
    # code += '\tprintf("%f\\n", out[0]);\n'
    # code += "\treturn 0;\n}"
    # fd = open("a.c", "w")
    # fd.write(code)
    # fd.close()
    # subprocess.check_output("clang -O3 a.c".split())
    # result = subprocess.run(["./a.out"], capture_output=True)
    # c_output = float(result.stdout)
    # print(f"Got c output to be {c_output} and numpy output to be: {f.data[0]}")

    # without any torch like abstraction, with the current codegen how to run an mlp?
    # forward method is one matmul and one tensor add
    # implement matmul
    # a1 = Tensor(np.random.rand(28*28).tolist())
    # w1 = Tensor(np.random.rand(512,28*28))
    # b1 = Tensor(np.random.rand(512).tolist())
    # r1 = a1 @ w1 + b1
    # w2 = Tensor(np.random.rand(512,512).tolist())
    # b2 = Tensor(np.random.rand(512).tolist())
    # r2 = r1 @ w2 + b2
    # w3 = Tensor(np.random.rand(10,512).tolist())
    # b3 = Tensor(np.random.rand(10).tolist())
    # r3 = r2 @ w3 + b3
    # print(r3)
