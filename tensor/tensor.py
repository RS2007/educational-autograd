import numpy as np
from typing import List


class Experiment:
    def __init__(self, op_name, args):
        self.op_name = op_name
        self.args = args

    def __repr__(self):
        return f"{self.op_name}"


class Plus:
    @staticmethod
    def forward(*args):
        assert len(args) == 2
        [a, b] = args
        ret = Tensor(a.data + b.data, is_load=False)
        ret.ops = Experiment("plus", [a, b])
        return ret

    @staticmethod
    def backward(parents, grad):
        x, y = parents
        if isinstance(grad, Tensor):
            return grad, grad
        return Tensor([grad], is_load=False), Tensor([grad], is_load=False)


class Max:
    @staticmethod
    def forward(*args):
        assert len(args) == 1
        [a] = args
        ret = Tensor(np.array([np.max(a.data)]), is_load=False)
        ret.ops = Experiment("max", [a])
        return ret

    @staticmethod
    def backward(parents, grad):
        pass


class Transpose:
    @staticmethod
    def forward(*args):
        assert len(args) == 1
        return Tensor(args[0].data.transpose())

    def backward(parents, grad):
        return Tensor(grad.data.transpose())


class Sum:
    @staticmethod
    def forward(*args):
        assert len(args) == 1
        ret = Tensor([np.sum(args[0].data)], is_load=False)
        ret.ops = Experiment("sum", [args[0]])
        return ret

    @staticmethod
    def backward(parents, grad):
        assert len(parents) == 1
        (parent,) = parents
        return Tensor(np.broadcast_to(grad.data, parent.data.shape), is_load=False)


class Mul:
    @staticmethod
    def forward(*args):
        assert len(args) == 2
        [a, b] = args
        return Tensor(a.data * b.data)

    @staticmethod
    def backward(parents, grad):
        x, y = parents
        if isinstance(grad, Tensor):
            return Tensor(y.data * grad.data), Tensor(x.data * grad.data)
        return Tensor(y.data * grad), Tensor(x.data * grad)


class Pow:
    @staticmethod
    def forward(*args):
        assert len(args) == 2
        [a, b] = args
        return Tensor(np.power(a.data, b.data))

    @staticmethod
    def backward(parents, grad):
        [a, b] = parents
        if isinstance(grad, Tensor):
            return Tensor(b.data * np.power(a.data, b.data - 1) * grad.data), None
        return Tensor(b.data * np.power(a.data, b.data - 1) * grad), None


class MatMul:
    @staticmethod
    def forward(*args):
        assert len(args) == 2
        [a, b] = args
        ret = Tensor(a.data @ b.data)
        ret.ops = Experiment("matmul", [a, b])
        return ret

    @staticmethod
    def backward(parents, grad):
        [a, b] = parents
        # TODO: works only for 2d, thats enough for now, change later
        print("")
        print("")
        print("Matmul backward")
        print(grad.data)
        print(a.data.T)
        print(b.data.T)
        print("")
        print("")
        if isinstance(grad, Tensor):
            return Tensor(grad.data @ b.data.T), Tensor(a.data.T @ grad.data)
        print(grad.data)
        print(b.data)
        return Tensor(grad @ b.data.T), Tensor(a.data.T @ grad)


class TempVarGenerator:
    def __init__(self, prefix="temp"):
        self.prefix = prefix
        self.counter = 0

    def next_var(self):
        var_name = f"{self.prefix}{self.counter}"
        self.counter += 1
        return var_name


def generate_array_suffix(buffer_dim):
    string = ""
    for dim in buffer_dim:
        string += f"[{dim}]"
    return string


def gen_load_tensor(arr_type, temp_sym, arr, buffer_shape=None):
    np_to_c_map = {np.dtype("float32"): "float"}
    dims = generate_array_suffix(buffer_shape)
    return f"{np_to_c_map[arr_type]} {temp_sym}{dims} = {np.array2string(arr,separator=',').replace('[','{').replace(']','}')};\n"


def gen_tensor_sum(sym1, sym2, loop_dim, target, buffer_shape=None):
    if buffer_shape is not None:
        buffer_dim = buffer_shape
    else:
        buffer_dim = [loop_dim]
    dims = generate_array_suffix(buffer_dim)
    str0 = f"float {target}{dims};\n"
    str1 = f"for(int i = 0; i < {loop_dim};i++)" + "{" + "\n"
    str2 = f"\t((float*){target})[i] = ((float*){sym1})[i] + ((float*){sym2})[i];\n"
    str3 = "}\n"
    return str0 + str1 + str2 + str3


def gen_sum_loop(arr, loop_dim, target):
    # INFO: float cause of overflows for larger matrices
    str0 = f"float {target}[{1}] =" + "{0.};" + "\n"
    str1 = f"for(int i = 0; i < {loop_dim};i++)" + "{" + "\n"
    str2 = f"\t{target}[0] += ((float*){arr})[i];\n"
    str3 = "}\n"
    return str0 + str1 + str2 + str3


def gen_max_loop(arr, loop_dim, target):
    str0 = f"float {target}[1] = " + "{0.};" + "\n"
    str1 = f"for(int i = 0; i < {loop_dim};i++)" + "{" + "\n"
    str2 = f"\t{target}[0] = max({target}[0], ((float*){arr})[i]);\n"
    str3 = "}\n"
    return str0 + str1 + str2 + str3


def gen_matmul(arr1, arr2, loop_dims, out_shape, target):
    dims = generate_array_suffix(out_shape)
    str0 = f"float {target}{dims};\n"
    str1 = f"for(int i = 0; i < {loop_dims[0]}; i++)" + "{" + "\n"
    str2 = f"\tfor(int j = 0; j < {loop_dims[1]}; j++)" + "{" + "\n"
    str3 = f"\t\t{target}[i][j] = 0;\n"
    str4 = f"\t\tfor(int k = 0; k < {loop_dims[2]}; k++)" + "{" + "\n"
    str5 = f"\t\t\t{target}[i][j] += ({arr1})[i][k] * ({arr2})[k][j];\n"
    str6 = "\t\t}\n"
    str7 = "\t}\n"
    str8 = "}\n"
    # void matrix_multiply(float A[ROW1][COL1], float B[COL1][COL2], float C[ROW1][COL2]) {
    #     for (int i = 0; i < ROW1; i++) {
    #         for (int j = 0; j < COL2; j++) {
    #             C[i][j] = 0;
    #             for (int k = 0; k < COL1; k++) {
    #                 C[i][j] += A[i][k] * B[k][j];
    #             }
    #         }
    #     }
    # }
    return str0 + str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8


class Tensor:
    # TODO: Unfinished abstraction, need to make lots of changes
    def __init__(self, arr, parents=None, op=None, requires_grad=False, is_load=True):
        self.data = np.array(arr, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self.op = op if op is not None else None
        self.parents = parents if parents is not None else None
        self._backward = lambda: None
        self.ops = Experiment("load", ()) if is_load else None

    def print_ops(self, depth=0):
        if self.ops is None:
            return
        print("\t" * depth + self.ops.__repr__())
        for arg in self.ops.args:
            arg.print_ops(depth + 1)

    def transpose(self):
        val = Transpose.forward(self)
        val.op = Transpose.backward
        val.parents = [self]
        return val

    def codegen(self):
        visited = {}
        queue: List[Tensor] = []

        def _topo(graph):
            if graph in visited:
                return
            if graph.parents is not None:
                for parent in graph.parents:
                    _topo(parent)
            queue.append(graph)
            visited[graph] = True

        _topo(self)
        code = ""
        scope = {}
        tempVarGenerator = TempVarGenerator()
        for inst in queue:
            temp_sym = tempVarGenerator.next_var() if inst != queue[-1] else "out"
            scope[inst] = temp_sym
            buffer_shape = None if inst != queue[-1] else inst.data.shape
            if inst.ops.op_name == "load":
                code += gen_load_tensor(
                    inst.data[0].dtype,
                    temp_sym,
                    inst.data,
                    buffer_shape=inst.data.shape,
                )
            elif inst.ops.op_name == "plus":
                # TODO: replace with the accelerators instruction
                loop_dim = np.prod((inst.parents[0].data.shape))  # Only for 1D tensor
                code += gen_tensor_sum(
                    scope.get(inst.parents[0]),
                    scope.get(inst.parents[1]),
                    loop_dim,
                    temp_sym,
                    buffer_shape=inst.parents[0].data.shape,
                )
            elif inst.ops.op_name == "sum":
                # Dont know if theres basic op for this in the accelerator, if not this will evaluate it within the shakti core
                loop_dim = np.prod(inst.parents[0].data.shape)  # Only for 1D tensor
                code += gen_sum_loop(scope.get(inst.parents[0]), loop_dim, temp_sym)
            elif inst.ops.op_name == "max":
                code += gen_max_loop(scope.get(inst.parents[0]), loop_dim, temp_sym)
            elif inst.ops.op_name == "matmul":
                loop_dims = [
                    inst.parents[0].data.shape[0],
                    inst.parents[1].data.shape[1],
                    inst.parents[0].data.shape[1],
                ]
                out_shape = [
                    inst.parents[0].data.shape[0],
                    inst.parents[1].data.shape[1],
                ]
                code += gen_matmul(
                    scope.get(inst.parents[0]),
                    scope.get(inst.parents[1]),
                    loop_dims,
                    out_shape,
                    temp_sym,
                )

        return code

    def zero_grad(self):
        if self.grad is not None:
            self.grad = Tensor(np.zeros_like(self.grad))
        else:
            self.grad = Tensor(np.zeros_like(self.data).astype(np.float32))

    def max(self):
        val = Max.forward(self)
        val.op = Max.backward
        val.parents = [self]
        return val

    def __add__(self, b):
        val = Plus.forward(self, b)
        val.op = Plus.backward
        val.parents = [self, b]
        return val

    def __sub__(self, b):
        return self + (-b)

    def __neg__(self):
        return self * [-1]

    def __mul__(self, b):
        if not isinstance(b, Tensor):
            b = Tensor(b)
        val = Mul.forward(self, b)
        val.op = Mul.backward
        val.parents = [self, b]
        return val

    def __repr__(self):
        return f"data: {self.data} and grad: {self.grad} with op: {self.op}"

    def __pow__(self, n):
        if not isinstance(n, Tensor):
            n = Tensor(n)
        val = Pow.forward(self, n)
        val.op = Pow.backward
        val.parents = [self, n]
        return val

    def __matmul__(self, b):
        if not isinstance(b, Tensor):
            b = Tensor(b)
        val = MatMul.forward(self, b)
        val.op = MatMul.backward
        val.parents = [self, b]
        return val

    def __truediv__(self, b):
        raise NotImplementedError

    def sum(self):
        val = Sum.forward(self)
        val.op = Sum.backward
        val.parents = [self]
        return val

    def mean(self):
        raise NotImplementedError

    def backward(self):
        if self.op is None:
            return

        if self.grad is None:
            assert (
                self.data.size == 1
            ), f"Backward should be called only for scalar tensors, got shape {self.data.shape}"
            self.grad = Tensor([1.0])
        visited = {}
        queue = []

        def _topo(graph):
            if graph in visited:
                return
            if graph.parents is not None:
                for parent in graph.parents:
                    _topo(parent)
            queue.append(graph)
            visited[graph] = True

        _topo(self)
        print(queue)
        for node in reversed(queue):
            if node.op is None:
                continue
            print(f"Calling back on {node} with {node.parents}")
            grads = node.op(node.parents, node.grad)
            if len(node.parents) == 1:
                grads = [grads]
            for tensor, grad in zip(node.parents, grads):
                print(tensor.grad)
                if grad is None:
                    continue
                if tensor.grad is None:
                    tensor.grad = Tensor(np.zeros_like(tensor.data).astype(np.float32))
                print(grad.data)
                print(tensor.data)
                tensor.grad.data += grad.data


class Value:
    def __init__(self, val, parents=None, op=None):
        self._val = val
        self._grad = 0.0
        self._parents = None if parents is None else np.array([*parents])
        self._op = None if op is None else op
        self._backward = lambda: None  # closures for storage

    def __repr__(self):
        return f"val: {self._val} and grad: {self._grad} and op: {self._op}"

    def __add__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        val = Value(self._val + b._val, [self, b], "+")

        def _backward():
            self._grad += 1 * val._grad
            b._grad += 1 * val._grad

        val._backward = _backward
        return val

    def __neg__(self):
        return self * (Value(-1))

    def __sub__(self, b):
        return self + (-b)

    def __truediv__(self, b):
        if not isinstance(b, Value):
            b = Value(b)
        return self * (b**-1)

    def __pow__(self, n):
        val = Value(self._val**n, [self], f"**{n}")

        def _backward():
            self._grad += n * (self._val ** (n - 1)) * val._grad

        val._backward = _backward
        return val

    def __mul__(self, b):
        if not isinstance(b, Value):
            b = Value(b)

        val = Value(self._val * b._val, [self, b], "-")

        def _backward():
            self._grad += b._val * val._grad
            b._grad += self._val * val._grad

        val._backward = _backward
        return val

    def relu(self):
        val = Value(0 if self._val < 0 else self._val, [self], "relu")

        def _backward():
            self._grad += (1 if self._val > 0 else 0) * val._grad

        val._backward = _backward
        return val

    def dropout(self, p):
        assert isinstance(p, float)
        val = Value(
            np.random.choice([self._val * (1.0 / (1.0 - p)), 0], p=[1 - p, p]),
            [self],
            "dropout",
        )

        def _backward():
            self._grad += val._grad

        val._backward = _backward
        return val

    def backward(self):
        visited = {}
        queue = []

        def _topo(graph):
            if graph in visited:
                return
            if graph._parents is not None:
                for parent in graph._parents:
                    _topo(parent)
            queue.append(graph)
            visited[graph] = True

        _topo(self)
        self._grad = 1.0
        for node in reversed(queue):
            node._backward()

    def zero_grad(self):
        self._grad = 0
