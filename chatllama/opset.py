import torch

# the default OP has only torch implementation
class op(object):
    def __init__(self):
        pass

    def forward(self):
        pass

class embedding(op):
    def __init__(self, weight) -> None:
        self.weight = torch.clone(weight)

    def __call__(self, input):
        return torch.nn.functional.embedding(input, self.weight)

    def __repr__(self):
        return f"OP_embedding(weight: {self.weight.shape}{self.weight.dtype})"

class rms_norm(op):
    def __init__(self, weight, variance_epsilon) -> None:
        self.weight = torch.clone(weight)
        self.variance_epsilon = variance_epsilon
        pass

    def __call__(self, input):
        input_dtype = input.dtype
        input = input.to(torch.float32)
        variance = input.pow(2).mean(-1, keepdim=True)
        input = input * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * input.to(input_dtype)

    def __repr__(self):
        return f"OP_rms_norm(weight: {self.weight.shape}{self.weight.dtype}, esp:{self.variance_epsilon})"

class fc_f32(op):
    def __init__(self, weight, bias) -> None:
        print(weight.shape)
        # weight.shape : [N, K]
        self.N = weight.shape[0]
        self.bias = torch.clone(bias) if bias is not None else bias
        self.weight = weight

    def __call__(self, input):
        assert(len(input.shape) == 3)
        return torch.nn.functional.linear(input, self.weight, self.bias)

    def __repr__(self):
        return f"OP_fc_f32(weight: {self.weight.shape}{self.weight.dtype}" + ")" if self.bias is None else f", bias: {self.bias.shape}{self.bias.dtype})"




# OP type system/fields
class tensor(object):
    var_id = 0

    @classmethod
    def from_aten(cls, a : torch.Tensor):
        memview = a.detach().numpy().data
        info = f"{memview.shape} {memview.format}"
        return cls("const", value=memview, info=info)

    def __init__(self, op:str, *args, **kwargs):
        self.op = op
        self.args = []
        for i, a in enumerate(args):
            if issubclass(type(a), torch.Tensor):
                # automatically construct "const" OP from torch tensor
                self.args.append(tensor.from_aten(a))
            elif a is None:
                self.args.append(None)
            else:
                if type(a) is not tensor:
                    raise Exception(f"{i}'th argument with unsupported type : {type(a)}")
                self.args.append(a)
        self.kwargs = kwargs
        self.id = tensor.var_id
        tensor.var_id += 1

    def __add__(self, rhs):
        return tensor("add", self, rhs)

    def __sub__(self, rhs):
        return tensor("sub", self, rhs)

    def __mul__(self, rhs):
        return tensor("mul", self, rhs)

    def _get_ordered_ops(self):
        ops = []
        for a in self.args:
            if a:
                ops += a._get_ordered_ops()
        ops.append(self)
        return ops

    def _get_ssa(self):
        ssa = ""
        ops = self._get_ordered_ops()
        ops_set = set()
        for op in ops:
            # avoid duplicate
            if op in ops_set:
                continue
            ops_set.add(op)

            if op.kwargs:
                str_kwargs = f"{op.kwargs}"
            else:
                str_kwargs = ""
            arg_names = ",".join([f"v{a.id}" if a else "_" for a in op.args])
            ssa += f"\t v{op.id} = {op.op}{str_kwargs}({arg_names})\n"
        return ssa

    def __repr__(self):
        return self._get_ssa()

def ones(shape, dtype=None):
    return tensor("ones", shape=shape)

def fc(x:tensor, w:tensor) -> tensor:
    return tensor("fc", x, w)
