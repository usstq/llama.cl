## Static graph topology stays in python
by describing static graph in python, it becomes very convenient to adjust
micro topology in source code by introducing custom Ops or fusing some OPs into
one for optimization purpose, because in following description, the repeatation
of decoding layer is not unrolled and it's easy to change.

this function is actually low level IR generation process. and optimizations
like ops-fusing requiring pattern match based transformation should be done
after graph is generated. and this transformation can also be done in python scope.
so
    1. custom/manually topology adjustment is done first in python script.
    2. automatic topology adjustment (transformation) can be done afterward in lower-level.
        and can be also in python since it's offline transformation.
    3. the final generated graph is serializable and executable in final CPP runtime.
        since most logic are determined offline on python side, and execution graph is
        very detailed, thus CPP runtime becomes very simple.

## Runtime
the final runtime is similar to RISC-VM, executing low-level & detailed instructions which is not expected to be further transformed. all of which have been done offline on compiler side
(which is python code).

instructions are stateless, just like torch.nn.functional. all inputs are tensors eighter
loaded from file (constant) or created by instruction itself at runtime(so each instruction
will derive shape at runtime and allocate output tensor based on it).

python code can serialize the constant tensors & OP instruction sequence into a format that
runtime can unserstand and run. for example, in SSA text form. and this code is not supposed
to be programable by human.

how python generate the runtime is flexible, manually generate using python-written code-gen
is the fundamental way. and the code can also be executed inside python, since it's syntax is
also a valid subset of python (like torch script). it's a good feature because copy the IR
into python enviroment, we can debug it or even use it as reference.

each basic instruction (function) has python reference implementation too.

the automatic generation of this IR is not mandatory, but nice to have, since VM is a extreamly
simplified runtime executing the IR like a dumb interpretor. and the IR is as usual static graph,
but the variable flowing inside is not neccessary tensor, but any object type.

each instruction is allowed to accept object as input and return one output.

object types allowed is mainly tensor, and graph description in python can take advantage of python's
syntax surgar (magic methods), but the IR generated has no syntax surgar of any kind.

all syntax surgar finally becomes a instruction to VM. for example, the python indexing syntax with `__getitem__`
and `__setitem__` magic methods becomes corresponding OP. but we should avoid such syntax because it
usually means low performance.

explicit expression (avoid syntax sugar) is helpful to emphasis the runtime cost.


### <span style="color:cyan"> Convert model from HF & optimize in eager mode (within python ecosystem) </span>
 - construct each OP from model's constant table & config
 - call each op in forward process

this is very convenient, with so many helpers.

### <span style="color:cyan"> Serialization is done with minimum effort </span>
a generic logic can `export` the model:

  - collect all OP used, and generate OP construction code.
  - call forward process but replace the OP executor with OP builder
    so the execution IR is generated.

The key for this design to work is the OP layer.


## Python program to runtime IR

  If we treat python part of source code as a program, then turn it into C++ source code
is not a huge effort with the help of pybind11, since C++ class can easily become a python
class, so OP and tensor are all such stuff, and even more, some complex class like kv-cache
can also be implemented in C++ and exposed as python class, so we can generalize the idea
and find a better way to implement model inference:
  
 **Simply implement C++ type system and expose class/ops to python** 


------------------------------------------------------------------------















two steps are required:
        1. build/construct of each OP, and it's required because there are
        constant to be preprocessed before inference.
        2. calling each OP on tensors.

both graph/eager modes require these 2 steps.
graph mode separates these 2 steps in a way different than eager mode.
in step1 graph mode also encodes tensor flowing information (the whole computational steps).
but in eager mode tensor flowing information is determined dynamically in step2.

all inference framework uses graph mode but model composer prefer pytorch eager mode.
to turn a eager mode code into graph mode, jit script/trace is invented to do that
automatically, but it's not good enough because after jit, the code structure is lost
and any further optimization is more difficult, since we cannot change it easily like
coding in python.

python is not just a glue logic but it provides many advantages for developing, and we
should hide optimizations behind python or after python eager mode being converted to
graph, we should apply optimizations directly in python and convert it back to other
glue logic (graph-runtime) only after all optimizations are done. and graph-runtime should be
as simple as it can, just faithfully expressing the computational steps done in the 
python forwad function.

the OP constructed by python in step1 should be serializable to a file that can
be accessed by both python or graph-runtime, graph runtime can care about only
step2.

So we can define OP in eager mode, and is callable in both python & graph-runtime
a pybind11 object wrapping a C++ object.

    - builder function : usually only python will call it with correct input constant tensor
    - serialization functions : called by python & C++
        https://pybind11.readthedocs.io/en/latest/advanced/classes.html#pickling-support
    - forward function : called by python & C++

serialization OPs and the whole static graph is two thing, but can be combined as one task.
the format of static graph is like ISA of a VM, since runtime is VM-like.

the static graph should be pseudo-code like which is easy to view & study & change.
the best is C++ pseudo-code which can be even compiled with runtime, if we view the
compilation as a part of the offline process too.

the pseudo-code with minimal editing can be compiled with runtime to form a runtime executable.
it would be even better if pseudo-code can keep the repeatative code structure.
