from .node import OperatorNode


class ConstantTrue(OperatorNode):
    """ A node with a constant True output ."""
    _arity = 0
    # _def_output = "1"
    # _def_numpy_output = "np.ones(len(x[0]),dtype=np.int8)"  # ,dtype=np.bool
    # _def_torch_output = "torch.ones(1,dtype=torch.int8).expand(x.shape[0])"  # ,dtype=torch.bool
    # _def_output = "True"
    # _def_numpy_output = "np.ones(len(x[0]),dtype=np.bool)"  # ,dtype=np.bool
    # _def_torch_output = "torch.ones(1,dtype=torch.bool).expand(x.shape[0])"  # ,dtype=torch.bool
    _def_output = "torch.tensor(True).cuda()"
    # _def_output = "np.ones(1,dtype=np.bool)"
    custom_attr_area = 0.


class ConstantFalse(OperatorNode):
    """ A node with a constant False output ."""
    _arity = 0
    # _def_output = "0"
    # _def_numpy_output = "np.zeros(len(x[0]),dtype=np.int8)"  # ,dtype=np.bool
    # _def_torch_output = "torch.zeros(1,dtype=torch.int8).expand(x.shape[0])"  # ,dtype=torch.bool
    # _def_output = "False"
    # _def_numpy_output = "np.zeros(len(x[0]),dtype=np.bool)"  # ,dtype=np.bool
    # _def_torch_output = "torch.zeros(1,dtype=torch.bool).expand(x.shape[0])"  # ,dtype=torch.bool
    _def_output = "torch.tensor(False).cuda()"
    # _def_output = "np.zeros(1,dtype=np.bool)"
    custom_attr_area = 0.


class Identity(OperatorNode):
    """ A node that inverts its input."""
    _arity = 1
    _def_output = "x_0"
    custom_attr_area = 0.


class NOT(OperatorNode):
    """ A node that inverts its input."""
    _arity = 1
    # _def_output = "1 - x_0"
    # _def_output = "np.logical_not(x_0)"
    _def_output = "torch.logical_not(x_0)"
    # _def_output = "~bool(x_0)"
    # _def_numpy_output = "~x_0.astype(np.bool)"
    # _def_torch_output = "~x.bool()"
    custom_attr_area = 2 / 3


class OR2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "x_0 + x_1 - x_0 * x_1"
    # _def_output = "np.logical_or(x_0, x_1)"
    _def_output = "torch.logical_or(x_0, x_1)"
    # _def_output = "x_0 | x_1"
    custom_attr_area = 4 / 3


class NOR2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "1 + x_0 * x_1 - x_0 - x_1"
    # _def_output = "~(x_0 | x_1)"
    # _def_output = "np.logical_not(np.logical_or(x_0, x_1))"
    _def_output = "torch.logical_not(torch.logical_or(x_0, x_1))"
    custom_attr_area = 1.


class AND2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "x_0 * x_1"
    # _def_output = "x_0 & x_1"
    # _def_output = "np.logical_and(x_0, x_1)"
    _def_output = "torch.logical_and(x_0, x_1)"
    custom_attr_area = 4 / 3


class NAND2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "1 - x_0 * x_1"
    # _def_output = "~(x_0 & x_1)"
    # _def_output = "np.logical_not(np.logical_and(x_0, x_1))"
    _def_output = "torch.logical_not(torch.logical_and(x_0, x_1))"
    custom_attr_area = 1.


class XOR2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "x_0 + x_1 - 2 * x_0 * x_1"
    # _def_output = "x_0 ^ x_1"
    # _def_output = "np.logical_xor(x_0, x_1)"
    _def_output = "torch.logical_xor(x_0, x_1)"
    custom_attr_area = 2.


class XNOR2(OperatorNode):
    """ A node that inverts its input."""
    _arity = 2
    # _def_output = "1 + 2 * x_0 * x_1 - x_0 - x_1"
    # _def_output = "~(x_0 ^ x_1)"
    # _def_output = "np.logical_not(np.logical_xor(x_0, x_1))"
    _def_output = "torch.logical_not(torch.logical_xor(x_0, x_1))"
    custom_attr_area = 2.
