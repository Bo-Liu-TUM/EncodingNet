from .node import OperatorNode


class Identity(OperatorNode):
    """ A node that copy its input."""
    _arity = 1
    _def_output = "x_0"
    custom_attr_area = 0.


class NOT(OperatorNode):
    """ A node that inverts its input."""
    _arity = 1
    _def_output = "torch.logical_not(x_0)"
    custom_attr_area = 2 / 3


class OR2(OperatorNode):
    """ A node that OR its input."""
    _arity = 2
    _def_output = "torch.logical_or(x_0, x_1)"
    custom_attr_area = 4 / 3


class NOR2(OperatorNode):
    """ A node that NOR its input."""
    _arity = 2
    _def_output = "torch.logical_not(torch.logical_or(x_0, x_1))"
    custom_attr_area = 1.


class AND2(OperatorNode):
    """ A node that AND its input."""
    _arity = 2
    _def_output = "torch.logical_and(x_0, x_1)"
    custom_attr_area = 4 / 3


class NAND2(OperatorNode):
    """ A node that NAND its input."""
    _arity = 2
    _def_output = "torch.logical_not(torch.logical_and(x_0, x_1))"
    custom_attr_area = 1.


class XOR2(OperatorNode):
    """ A node that XOR its input."""
    _arity = 2
    _def_output = "torch.logical_xor(x_0, x_1)"
    custom_attr_area = 2.


class XNOR2(OperatorNode):
    """ A node that XNOR its input."""
    _arity = 2
    _def_output = "torch.logical_not(torch.logical_xor(x_0, x_1))"
    custom_attr_area = 2.
