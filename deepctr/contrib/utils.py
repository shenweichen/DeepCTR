from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell import *
from tensorflow.python.util import nest

_BIAS_VARIABLE_NAME = "bias"

_WEIGHTS_VARIABLE_NAME = "kernel"


class _Linear_(object):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.



    Args:

      args: a 2D Tensor or a list of 2D, batch x n, Tensors.

      output_size: int, second dimension of weight variable.

      dtype: data type for variables.

      build_bias: boolean, whether to build a bias variable.

      bias_initializer: starting value to initialize the bias

        (default is all zeros).

      kernel_initializer: starting value to initialize the weight.



    Raises:

      ValueError: if inputs_shape is wrong.

    """

    def __init__(self,

                 args,

                 output_size,

                 build_bias,

                 bias_initializer=None,

                 kernel_initializer=None):

        self._build_bias = build_bias

        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")

        if not nest.is_sequence(args):

            args = [args]

            self._is_sequence = False

        else:

            self._is_sequence = True

        # Calculate the total size of arguments on dimension 1.

        total_arg_size = 0

        shapes = [a.get_shape() for a in args]

        for shape in shapes:

            if shape.ndims != 2:
                raise ValueError(
                    "linear is expecting 2D arguments: %s" % shapes)

            if shape[1] is None:

                raise ValueError("linear expects shape[1] to be provided for shape %s, "

                                 "but saw %s" % (shape, shape[1]))

            else:

                total_arg_size += int(shape[1])#.value

        dtype = [a.dtype for a in args][0]

        scope = vs.get_variable_scope()

        with vs.variable_scope(scope) as outer_scope:

            self._weights = vs.get_variable(

                _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],

                dtype=dtype,

                initializer=kernel_initializer)

            if build_bias:

                with vs.variable_scope(outer_scope) as inner_scope:

                    inner_scope.set_partitioner(None)

                    if bias_initializer is None:
                        bias_initializer = init_ops.constant_initializer(
                            0.0, dtype=dtype)

                    self._biases = vs.get_variable(

                        _BIAS_VARIABLE_NAME, [output_size],

                        dtype=dtype,

                        initializer=bias_initializer)

    def __call__(self, args):

        if not self._is_sequence:
            args = [args]

        if len(args) == 1:

            res = math_ops.matmul(args[0], self._weights)

        else:

            res = math_ops.matmul(array_ops.concat(args, 1), self._weights)

        if self._build_bias:
            res = nn_ops.bias_add(res, self._biases)

        return res


try:
    from tensorflow.python.ops.rnn_cell_impl import _Linear
except:
    _Linear = _Linear_


class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:

      num_units: int, The number of units in the GRU cell.

      activation: Nonlinearity to use.  Default: `tanh`.

      reuse: (optional) Python boolean describing whether to reuse variables

       in an existing scope.  If not `True`, and the existing scope already has

       the given variables, an error is raised.

      kernel_initializer: (optional) The initializer to use for the weight and

      projection matrices.

      bias_initializer: (optional) The initializer to use for the bias.

    """

    def __init__(self,

                 num_units,

                 activation=None,

                 reuse=None,

                 kernel_initializer=None,

                 bias_initializer=None):

        super(QAAttGRUCell, self).__init__(_reuse=reuse)

        self._num_units = num_units

        self._activation = activation or math_ops.tanh

        self._kernel_initializer = kernel_initializer

        self._bias_initializer = bias_initializer

        self._gate_linear = None

        self._candidate_linear = None

    @property
    def state_size(self):

        return self._num_units

    @property
    def output_size(self):

        return self._num_units

    def __call__(self, inputs, state, att_score):

        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""

        if self._gate_linear is None:

            bias_ones = self._bias_initializer

            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)

            with vs.variable_scope("gates"):  # Reset gate and update gate.

                self._gate_linear = _Linear(

                    [inputs, state],

                    2 * self._num_units,

                    True,

                    bias_initializer=bias_ones,

                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(

                    [inputs, r_state],

                    self._num_units,

                    True,

                    bias_initializer=self._bias_initializer,

                    kernel_initializer=self._kernel_initializer)

        c = self._activation(self._candidate_linear([inputs, r_state]))

        new_h = (1. - att_score) * state + att_score * c

        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).

    Args:

      num_units: int, The number of units in the GRU cell.

      activation: Nonlinearity to use.  Default: `tanh`.

      reuse: (optional) Python boolean describing whether to reuse variables

       in an existing scope.  If not `True`, and the existing scope already has

       the given variables, an error is raised.

      kernel_initializer: (optional) The initializer to use for the weight and

      projection matrices.

      bias_initializer: (optional) The initializer to use for the bias.

    """

    def __init__(self,

                 num_units,

                 activation=None,

                 reuse=None,

                 kernel_initializer=None,

                 bias_initializer=None):

        super(VecAttGRUCell, self).__init__(_reuse=reuse)

        self._num_units = num_units

        self._activation = activation or math_ops.tanh

        self._kernel_initializer = kernel_initializer

        self._bias_initializer = bias_initializer

        self._gate_linear = None

        self._candidate_linear = None

    @property
    def state_size(self):

        return self._num_units

    @property
    def output_size(self):

        return self._num_units

    def __call__(self, inputs, state, att_score):

        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""

        if self._gate_linear is None:

            bias_ones = self._bias_initializer

            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(
                    1.0, dtype=inputs.dtype)

            with vs.variable_scope("gates"):  # Reset gate and update gate.

                self._gate_linear = _Linear(

                    [inputs, state],

                    2 * self._num_units,

                    True,

                    bias_initializer=bias_ones,

                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))

        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state

        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(

                    [inputs, r_state],

                    self._num_units,

                    True,

                    bias_initializer=self._bias_initializer,

                    kernel_initializer=self._kernel_initializer)

        c = self._activation(self._candidate_linear([inputs, r_state]))

        u = (1.0 - att_score) * u

        new_h = u * state + (1 - u) * c

        return new_h, new_h
