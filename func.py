# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# From flax.nnx.nn.recurrent
"""RNN modules for Flax."""

from typing import Any, TypeVar
from collections.abc import Callable
from functools import partial
from typing_extensions import Protocol
from absl import logging

from linearRNN import forward_h
from linearRNN import forward
from linearRNN import init_lru_parameters

import jax
import jax.numpy as jnp

from flax import nnx
from flax.nnx import rnglib
from flax.nnx.module import Module
from flax.nnx.nn import initializers
from flax.nnx.nn.linear import Linear
from flax.nnx.nn.activations import sigmoid
from flax.nnx.nn.activations import tanh
from flax.nnx.transforms.iteration import Carry
from flax.typing import Dtype, Initializer, Shape
from flax.nnx.nn.recurrent import RNNCellBase

default_kernel_init = initializers.lecun_normal()
default_bias_init = initializers.zeros_init()

A = TypeVar("A")
Array = jax.Array
Output = Any


class LRNNCell(RNNCellBase):
    r"""Linear RNN cell.

    The mathematical definition of the cell is as follows

    .. math::

        \begin{array}{ll}
        h' = W_i x + b_i + W_h h
        \end{array}

    where x is the input and h is the output of the previous time step.

    If `residual` is `True`,

    .. math::

        \begin{array}{ll}
        h' = W_i x + b_i + W_h h + h
        \end{array}
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,  # not inferred from carry for now
        *,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        residual: bool = False,
        kernel_init: Initializer = initializers.lecun_normal(),
        recurrent_kernel_init: Initializer = initializers.orthogonal(),
        bias_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = residual
        self.kernel_init = kernel_init
        self.recurrent_kernel_init = recurrent_kernel_init
        self.bias_init = bias_init
        self.rngs = rngs

        # self.hidden_features = carry.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_h = Linear(
            in_features=self.hidden_features,
            out_features=self.hidden_features,
            use_bias=False,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.recurrent_kernel_init,
            rngs=rngs,
        )
        self.dense_i = Linear(
            in_features=self.in_features,
            out_features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            rngs=rngs,
        )

    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:  # type: ignore[override]
        new_carry = self.dense_i(inputs) + self.dense_h(carry)
        if self.residual:
            new_carry += carry
        return new_carry, new_carry

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array:  # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        if rngs is None:
            rngs = self.rngs
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(rngs.carry(), mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1


class DiagonalDense(nnx.Module):
    diag_elements: nnx.Param = nnx.Param(lambda key, shape: jnp.ones(shape))

    def __init__(self, in_features, kernel_init=None, rngs=jax.random.PRNGKey(0)):
        if kernel_init == None:
            self.diag_elements = jax.random.normal(rngs, (in_features,))
        else:
            self.diag_elements = kernel_init

    def __call__(self, x):
        diag_matrix = jnp.diag(
            self.diag_elements
        )  # Only the diagonal elements are learnable
        return jnp.dot(x, diag_matrix)


# From flax.nnx.nn.Linear
from flax.nnx.nn import dtypes, initializers
from flax.typing import (
    Dtype,
    Shape,
    Initializer,
    PrecisionLike,
    DotGeneralT,
    ConvGeneralDilatedT,
    PaddingLike,
    LaxPadding,
)
import typing as tp
from jax import lax


class Dense(Module):
    """A linear transformation applied over the last dimension of the input.

    Example usage::

      >>> from flax import nnx
      >>> import jax, jax.numpy as jnp

      >>> layer = nnx.Linear(in_features=3, out_features=4, rngs=nnx.Rngs(0))
      >>> jax.tree.map(jnp.shape, nnx.state(layer))
      State({
        'bias': VariableState(
          type=Param,
          value=(4,)
        ),
        'kernel': VariableState(
          type=Param,
          value=(3, 4)
        )
      })

    Attributes:
      in_features: the number of input features.
      out_features: the number of output features.
      use_bias: whether to add a bias to the output (default: True).
      dtype: the dtype of the computation (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      precision: numerical precision of the computation see ``jax.lax.Precision``
        for details.
      kernel_init: initializer function for the weight matrix.
      bias_init: initializer function for the bias.
      dot_general: dot product function.
      rngs: rng key.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        use_bias: bool = True,
        dtype: tp.Optional[Dtype] = None,
        param_dtype: Dtype = jnp.float32,
        precision: PrecisionLike = None,
        kernel_init: Initializer = default_kernel_init,
        bias_init: Initializer = default_bias_init,
        dot_general: DotGeneralT = lax.dot_general,
        rngs: rnglib.Rngs,
    ):
        kernel_key = rngs.params()
        self.kernel = nnx.Param(kernel_init)
        if use_bias:
            bias_key = rngs.params()
            self.bias = nnx.Param(bias_init(bias_key, (out_features,), param_dtype))
        else:
            self.bias = nnx.Param(None)

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dot_general = dot_general

    def __call__(self, inputs: Array) -> Array:
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
          inputs: The nd-array to be transformed.

        Returns:
          The transformed input.
        """
        kernel = self.kernel.value
        bias = self.bias.value

        inputs, kernel, bias = dtypes.promote_dtype(
            (inputs, kernel, bias), dtype=self.dtype
        )
        y = self.dot_general(
            inputs,
            kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        assert self.use_bias == (bias is not None)
        if bias is not None:
            y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class LRUCell(RNNCellBase):
    r"""Linear RNN cell.

    The mathematical definition of the cell is as follows

    .. math::

        \begin{array}{ll}
        h' = \Lambda h+\exp(\gamma^{\log})\odot(B x)
        \end{array}

    where x is the input and h is the output of the previous time step.

    If `residual` is `True`,

    .. math::

        \begin{array}{ll}
        h' = \Lambda h+\exp(\gamma^{\log})\odot(B x) + h
        \end{array}
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,  # not inferred from carry for now
        *,
        dtype: Dtype = jnp.float32,
        param_dtype: Dtype = jnp.float32,
        carry_init: Initializer = initializers.zeros_init(),
        residual: bool = False,
        # kernel_init: Initializer = initializers.lecun_normal(),
        # recurrent_kernel_init: Initializer = initializers.orthogonal(),
        # bias_init: Initializer = initializers.zeros_init(),
        rngs: rnglib.Rngs,
    ):
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.carry_init = carry_init
        self.residual = residual
        # self.kernel_init = kernel_init
        # self.recurrent_kernel_init = recurrent_kernel_init
        # self.bias_init = bias_init
        self.rngs = rngs
        (
            self.nu_log,
            self.theta_log,
            self.B_re,
            self.B_im,
            self.C_re,
            self.C_im,
            self.D,
            self.gamma_log,
        ) = init_lru_parameters(
            self.hidden_features, self.in_features, r_min=0.99, r_max=0.999
        )
        self.Lambda = jnp.exp(-jnp.exp(self.nu_log) + 1j * jnp.exp(self.theta_log))
        self.B_norm = (self.B_re + 1j * self.B_im) * jnp.expand_dims(
            jnp.exp(self.gamma_log), axis=-1
        )
        self.B_norm = jnp.real(self.B_norm)
        # self.hidden_features = carry.shape[-1]
        # input and recurrent layers are summed so only one needs a bias.
        self.dense_h = DiagonalDense(
            in_features=self.hidden_features,
            kernel_init=jnp.diag(self.Lambda),
            rngs=rngs,
        )
        self.dense_i = Dense(
            in_features=self.in_features,
            out_features=self.hidden_features,
            use_bias=True,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.B_norm,
            # bias_init=self.bias_init,
            rngs=rngs,
        )

    def __call__(self, carry: Array, inputs: Array) -> tuple[Array, Array]:  # type: ignore[override]
        new_carry = self.dense_i(inputs) + self.dense_h(carry)
        if self.residual:
            new_carry += carry
        return new_carry, new_carry

    def initialize_carry(self, input_shape: tuple[int, ...], rngs: rnglib.Rngs | None = None) -> Array:  # type: ignore[override]
        """Initialize the RNN cell carry.

        Args:
          rng: random number generator passed to the init_fn.
          input_shape: a tuple providing the shape of the input to the cell.

        Returns:
          An initialized carry for the given RNN cell.
        """
        if rngs is None:
            rngs = self.rngs
        batch_dims = input_shape[:-1]
        mem_shape = batch_dims + (self.hidden_features,)
        return self.carry_init(rngs.carry(), mem_shape, self.param_dtype)

    @property
    def num_feature_axes(self) -> int:
        return 1
