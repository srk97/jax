# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stage out a jitted function with multiple arguments.
"""

import jax
import jax.numpy as jnp
from jax.stages import Lowered
from jax.experimental.australis import exporter


@jax.jit
def f(x, y):
  return (x[0] + x[1] * y, x[0], x[1]), y + 1, x


def lower() -> Lowered:
  return f.lower(
        (jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32')),
         jax.ShapeDtypeStruct((2, 3), jnp.dtype('float32'))),
      jax.ShapeDtypeStruct((), jnp.dtype('float32')))


if __name__ == '__main__':
  exporter.run(lower)
