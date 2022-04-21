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

"""exporter.py exports jax.jit(fn).lower(x) to be embedded in a c++ program.

Usage:

```
import jax
import jax.numpy as jnp

from jax.experimental import exporter

@jax.jit
def square(x):
  return x * x

if __name__ == "__main__":
  exporter.run(lambda: square.lower(jax.ShapeDtypeStruct((2, 4), jnp.float32)))
```
"""

from typing import Any, Callable, List, Sequence, Tuple, Union

import re
from absl import app
from absl import flags
import jax
import jax.experimental.maps
from jax.interpreters.pxla import Mesh
from jax.stages import Lowered
import numpy as np
import sys
import struct

FLAGS = flags.FLAGS

flags.DEFINE_string('header_name', None, 'Path to header.')
flags.DEFINE_string('impl_name', None, 'Path to impl.')
flags.DEFINE_string('cc_embed_impl_name', None, 'Path to cc_embed impl.')
flags.DEFINE_string('platform', 'cpu',
                    'String hardware platform.')  # TODO(mattjj): Enum?
flags.DEFINE_string('cc_namespace', 'jax::experimental',
                    'C++ namespace to use for generated code.')
flags.DEFINE_string('name', 'Compute', 'Function name.')
flags.DEFINE_bool(
    'canonicalize_function_name', True,
    'Set to false to use the name as specified without '
    'canonicalization.')

Proto = Any
Array = Any
Device = Any
DType = Any


class FakeDevice:

  def __init__(self, client, i, platform):
    self.client = client
    self.id = i
    self.process_index = 0
    self.platform = platform


class FakeClient:

  def __init__(self, n, platform):
    self.platform = platform
    self.process_index = 0
    self.devices = [FakeDevice(self, i, platform) for i in range(n)]


def fake_devices(n, platform):
  return FakeClient(n, platform).devices


def is_pjit(l: Lowered) -> bool:
  return 'mesh' in l._lowering.compile_args  # pytype: disable=attribute-error


def devices(l: Lowered) -> np.ndarray:
  mesh: Mesh = l._lowering.compile_args['mesh']  # pytype: disable=attribute-error
  return mesh.devices


def to_include_guard(value):
  return '_AUSTRALIS_' + re.sub('[^0-9a-zA-Z]+', '_', value).upper()


def to_data_symbol(value):
  return '_australis_' + re.sub('[^0-9a-zA-Z]+', '_', value).lower()


def fn_arity(staged: Lowered) -> int:
  return len(staged.in_tree.children()[0].children())


def argument_types(staged: Lowered) -> str:
  arity = fn_arity(staged)
  arg_text = []
  donated_args = set(staged.donate_argnums)
  for i in range(arity):
    # TODO(saeta): make into names & types!
    if i in donated_args:
      arg_text.append(f'aux::PTree')
    else:
      arg_text.append(f'const aux::PTree&')
  return ', '.join(arg_text)


def argument_names(staged: Lowered) -> List[str]:
  return list(map(lambda x: f'x{x}', range(fn_arity(staged))))


def embed_c_string(bin_text, *, sym, file):
  bin_c_data = '\n'.join("  \"%s\"" %
                         ''.join(_chrs[b]
                                 for b in bin_text[i * 19:(i + 1) * 19])
                         for i in range((len(bin_text) + 18) // 19))
  print(
      """static constexpr std::string_view %s = {
%s, %d};
""" % (sym, bin_c_data, len(bin_text)),
      file=file)


_chrs = ['\\x%02x' % i for i in range(256)]


def _canonicalize_function_name(name: str) -> str:
  components = name.split('_')
  return ''.join(map(lambda s: s.capitalize(), components))


def serialize_executable_spec(fn):
  """Temporary low-dependency serialization format for executable specifications."""
  out, scopes = [], []

  def pop_scope():
    i = scopes.pop()
    out[i] = len(out) - 1 - i

  def push_scope():
    scopes.append(len(out))
    out.append(-1)

  push_scope()

  if is_pjit(fn):
    num_replicas = 1
    num_partitions = len(fn._lowering.compile_args['mesh'].devices)  # pytype: disable=attribute-error
    use_spmd = 1
  else:
    num_replicas = 1
    num_partitions = 1
    use_spmd = 0

  out += [0, num_replicas, num_partitions, use_spmd]
  out.append(1)

  def encode_pytree(pytree):
    push_scope()
    if pytree.children():
      out.append(1)
      for child in pytree.children():
        encode_pytree(child)
    else:
      out.append(0)
    pop_scope()

  encode_pytree(fn.out_tree)

  pop_scope()

  return struct.pack('<%si' % len(out), *out), 1


def run(user_fn: Callable[[], Lowered]):

  def main(argv: Sequence[str]) -> None:
    fns = user_fn()

    if not isinstance(fns, list):
      fns = [(FLAGS.name, fns)]

    with open(FLAGS.header_name, 'w') as header_file,\
         open(FLAGS.impl_name, 'w') as impl_file,\
         open(FLAGS.cc_embed_impl_name, 'w') as embed_impl_file:
      print(
          f"""\
#ifndef { to_include_guard(FLAGS.header_name) }
#define { to_include_guard(FLAGS.header_name) }

#include "jax/experimental/australis/australis_computation.h"

namespace {FLAGS.cc_namespace} {{
""",
          file=header_file)
      print(
          f"""\
#include "{ FLAGS.header_name }"

#include "{ FLAGS.cc_embed_impl_name }"

using namespace aux;

namespace {FLAGS.cc_namespace} {{
""",
          file=impl_file)

      for executable_id, (name, fn) in enumerate(fns):
        executable_spec, spec_version = serialize_executable_spec(fn)

        if FLAGS.canonicalize_function_name:
          name = _canonicalize_function_name(name)
        arg_types = argument_types(fn)
        print(
            f"""
absl::StatusOr<{ name }> { name }::Load(aux::Client client) {{
  auto executable = internal::AustralisComputation::LoadHlo(
      client,
      hlo_text_{executable_id},
      executable_spec_{executable_id}, {spec_version});

  if (!executable.ok()) return executable.status();
  return {name}(*std::move(executable));
}}

""",
            file=impl_file)

        print(
            f"""

class { name } : public aux::TypedComputation<{arg_types}> {{
 public:
  // Compiles the Australis-staged program and returns a callable functor.
  static absl::StatusOr<{ name }> Load(aux::Client client);

 private:
  using TypedComputation::TypedComputation;
}};
""",
            file=header_file)

        embed_c_string(
            executable_spec,
            sym='executable_spec_%d' % executable_id,
            file=embed_impl_file)
        embed_c_string(
            bytes(fn._lowering.hlo().as_serialized_hlo_module_proto()),
            sym='hlo_text_%d' % executable_id,
            file=embed_impl_file)

      print(
          f"""\
}}  // namespace {FLAGS.cc_namespace}

#endif  // { to_include_guard(FLAGS.header_name) }
""",
          file=header_file)
      print(f"""\
}}  // namespace {FLAGS.cc_namespace}
""", file=impl_file)

  return app.run(main)
