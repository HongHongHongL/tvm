# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument, redefined-argument-from-local
"""Fuse layer_norm pass."""
import tvm
from tvm import topi, relax
from tvm.ir.module import IRModule
from tvm.relax.dpl.pattern import is_op, wildcard
from tvm.relax import Call, Expr, PyExprMutator, expr_functor


@expr_functor.mutator
class LayerNormCodeGenerator(PyExprMutator):
    """
    Converts the expensive non linear functions to their fast but approximate counterparts.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """

    def __init__(self, mod):
        super().__init__(mod)

    def visit_function_(self, op):
        return super().visit_function_(op)

    def visit_call_(self, call: Call) -> Expr:
        if isinstance(call.op, relax.GlobalVar):
            function = self.builder_.get()[call.op]
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "layer_norm"
            ):
                if not isinstance(call.struct_info, tvm.relax.struct_info.TupleStructInfo):
                    axis = function.body.blocks[0].bindings[0].value.attrs.get_int('axis')
                    epsilon = function.body.blocks[0].bindings[4].value.args[1].data.numpy()[()]
                    return self.builder_.call_te(
                        topi.nn.layer_norm,
                        call.args[0],
                        call.args[-2],
                        call.args[-1],
                        axis,
                        float(epsilon)
                    )

        return super().visit_call_(call)


def _pattern():
    """Pattern for layer norm op."""
    in0 = wildcard()
    rm1 = is_op("relax.mean")(in0)
    sub = is_op("relax.subtract")(in0, rm1)
    power = is_op("relax.multiply")(sub, sub)
    rm2 = is_op("relax.mean")(power)
    add1 = is_op("relax.add")(rm2, wildcard())
    sqrt = is_op("relax.sqrt")(add1)
    div = is_op("relax.divide")(sub, sqrt)
    mul = is_op("relax.multiply")(div, wildcard())
    add2 = is_op("relax.add")(mul, wildcard())
    return add2


@tvm.transform.module_pass(opt_level=0, name="FuseLayerNormTransform")
class FuseLayerNormTransform:
    """
    Pass to fuse layer norm op.
    """

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        mod = relax.transform.FuseOpsByPattern([("layer_norm", _pattern())])(mod)
        layer_norm_codegen = LayerNormCodeGenerator(mod)
        for gv, func in mod.functions_items():
            if isinstance(func, tvm.relax.Function):
                func = layer_norm_codegen.visit_expr(func)
                layer_norm_codegen.builder_.update_func(gv, func)
        return layer_norm_codegen.builder_.get()
