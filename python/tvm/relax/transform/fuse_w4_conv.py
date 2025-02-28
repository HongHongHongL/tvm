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
from tvm.relax.dpl.pattern import is_op, is_call_tir, wildcard, is_tuple
from tvm.relax import Call, Expr, PyExprMutator, expr_functor
from tvm.relax.op.base import call_tir
from tvm.script import tir as T

def sch_w4_conv(sch, conv_type="w4_conv_oc", device="Android"):
    conv_block = sch.get_block(conv_type)
    N, OH, OC, FH, IC = [iter_var.dom.extent for iter_var in sch.get(conv_block).iter_vars]
    m3, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
    if device == "Android":
        if OC >= 16 and OC % 16 == 0:
            n1 = 16
        elif OC >= 8 and OC % 8 == 0:
            n1 = 8
        elif OC >= 4 and OC % 4 == 0:
            n1 = 4
        if FH * IC >= 16 and FH * IC % 16 == 0:
            k1 = 16
        elif FH * IC >= 4 and FH * IC % 4 == 0:
            k1 = 4
        elif FH * IC < 4:
            k1 = FH * IC
    else:
        if OC >= 32 and OC % 32 == 0:
            n1 = 32
        elif OC >= 16 and OC % 16 == 0:
            n1 = 16
        elif OC >= 8 and OC % 8 == 0:
            n1 = 8
        elif OC >= 4 and OC % 4 == 0:
            n1 = 4
        if FH * IC >= 4 and FH * IC % 4 == 0:
            k1 = 4
        elif FH * IC < 4:
            k1 = FH * IC
        # if (OC // n1) % 2 == 0:
        #     n3 = OC // n1 // 2
        # else:
        #     n3 = OC // n1
    reindex_block = sch.reindex(conv_block, ("read", 0))
    sch.pad_einsum(conv_block, [1, 1, 1, 1, 1])
    sch.compute_inline(reindex_block)
    pad_A = sch.get_block("pad_A")
    sch.compute_inline(pad_A)
    unpack_block = sch.get_block("unpack")
    sch.transform_layout(unpack_block, ("write", 0), lambda vfw, vic, voc: (
        voc // (n3 * n2 * n1), (voc // (n2 * n1)) % n3, (vfw * IC + vic) // k1, (voc // (n1)) % n2, (vfw * IC + vic) % k1, voc % n1))
    
    vbatch, voh, voc, vfh, vic = sch.get_loops(conv_block)
    vm = sch.fuse(vbatch, voh)
    vn = voc
    vk = sch.fuse(vfh, vic)
    
    sch.split(vm, factors=[m3, None, m2, m1])
    sch.split(vn, factors=[None, n3, n2, n1])
    sch.split(vk, factors=[None, k1])
    vm4, vm3, vm2, vm1, vn4, vn3, vn2, vn1, vk2, vk1 = sch.get_loops(conv_block)
    sch.reorder(vm4, vn4, vm3, vn3, vk2, vm2, vn2, vk1, vm1, vn1)
    sch.vectorize(vn1)
    
    unpack_block = sch.get_block("unpack")
    vufh, vuic, vuoc = sch.get_loops(unpack_block)
    vuk = sch.fuse(vufh, vuic)
    vun = vuoc
    sch.split(vuk, factors=[None, k1])
    sch.split(vun, factors=[None, n3, n2, (n1 + 1) // 2])
    vuk2, vuk1, vun4, vun3, vun2, vun1 = sch.get_loops(unpack_block)
    sch.reorder(vun4, vun3, vuk2, vun2, vuk1, vun1)
    sch.parallel(vun4)
    # sch.reverse_compute_at(block=unpack_block, loop=vn4, preserve_unit_loops=False)
    
    cache_block = sch.cache_write(block=conv_block, write_buffer_index=0, storage_scope="local")
    sch.reverse_compute_at(block=cache_block, loop=vn3, preserve_unit_loops=True)
    vcm, vcn= sch.get_loops(cache_block)[-2:]
    sch.split(vcm, factors=[None, m1])
    sch.split(vcn, factors=[None, n1])
    
    vcm2, vcm1, vcn2, vcn1 = sch.get_loops(cache_block)[-4:]
    sch.vectorize(vcn1)

    vparallel = sch.fuse(vm4, vn4)
    sch.parallel(vparallel)
    sch.annotate(vparallel, ann_key="pragma_auto_unroll_max_step", ann_val=128)
    sch.annotate(vparallel, ann_key="pragma_unroll_explicit", ann_val=1)
    
    try:
        bias = sch.get_block("bias")
        sch.reverse_compute_inline(bias)
    except:
        pass

    sch.decompose_reduction(conv_block, vk2)
    return sch


@expr_functor.mutator
class W4ConvCodeGenerator(PyExprMutator):
    """
    Converts the expensive non linear functions to their fast but approximate counterparts.

    Parameters
    ----------
    mod: IRModule
        The module to be transformed
    """
    
    def __init__(self, mod, device):
        super().__init__(mod)
        self.id = 0
        self.device = device

    def visit_call_(self, call: Call) -> Expr:
        from .w4_conv_func import create_w4_conv_oc, create_w4_group_conv_oc, create_w4_conv_ic # 局部导入TIR函数
        if isinstance(call.op, relax.GlobalVar):
            function = self.builder_.get()[call.op]
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "w4_conv"
            ):
                dtype = function.body.blocks[0].bindings[-1].value.args[0].struct_info.dtype

                groups = function.body.blocks[0].bindings[-1].value.attrs.groups
                pad = function.body.blocks[0].bindings[-1].value.attrs.padding[0]
                data_layout = function.body.blocks[0].bindings[-1].value.attrs.data_layout
                kernel_layout = function.body.blocks[0].bindings[-1].value.attrs.kernel_layout
                strides = function.body.blocks[0].bindings[-1].value.attrs.strides[0]

                if data_layout == "NWC":
                    N = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[1]
                    IC = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[2]
                else:
                    N = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[2]
                    IC = function.body.blocks[0].bindings[-1].value.args[0].struct_info.shape[1]
                if kernel_layout == "WIO":
                    OC = function.body.blocks[0].bindings[-1].value.args[1].struct_info.shape[2]
                    KW = function.body.blocks[0].bindings[-1].value.args[1].struct_info.shape[0]
                else:
                    OC = function.body.blocks[0].bindings[-1].value.args[1].struct_info.shape[0]
                    KW = function.body.blocks[0].bindings[-1].value.args[1].struct_info.shape[2]
                
                N = N if isinstance(N,tvm.tir.IntImm) else None
                IW = IW if isinstance(IW,tvm.tir.IntImm) else None
                IC = IC if isinstance(IC,tvm.tir.IntImm) else None
                OC = OC if isinstance(OC,tvm.tir.IntImm) else None
                KW = KW if isinstance(KW,tvm.tir.IntImm) else None

                if OC != 1:
                    if groups == 1:
                        sch = tvm.tir.Schedule(create_w4_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype))
                        sch = sch_w4_conv(sch, "w4_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                    else:
                        sch = tvm.tir.Schedule(create_w4_group_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype))
                        sch = sch_w4_conv(sch, "w4_group_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                else:
                    sch = tvm.tir.Schedule(create_w4_conv_ic(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype))
                    sch = sch_w4_conv(sch, "w4_conv_ic", device=self.device)
                    gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")

                info = (call.args[2],call.args[0],call.args[1])
                self.id += 1
                return call_tir(
                    gvar,
                    info,
                    call.struct_info
                )
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "w4_conv_with_bias"
            ):
                dtype = function.body.blocks[0].bindings[-2].value.args[0].struct_info.dtype

                groups = function.body.blocks[0].bindings[-2].value.attrs.groups
                pad = function.body.blocks[0].bindings[-2].value.attrs.padding[0]
                data_layout = function.body.blocks[0].bindings[-2].value.attrs.data_layout
                kernel_layout = function.body.blocks[0].bindings[-2].value.attrs.kernel_layout
                strides = function.body.blocks[0].bindings[-2].value.attrs.strides[0]

                if data_layout == "NWC":
                    N = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[1]
                    IC = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[2]
                else:
                    N = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[2]
                    IC = function.body.blocks[0].bindings[-2].value.args[0].struct_info.shape[1]
                if kernel_layout == "WIO":
                    OC = function.body.blocks[0].bindings[-2].value.args[1].struct_info.shape[2]
                    KW = function.body.blocks[0].bindings[-2].value.args[1].struct_info.shape[0]
                else:
                    OC = function.body.blocks[0].bindings[-2].value.args[1].struct_info.shape[0]
                    KW = function.body.blocks[0].bindings[-2].value.args[1].struct_info.shape[2]
                
                N = N if isinstance(N,tvm.tir.IntImm) else None
                IW = IW if isinstance(IW,tvm.tir.IntImm) else None
                IC = IC if isinstance(IC,tvm.tir.IntImm) else None
                OC = OC if isinstance(OC,tvm.tir.IntImm) else None
                KW = KW if isinstance(KW,tvm.tir.IntImm) else None

                if OC != 1:
                    if groups == 1:
                        sch = tvm.tir.Schedule(create_w4_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias=True))
                        sch = sch_w4_conv(sch, "w4_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                    else:
                        sch = tvm.tir.Schedule(create_w4_group_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias=True))
                        sch = sch_w4_conv(sch, "w4_group_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                else:
                    sch = tvm.tir.Schedule(create_w4_conv_ic(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype))
                    sch = sch_w4_conv(sch, "w4_conv_ic", device=self.device)
                    gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")

                info = (call.args[2],call.args[0],call.args[1], call.args[3])
                self.id += 1
                return call_tir(
                    gvar,
                    info,
                    call.struct_info
                )
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "w4_conv_with_bias_with_relu"
            ):
                dtype = function.body.blocks[0].bindings[-3].value.args[0].struct_info.dtype

                groups = function.body.blocks[0].bindings[-3].value.attrs.groups
                pad = function.body.blocks[0].bindings[-3].value.attrs.padding[0]
                data_layout = function.body.blocks[0].bindings[-3].value.attrs.data_layout
                kernel_layout = function.body.blocks[0].bindings[-3].value.attrs.kernel_layout
                strides = function.body.blocks[0].bindings[-3].value.attrs.strides[0]

                if data_layout == "NWC":
                    N = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[1]
                    IC = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[2]
                else:
                    N = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[2]
                    IC = function.body.blocks[0].bindings[-3].value.args[0].struct_info.shape[1]
                if kernel_layout == "WIO":
                    OC = function.body.blocks[0].bindings[-3].value.args[1].struct_info.shape[2]
                    KW = function.body.blocks[0].bindings[-3].value.args[1].struct_info.shape[0]
                else:
                    OC = function.body.blocks[0].bindings[-3].value.args[1].struct_info.shape[0]
                    KW = function.body.blocks[0].bindings[-3].value.args[1].struct_info.shape[2]
                
                N = N if isinstance(N,tvm.tir.IntImm) else None
                IW = IW if isinstance(IW,tvm.tir.IntImm) else None
                IC = IC if isinstance(IC,tvm.tir.IntImm) else None
                OC = OC if isinstance(OC,tvm.tir.IntImm) else None
                KW = KW if isinstance(KW,tvm.tir.IntImm) else None

                if OC != 1:
                    if groups == 1:
                        sch = tvm.tir.Schedule(create_w4_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu=True))
                        sch = sch_w4_conv(sch, "w4_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                    else:
                        sch = tvm.tir.Schedule(create_w4_group_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu=True))
                        sch = sch_w4_conv(sch, "w4_group_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                else:
                    sch = tvm.tir.Schedule(create_w4_conv_ic(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype))
                    sch = sch_w4_conv(sch, "w4_conv_ic", device=self.device)
                    gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")

                info = (call.args[2],call.args[0],call.args[1], call.args[3])
                self.id += 1
                return call_tir(
                    gvar,
                    info,
                    call.struct_info
                )
            if (
                "Composite" in function.attrs
                and function.attrs["Composite"] == "w4_conv_with_bias_with_relu_with_transpose"
            ):
                dtype = function.body.blocks[0].bindings[-4].value.args[0].struct_info.dtype

                groups = function.body.blocks[0].bindings[-4].value.attrs.groups
                pad = function.body.blocks[0].bindings[-4].value.attrs.padding[0]
                data_layout = function.body.blocks[0].bindings[-4].value.attrs.data_layout
                kernel_layout = function.body.blocks[0].bindings[-4].value.attrs.kernel_layout
                strides = function.body.blocks[0].bindings[-4].value.attrs.strides[0]

                if data_layout == "NWC":
                    N = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[1]
                    IC = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[2]
                else:
                    N = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[0]
                    IW = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[2]
                    IC = function.body.blocks[0].bindings[-4].value.args[0].struct_info.shape[1]
                if kernel_layout == "WIO":
                    OC = function.body.blocks[0].bindings[-4].value.args[1].struct_info.shape[2]
                    KW = function.body.blocks[0].bindings[-4].value.args[1].struct_info.shape[0]
                else:
                    OC = function.body.blocks[0].bindings[-4].value.args[1].struct_info.shape[0]
                    KW = function.body.blocks[0].bindings[-4].value.args[1].struct_info.shape[2]
                
                N = N if isinstance(N,tvm.tir.IntImm) else None
                IW = IW if isinstance(IW,tvm.tir.IntImm) else None
                IC = IC if isinstance(IC,tvm.tir.IntImm) else None
                OC = OC if isinstance(OC,tvm.tir.IntImm) else None
                KW = KW if isinstance(KW,tvm.tir.IntImm) else None
                
                if function.body.blocks[0].bindings[-1].value.attrs.axes[1] == 1:
                    if groups == 1:
                        sch = tvm.tir.Schedule(create_w4_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu=True))
                        sch = sch_w4_conv(sch, "w4_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                    else:
                        sch = tvm.tir.Schedule(create_w4_group_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu=True))
                        sch = sch_w4_conv(sch, "w4_group_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                else:
                    if groups == 1:
                        sch = tvm.tir.Schedule(create_w4_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu_transpose=True))
                        sch = sch_w4_conv(sch, "w4_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")
                    else:
                        sch = tvm.tir.Schedule(create_w4_group_conv_oc(N_val=N, IC_val=IC, OC_val=OC, IW_val=IW, FW_val=KW, pad_val=pad, strides=strides, dtype=dtype,bias_relu_transpose=True))
                        sch = sch_w4_conv(sch, "w4_group_conv_oc", device=self.device)
                        gvar = self.builder_.add_func(sch.mod["main"].with_attr("global_symbol",f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}").with_attr("tir.is_scheduled", True), f"w4_group_conv_{OC.value}_{IC.value}_{KW.value}_{self.id}")

                info = (call.args[2],call.args[0],call.args[1], call.args[3])
                self.id += 1
                return call_tir(
                    gvar,
                    info,
                    call.struct_info
                )

        return super().visit_call_(call)


def _pattern():
    """Pattern for layer norm op."""
    repeat_1 = is_op("relax.repeat")(wildcard())
    astype_1 = is_op("relax.astype")(repeat_1)
    # div_1 = is_op("relax.divide")(wildcard(),wildcard())
    mul_2 = is_op("relax.multiply")(astype_1,wildcard())
    # permute_dims_1 = is_op("relax.permute_dims")(div_1)
    conv1d = is_op("relax.nn.conv1d")(wildcard(),mul_2)
    return conv1d

def _pattern_with_bias():
    """Pattern for layer norm op."""
    repeat_1 = is_op("relax.repeat")(wildcard())
    astype_1 = is_op("relax.astype")(repeat_1)
    # div_1 = is_op("relax.divide")(wildcard(),wildcard())
    mul_2 = is_op("relax.multiply")(astype_1,wildcard())
    # permute_dims_1 = is_op("relax.permute_dims")(div_1)
    conv1d = is_op("relax.nn.conv1d")(wildcard(),mul_2)
    add_1 = is_op("relax.add")(conv1d,wildcard())
    return add_1

def _pattern_with_bias_with_relu():
    """Pattern for layer norm op."""
    repeat_1 = is_op("relax.repeat")(wildcard())
    astype_1 = is_op("relax.astype")(repeat_1)
    # div_1 = is_op("relax.divide")(wildcard(),wildcard())
    mul_2 = is_op("relax.multiply")(astype_1,wildcard())
    # permute_dims_1 = is_op("relax.permute_dims")(div_1)
    conv1d = is_op("relax.nn.conv1d")(wildcard(),mul_2)
    add_1 = is_op("relax.add")(conv1d,wildcard())
    relu_1 = is_op("relax.nn.relu")(add_1)
    return relu_1

def _pattern_with_bias_with_relu_with_transpose():
    """Pattern for layer norm op."""
    repeat_1 = is_op("relax.repeat")(wildcard())
    astype_1 = is_op("relax.astype")(repeat_1)
    # div_1 = is_op("relax.divide")(wildcard(),wildcard())
    mul_2 = is_op("relax.multiply")(astype_1,wildcard())
    # permute_dims_1 = is_op("relax.permute_dims")(div_1)
    conv1d = is_op("relax.nn.conv1d")(wildcard(),mul_2)
    add_1 = is_op("relax.add")(conv1d,wildcard())
    relu_1 = is_op("relax.nn.relu")(add_1)
    permute_dims_1 = is_op("relax.permute_dims")(relu_1)
    return permute_dims_1


@tvm.transform.module_pass(opt_level=0, name="FuseW4ConvTransform")
class FuseW4ConvTransform:
    """
    Pass to fuse w4 conv op.
    """
    def __init__(self, device="Android"):
        self.device = device

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        """IRModule-level transformation"""
        # import pdb
        # pdb.set_trace()
        mod = relax.transform.FuseOpsByPattern([("w4_conv_with_bias_with_relu_with_transpose", _pattern_with_bias_with_relu_with_transpose()), ("w4_conv_with_bias_with_relu", _pattern_with_bias_with_relu()), ("w4_conv_with_bias", _pattern_with_bias()), ("w4_conv", _pattern())])(mod)
        w4_conv_codegen = W4ConvCodeGenerator(mod, self.device)
        for gv, func in mod.functions_items():
            if isinstance(func, tvm.relax.Function):
                func = w4_conv_codegen.visit_expr(func)
                w4_conv_codegen.builder_.update_func(gv, func)
        return w4_conv_codegen.builder_.get()