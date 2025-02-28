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
# pylint: disable=missing-docstring, invalid-name
"""A GEMM schedule rule for CPU operators."""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import uuid

from tvm import tir
from tvm.ir import Range
from tvm.target import Target
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.analysis import undefined_vars
from tvm.tir.schedule.schedule import BlockRV
from tvm.script import tir as T
from tvm.contrib import utils, clang
from ..base import analysis, BlockInfo, IterInfo
from .base import CPUScheduleRule


def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    auto_inline_producers(sch, p)
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0)) for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes]
    B_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes]
    C_traits = [traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(block: tir.Block) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    A_index_map = make_iter_fusion_index_map(
        A_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_K]
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, [IterKind.kIter_S, IterKind.kIter_J, IterKind.kIter_K]
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J]
    )
    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def get_block_info(sch: tir.Schedule, block: tir.schedule.BlockRV) -> BlockInfo:
    def _iter_kind(loop: tir.IterVar) -> str:
        return {tir.IterVar.DataPar: "S", tir.IterVar.CommReduce: "R"}.get(loop.iter_type, "O")

    def _is_reduction_block(block: tir.schedule.BlockRV):
        for iter_var in sch.get(block).iter_vars:
            if _iter_kind(iter_var) == "R":
                return True
        return False

    return BlockInfo(
        name=sch.get(block).name_hint,
        iters=[
            IterInfo(
                kind=_iter_kind(iter_var),
                var=iter_var.var,
                dom=iter_var.dom.extent,
                loop_rv=loop_rv,
            )
            for loop_rv, iter_var in zip(sch.get_loops(block), sch.get(block).iter_vars)
        ],
        block_rv=block,
        reduction_block=_is_reduction_block(block),
    )


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None

    return reduction_blocks


def get_in_out_dtypes(block: tir.Block) -> Tuple[str]:
    """
    Detect In/Out data types for the given block based on the analysis if read/write buffers.
    """
    assert len(block.reads) > 0 and len(block.writes) > 0
    in_dtype = block.reads[0].buffer.dtype
    out_dtype = block.writes[0].buffer.dtype
    return (in_dtype, out_dtype)


def _c_to_llvm(c_code: str) -> str:
    unique_filename = str(uuid.uuid4())
    temp = utils.tempdir()
    ll_path = temp.relpath(f"{unique_filename}.ll")
    ll_code = clang.create_llvm([c_code], output=ll_path)
    return ll_code


def add_llvm_to_block(
    sch: tir.Schedule, loop: tir.schedule.LoopRV, c_code_str: str = ""
) -> tir.Schedule:
    sch.annotate(loop, "pragma_import_llvm", _c_to_llvm(c_code_str))
    return sch


class Matmul(CPUScheduleRule):
    """The schedule rule for matmul-like computation"""

    @dataclass
    class Config:
        m1: int = 1
        m2: int = 1
        m3: int = 1
        n1: int = 1
        n2: int = 1
        n3: int = 1
        k1: int = 1

    def get_configs(self, target: Target) -> Config:
        """Get the schedule config for the target"""
        return Matmul.Config()

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> Optional[tir.Schedule]:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None
        sch = tir.Schedule(func)
        # config = self.get_configs(target)
        root_block = analysis.get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)

        reduction_blocks = get_reduction_blocks(sch, blocks)
        if reduction_blocks is None:
            return None
        
        main_block = reduction_blocks[0]
        block_stmt = sch.get(main_block)
        if len(block_stmt.reads) != 2 or len(block_stmt.writes) != 1:
            return None
        # print(sch.mod)

        # index_maps = get_index_map(block_stmt)
        # if index_maps is None:
        #     return None
        # print("######", index_maps)

        m4, m3, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1, 1
        pack_block = None
        cnt_reduce = 0
        for iter_var in sch.get(main_block).iter_vars:
            if iter_var.iter_type == 2:
                cnt_reduce += 1
        if cnt_reduce > 1:
            # conv
            if len(sch.get(main_block).iter_vars) == 7:
                # conv2d
                pad_temp_block = sch.get_block("pad_temp")
                sch.compute_inline(pad_temp_block)
                N, OC, OH, OW, IC, FH, FW = [iter_var.dom.extent for iter_var in sch.get(main_block).iter_vars]
                if IC == 1 and FH == 1 and FW == 1:
                    vbatch, voc, voh, vow, vic, vfh, vfw = sch.get_loops(main_block)
                    n3, n1 = 4, 4
                    vm = sch.fuse(vbatch, voc, voh)
                    vn = vow
                    vk = sch.fuse(vic, vfh, vfw)
                    sch.split(vn, factors=[n3, None, n1])
                    vm, vn3, vn2, vn1, vk = sch.get_loops(main_block)
                    sch.reorder(vn3, vm, vn2, vk, vn1)
                    sch.parallel(vn3)
                    sch.vectorize(vn1)
                    sch.decompose_reduction(main_block, vk)
                    # print(sch.mod)
                    return sch
                m4, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
                if OC >= 16 and OC % 16 == 0:
                    n1 = 16
                elif OC >= 8 and OC % 8 == 0:
                    n1 = 8
                elif OC >= 4 and OC % 4 == 0:
                    n1 = 4
                if FH * FW * IC >= 4 and FH * FW * IC % 4 == 0:
                    k1 = 4
                elif FH * FW * IC < 4:
                    k1 = FH * FW * IC
            elif len(sch.get(main_block).iter_vars) == 5:
                # conv1d
                N, OH, OC, FH, IC = [iter_var.dom.extent for iter_var in sch.get(main_block).iter_vars]
                
                OH_pad = OH
                # OH_pad = ((OH + 3) // 4) * 4
                # OH_pad = ((OH + 15) // 16) * 16
                m3, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
                m_r = N * OH_pad
                if (not isinstance(N, tir.expr.Var)) and (not isinstance(OH, tir.expr.Var)):
                    # if m_r % 6 == 0:
                    #     m1 = 6
                    # elif m_r % 5 == 0:
                    #     m1 = 5
                    # el
                    if m_r % 4 == 0:
                        m1 = 4
                    elif m_r % 3 == 0:
                        m1 = 3
                    elif m_r % 2 == 0:
                        m1 = 2
                    else:
                        m1 = 1
                    m_r = m_r // m1
                    if m_r >= 16 and m_r % 16 == 0:
                        m4 = 16
                    elif m_r >= 8 and m_r % 8 == 0:
                        m4 = 8
                    elif m_r >= 4 and m_r % 4 == 0:
                        m4 = 4
                    else:
                        m4 = m_r
                    m3 = (N * OH_pad) // (m4 * m2 * m1)
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
                # if OC // n1 % 16 == 0:
                #     n3 = 16
                # if m_r // m1 % 40 == 0:
                #     m3 = 40

                reindex_block = sch.reindex(main_block, ("read", 0))
                sch.pad_einsum(main_block, [1, 1, 1, 1, 1])
                try:
                    pad_temp_block = sch.get_block("pad_temp")
                    sch.compute_inline(pad_temp_block)
                except:
                    pass
                sch.compute_inline(reindex_block)
                if OH_pad != OH:
                    pad_block = sch.get_block("pad_temp_reindex_pad")
                    sch.compute_inline(pad_block)
                # else:
                #     try:
                #         sch.get_block("group_conv1d_nwc")
                #         # pack_block = sch.reindex_cache_read(main_block, 0, "local", lambda vn, voh, voc, vfh, vic: (
                #         #     (vn * OH_pad + voh) // (m3 * m2 * m1), ((vn * OH_pad + voh) // (m2 * m1)) % m3, (vfh * IC + vic + voc) // k1, ((vn * OH_pad + voh) // m1) % m2, (vfh * IC + vic + voc) % k1, (vn * OH_pad + voh) % m1))
                #     except:
                #         if FH == 1:
                #             pack_block = sch.reindex_cache_read(main_block, 0, "local", lambda vn, voh, voc, vfh, vic: (
                #                 (vn * OH_pad + voh) // (m3 * m2 * m1), ((vn * OH_pad + voh) // (m2 * m1)) % m3, vic // k1, ((vn * OH_pad + voh) // m1) % m2, vic % k1, (vn * OH_pad + voh) % m1))
                #         else:
                #             pack_block = sch.reindex_cache_read(main_block, 0, "local", lambda vn, voh, voc, vfh, vic: (
                #                 (vn * OH_pad + voh) // (m3 * m2 * m1), ((vn * OH_pad + voh) // (m2 * m1)) % m3, (vfh * IC + vic) // k1, ((vn * OH_pad + voh) // m1) % m2, (vfh * IC + vic) % k1, (vn * OH_pad + voh) % m1))

                pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vn, voh, voc, vfh, vic: (
                    voc // (n3 * n2 * n1), (voc // (n2 * n1)) % n3, (vfh * IC + vic) // k1, (voc // (n1)) % n2, (vfh * IC + vic) % k1, voc % n1))
                sch.annotate(pack_b_block, "meta_schedule.layout_rewrite_preproc", T.bool(True))
                vpboc, vpbfh, vpbic = sch.get_loops(pack_b_block)
                vpbk = sch.fuse(vpbfh, vpbic)
                sch.split(vpboc, factors=[None, n2, n1])
                sch.split(vpbk, factors=[None, k1])
                vpboc3, vpboc2, vpboc1, vpbk2, vpbk1 = sch.get_loops(pack_b_block)
                sch.reorder(vpboc3, vpbk2, vpboc2, vpbk1, vpboc1)
                sch.vectorize(vpboc1)

                vbatch, voh, voc, vfh, vic = sch.get_loops(main_block)
                vm = sch.fuse(vbatch, voh)
                vn = voc
                vk = sch.fuse(vfh, vic)
        else:
            # gemm
            # print([iter_var for iter_var in sch.get(main_block).iter_vars])
            if len(sch.get(main_block).iter_vars) == 5:
                # batch gemm
                B1, B2, M, N, K = [iter_var.dom.extent for iter_var in sch.get(main_block).iter_vars]
                m4, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
                if N >= 16 and N % 16 == 0:
                    n1 = 16
                elif N >= 8 and N % 8 == 0:
                    n1 = 8
                elif N >= 4 and N % 4 == 0:
                    n1 = 4
                m_r = B1 * B2 * M
                if (not isinstance(B1, tir.expr.Var)) and (not isinstance(B2, tir.expr.Var)) and (not isinstance(M, tir.expr.Var)):
                    if m_r % 4 == 0:
                        m1 = 4
                    elif m_r % 3 == 0:
                        m1 = 3
                    elif m_r % 2 == 0:
                        m1 = 2
                    else:
                        m1 = 1
                    m_r = m_r // m1
                    if m_r >= 4 and m_r % 4 == 0:
                        m4 = 4
                    # else:
                    #     m4 = B * M
                    m3 = B1 * B2 * M // (m4 * m2 * m1)
                if K >= 4 and K % 4 == 0:
                    k1 = 4
                elif K < 4:
                    k1 = K
                
                pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vbatch1, vbatch2, vm, vn, vk: (
                    vbatch1, vbatch2, vn // (n3 * n2 * n1), (vn // (n2 * n1)) % n3, vk // k1, (vn // (n1)) % n2, vk % k1, vn % n1))
                sch.annotate(pack_b_block, "meta_schedule.layout_rewrite_preproc", T.bool(True))
                # vpbn, vpbk = sch.get_loops(pack_b_block)
                # sch.split(vpbn, factors=[None, n2, n1])
                # sch.split(vpbk, factors=[None, k1])
                # vpbn3, vpbn2, vpbn1, vpbk2, vpbk1 = sch.get_loops(pack_b_block)
                # sch.reorder(vpbn3, vpbk2, vpbn2, vpbk1, vpbn1)
                # sch.vectorize(vpbn1)

                vbatch1, vbatch2, vm, vn, vk = sch.get_loops(main_block)
                vm = sch.fuse(vbatch1, vbatch2, vm)
            elif len(sch.get(main_block).iter_vars) == 4:
                # batch gemm
                B, M, N, K = [iter_var.dom.extent for iter_var in sch.get(main_block).iter_vars]
                m4, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
                if N >= 16 and N % 16 == 0:
                    n1 = 16
                elif N >= 8 and N % 8 == 0:
                    n1 = 8
                elif N >= 4 and N % 4 == 0:
                    n1 = 4
                elif N >= 2 and N % 2 == 0:
                    n1 = 2
                m_r = B * M
                if (not isinstance(B, tir.expr.Var)) and (not isinstance(M, tir.expr.Var)):
                    if m_r % 4 == 0:
                        m1 = 4
                    elif m_r % 3 == 0:
                        m1 = 3
                    elif m_r % 2 == 0:
                        m1 = 2
                    else:
                        m1 = 1
                    m_r = m_r // m1
                    if m_r >= 4 and m_r % 4 == 0:
                        m4 = 4
                    # else:
                    #     m4 = B * M
                    m3 = B * M // (m4 * m2 * m1)
                if K >= 4 and K % 4 == 0:
                    k1 = 4
                elif K < 4:
                    k1 = K
                
                try:
                    pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vbatch, vm, vn, vk: (
                        vbatch, vn // (n3 * n2 * n1), (vn // (n2 * n1)) % n3, vk // k1, (vn // (n1)) % n2, vk % k1, vn % n1))
                except:
                    pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vbatch, vm, vn, vk: (
                        vn // (n3 * n2 * n1), (vn // (n2 * n1)) % n3, vk // k1, (vn // (n1)) % n2, vk % k1, vn % n1))
                sch.annotate(pack_b_block, "meta_schedule.layout_rewrite_preproc", T.bool(True))
                vpbn, vpbk = sch.get_loops(pack_b_block)[-2:]
                sch.split(vpbn, factors=[None, n2, n1])
                sch.split(vpbk, factors=[None, k1])
                vpbn3, vpbn2, vpbn1, vpbk2, vpbk1 = sch.get_loops(pack_b_block)[-5:]
                sch.reorder(vpbn3, vpbk2, vpbn2, vpbk1, vpbn1)
                sch.vectorize(vpbn1)

                vbatch, vm, vn, vk = sch.get_loops(main_block)
                vm = sch.fuse(vbatch, vm)
            else:
                # gemm
                M, N, K = [iter_var.dom.extent for iter_var in sch.get(main_block).iter_vars]
                m4, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 8, 1
                if not isinstance(M, tir.expr.Var):
                    if M >= 4 and M % 4 == 0:
                        m4 = 4
                    else:
                        m4 = M
                    m_r = M // m4
                    if m_r % 4 == 0:
                        m1 = 4
                    elif m_r % 3 == 0:
                        m1 = 3
                    elif m_r % 2 == 0:
                        m1 = 2
                    else:
                        m1 = 1
                    m3 = M // (m4 * m2 * m1)
                if N >= 16 and N % 16 == 0:
                    n1 = 16
                elif N >= 8 and N % 8 == 0:
                    n1 = 8
                elif N >= 4 and N % 4 == 0:
                    n1 = 4
                if K >= 4 and K % 4 == 0:
                    k1 = 4
                elif K < 4:
                    k1 = K
                
                pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vm, vn, vk: (
                    vn // (n3 * n2 * n1), (vn // (n2 * n1)) % n3, vk // k1, (vn // (n1)) % n2, vk % k1, vn % n1))
                sch.annotate(pack_b_block, "meta_schedule.layout_rewrite_preproc", T.bool(True))
                vpbn, vpbk = sch.get_loops(pack_b_block)
                sch.split(vpbn, factors=[None, n2, n1])
                sch.split(vpbk, factors=[None, k1])
                vpbn3, vpbn2, vpbn1, vpbk2, vpbk1 = sch.get_loops(pack_b_block)
                sch.reorder(vpbn3, vpbk2, vpbn2, vpbk1, vpbn1)
                sch.vectorize(vpbn1)
                
                # pack_b_block = sch.reindex_cache_read(main_block, 1, "local", lambda vm, vn, vk: (
                #     vk // k1, vn // n1, vk % k1, vn % n1))
                # sch.annotate(pack_b_block, "meta_schedule.layout_rewrite_preproc", T.bool(True))
                # vpbn, vpbk = sch.get_loops(pack_b_block)
                # sch.split(vpbn, factors=[None, n1])
                # sch.split(vpbk, factors=[None, k1])
                # vpbn2, vpbn1, vpbk2, vpbk1 = sch.get_loops(pack_b_block)
                # sch.reorder(vpbk2, vpbn2, vpbk1, vpbn1)
                # sch.vectorize(vpbn1)

                vm, vn, vk = sch.get_loops(main_block)

        # print(sch.mod)
        # m4, m2, m1, n3, n2, n1, k1 = 1, 1, 1, 1, 1, 1, 1
        # sch.compute_at(block=pack_block, loop=vm, preserve_unit_loops=True)
        # print(sch.mod)
        # print(m3, m2, m1)
        sch.split(vm, factors=[None, m3, m2, m1])
        sch.split(vn, factors=[None, n3, n2, n1])
        sch.split(vk, factors=[None, k1])
        vm4, vm3, vm2, vm1, vn4, vn3, vn2, vn1, vk2, vk1 = sch.get_loops(main_block)
        sch.reorder(vm4, vn4, vm3, vn3, vk2, vm2, vn2, vk1, vm1, vn1)
        sch.vectorize(vn1)

        # add_llvm_to_block(sch, vm2, xsmm_asm_armv8_code(m1, k1, n1, K, n1, N, ))

        if pack_block is not None:
            sch.compute_at(block=pack_block, loop=vm4, preserve_unit_loops=True)
            if FH == 1:
                _, vpn, vpoh, vpic = sch.get_loops(pack_block)
                vpm = sch.fuse(vpn, vpoh)
                vpk = sch.fuse(vpic)
            else:
                _, vpn, vpoh, vpfh, vpic = sch.get_loops(pack_block)
                vpm = sch.fuse(vpn, vpoh)
                vpk = sch.fuse(vpfh, vpic)
            sch.split(vpm, factors=[None, m2, m1])
            sch.split(vpk, factors=[None, k1])
            _, vpm3, vpm2, vpm1, vpk2, vpk1 = sch.get_loops(pack_block)
            sch.reorder(vpm3, vpk2, vpm2, vpk1, vpm1)

        # if OH_pad != OH:
        if False:
            cache_block = sch.get_block("B_pad")
        else:
            cache_block = sch.cache_write(block=main_block, write_buffer_index=0, storage_scope="local")
        # sch.reverse_compute_at(block=cache_block, loop=vn4, preserve_unit_loops=True)
        # if len(sch.get(main_block).iter_vars) == 3:
        #     _, _, vcm, vcn= sch.get_loops(cache_block)
        # else:
        #     _, _, _, vcm, vcn= sch.get_loops(cache_block)
        # sch.split(vcm, factors=[None, m2, m1])
        # sch.split(vcn, factors=[None, n2, n1])
        # if len(sch.get(main_block).iter_vars) == 3:
        #     _, _, vcm3, vcm2, vcm1, vcn3, vcn2, vcn1 = sch.get_loops(cache_block)
        # else:
        #     _, _, _, vcm3, vcm2, vcm1, vcn3, vcn2, vcn1 = sch.get_loops(cache_block)
        # sch.reorder(vcm3, vcn3, vcm2, vcn2, vcm1, vcn1)
        # sch.vectorize(vcn1)

        sch.reverse_compute_at(block=cache_block, loop=vn3, preserve_unit_loops=True)
        # if len(sch.get(main_block).iter_vars) == 3:
        #     _, _, _, _, vcm, vcn= sch.get_loops(cache_block)
        # else:
        #     _, _, _, _, _, vcm, vcn= sch.get_loops(cache_block)
        vcm, vcn= sch.get_loops(cache_block)[-2:]
        sch.split(vcm, factors=[None, m1])
        sch.split(vcn, factors=[None, n1])
        # if len(sch.get(main_block).iter_vars) == 3:
        #     _, _, _, _, vcm2, vcm1, vcn2, vcn1 = sch.get_loops(cache_block)
        # else:
        #     _, _, _, _, _, vcm2, vcm1, vcn2, vcn1 = sch.get_loops(cache_block)
        vcm2, vcm1, vcn2, vcn1 = sch.get_loops(cache_block)[-4:]
        # sch.reorder(vcm2, vcn2, vcm1, vcn1)
        sch.vectorize(vcn1)

        vparallel = sch.fuse(vm4, vn4)
        sch.parallel(vparallel)
        sch.annotate(vparallel, ann_key="pragma_auto_unroll_max_step", ann_val=128)
        sch.annotate(vparallel, ann_key="pragma_unroll_explicit", ann_val=1)

        auto_inline_consumer_chain(sch, cache_block)
        sch.decompose_reduction(main_block, vk2)

        # print(sch.mod)
        return sch