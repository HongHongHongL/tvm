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
# pylint: disable=missing-docstring
"""A reduction schedule rule for CPU operators."""
from typing import List, Tuple

from tvm import tir
from tvm.target import Target

from ..base import normalize_prim_func, try_inline
from . import utils
from .base import CPUScheduleRule


class Reduction(CPUScheduleRule):
    """The schedule rule for reduction computation"""

    def apply(  # pylint: disable=too-many-locals,missing-docstring
        self,
        func: tir.PrimFunc,
        target: Target,
        _: bool,
    ) -> tir.Schedule:
        if not isinstance(func, tir.PrimFunc) or not self.is_target_available(target):
            return None

        sch = tir.Schedule(func)
        block_infos = normalize_prim_func(sch)

        if block_infos is None:
            return None

        block_infos = try_inline(sch, block_infos)
        reduction_blocks: List[Tuple[tir.schedule.BlockRV, tir.schedule.LoopRV]] = []
        cnt = 0
        out_tx = None
        for block in block_infos:
            vector_size = utils.get_vector_size(target)
            core_num = utils.get_core_num(target)
            core_num = core_num * 4
            # if not "concat" in block.name:
            #     core_num = 1
            vector_size = vector_size * 4
            s_len = 1
            for iter_dom, iter_type in zip(block.dom(), block.dom_kind()):
                if iter_type == "S":
                    s_len *= iter_dom
            if block.dom_kind() != "" and block.dom_kind()[-1] == "R":
                vector_size = 1
            while s_len % vector_size != 0:
                vector_size = vector_size // 2
            s_len = s_len // vector_size
            while s_len % core_num != 0:
                core_num = core_num // 2
            
            s_loops: List[tir.schedule.LoopRV] = []
            r_loops: List[tir.schedule.LoopRV] = []
            o_loops: List[tir.schedule.LoopRV] = []
            dom_kind = block.dom_kind()
            block = block.block_rv

            if (len(sch.get_loops(block)) == 0):
                continue

            for loop, iter_type in zip(sch.get_loops(block), dom_kind):
                {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

            if not s_loops:
                s_loops.append(sch.add_unit_loop(block))

            if len(s_loops) > 1:
                _, vx = sch.split(s_loops[-1], factors=[None, vector_size])
                tx, ux = sch.split(sch.fuse(*s_loops[:-1]), factors=[core_num, None])
            else:
                tx, ux, vx = sch.split(s_loops[0], factors=[core_num, None, vector_size])
            # sch.unroll(ux)
            sch.vectorize(vx)
            sch.parallel(tx)
            
            if cnt == 0:
                out_tx = tx
                cnt = 1
                # sch.annotate(out_tx, ann_key="pragma_auto_unroll_max_step", ann_val=128)
                # sch.annotate(out_tx, ann_key="pragma_unroll_explicit", ann_val=1)

            if len(r_loops) > 0:
                reduction_blocks.append((block, r_loops[0]))
            else:
                try:
                    sch.reverse_compute_at(block, out_tx, True)
                    for loop, iter_type in zip(sch.get_loops(block), dom_kind):
                        {"S": s_loops, "R": r_loops, "O": o_loops}[iter_type].append(loop)

                    if not s_loops:
                        s_loops.append(sch.add_unit_loop(block))

                    if len(s_loops) > 1:
                        _, vx = sch.split(s_loops[-1], factors=[None, vector_size])
                        tx, _ = sch.split(sch.fuse(*s_loops[:-1]), factors=[core_num, None])
                    else:
                        tx, _, vx = sch.split(s_loops[0], factors=[core_num, None, vector_size])
                    sch.vectorize(vx)
                    # sch.vectorize(sch.split(sch.get_loops(block)[-1], factors=[None, vector_size])[-1])
                except:
                    pass

        for block, r_loop in reduction_blocks:
            sch.decompose_reduction(block, r_loop)

        return sch
