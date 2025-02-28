# w4_conv_tir.py
from tvm.script import tir as T
dtype = "float16"
def create_w4_conv_oc(N_val=None, IC_val=None, OC_val=None, IW_val=None, FW_val=None, pad_val=0,strides=1, dtype = dtype, bias=False, bias_relu=False, bias_relu_transpose=False):
    N = N_val if N_val is not None else T.int64()
    IC = IC_val if IC_val is not None else T.int64()
    OC = OC_val if OC_val is not None else T.int64()
    IW = IW_val if IW_val is not None else T.int64()
    FW = FW_val if FW_val is not None else T.int64()
    pad = pad_val if pad_val is not None else T.int64()

    @T.prim_func
    def w4_conv_oc(a: T.handle, w: T.handle, d: T.handle, c: T.handle):        
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, IC, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, IC, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, IC, OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, IC):
            with T.block("w4_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C[vn, vw, voc] = 0.0
                C[vn, vw, voc] = C[vn, vw, voc] + pad_A[vn, vw * strides + vfw, vic] * W_unpack[vfw, vic, voc]

    @T.prim_func
    def w4_conv_oc_with_bias(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):        
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, IC, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, IC, OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, IC, OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, IC):
            with T.block("w4_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, vic] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, vw, voc] = C_tmp[vn, vw, voc] + B[0, 0, voc]

    @T.prim_func
    def w4_conv_oc_with_bias_with_relu(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):        
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, IC, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, IC, OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, IC, OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, IC):
            with T.block("w4_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, vic] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, vw, voc] = T.max(C_tmp[vn, vw, voc] + B[0, 0, voc], T.float16(0))

    @T.prim_func
    def w4_conv_oc_with_bias_with_relu_with_transpose(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):        
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, IC, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, OC, IW), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, IC, OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, IC, OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, IC):
            with T.block("w4_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, vic] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, voc, vw] = T.max(C_tmp[vn, vw, voc] + B[0, 0, voc], T.float16(0))

    if bias_relu_transpose:
        return w4_conv_oc_with_bias_with_relu_with_transpose
    elif bias_relu:
        return w4_conv_oc_with_bias_with_relu
    elif bias:
        return w4_conv_oc_with_bias
    else:
        return w4_conv_oc

def create_w4_group_conv_oc(N_val=None, IC_val=None, OC_val=None, IW_val=None, FW_val=None, pad_val=0,strides=1, dtype = dtype, bias=False, bias_relu=False, bias_relu_transpose=False):
    N = N_val if N_val is not None else T.int64()
    IC = IC_val if IC_val is not None else T.int64()
    OC = OC_val if OC_val is not None else T.int64()
    IW = IW_val if IW_val is not None else T.int64()
    FW = FW_val if FW_val is not None else T.int64()
    pad = pad_val if pad_val is not None else T.int64()

    @T.prim_func
    def w4_group_conv_oc(a: T.handle, w: T.handle, d: T.handle, c: T.handle):
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, 1, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, T.int64(1), OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, T.int64(1), OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, T.int64(1)):
            with T.block("w4_group_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C[vn, vw, voc] = 0.0
                C[vn, vw, voc] = C[vn, vw, voc] + pad_A[vn, vw * strides + vfw, voc] * W_unpack[vfw, vic, voc]

    @T.prim_func
    def w4_group_conv_oc_with_bias(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, 1, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, T.int64(1), OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, T.int64(1), OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, T.int64(1)):
            with T.block("w4_group_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, voc] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, vw, voc] = C_tmp[vn, vw, voc] + B[0, 0, voc]

    @T.prim_func
    def w4_group_conv_oc_with_bias_with_relu(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, 1, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, T.int64(1), OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, T.int64(1), OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, T.int64(1)):
            with T.block("w4_group_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, voc] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, vw, voc] = T.max(C_tmp[vn, vw, voc] + B[0, 0, voc], T.float16(0))

    @T.prim_func
    def w4_group_conv_oc_with_bias_with_relu_with_transpose(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC // 2, 1, FW), "int8")
        D = T.match_buffer(d, (OC, 1, 1), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, OC, IW), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC),dtype)
        W_unpack = T.alloc_buffer((FW, T.int64(1), OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, T.int64(1), OC // 2):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic, voc * 2] = T.Cast(dtype, (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8)) * D[voc * 2, 0, 0])
                W_unpack[vfw, vic, voc * 2 + 1] = T.Cast(dtype, (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8)) * D[voc * 2 + 1, 0, 0])
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, T.int64(1)):
            with T.block("w4_group_conv_oc"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, voc] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, voc, vw] = T.max(C_tmp[vn, vw, voc] + B[0, 0, voc], T.float16(0))

    if bias_relu_transpose:
        return w4_group_conv_oc_with_bias_with_relu_with_transpose
    elif bias_relu:
        return w4_group_conv_oc_with_bias_with_relu
    elif bias:
        return w4_group_conv_oc_with_bias
    else:
        return w4_group_conv_oc

def create_w4_conv_ic(N_val=None, IC_val=None, OC_val=None, IW_val=None, FW_val=None, pad_val=0,strides=1, dtype = dtype):
    N = N_val if N_val is not None else T.int64()
    IC = IC_val if IC_val is not None else T.int64()
    OC = OC_val if OC_val is not None else T.int64()
    IW = IW_val if IW_val is not None else T.int64()
    FW = FW_val if FW_val is not None else T.int64()
    pad = pad_val if pad_val is not None else T.int64()
    @T.prim_func
    def w4_conv_ic(a: T.handle, w: T.handle, d: T.handle, b: T.handle, c: T.handle):        
        A = T.match_buffer(a, (N, IW, IC), dtype)
        W = T.match_buffer(w, (OC, IC // 2, FW), "int8")
        D = T.match_buffer(d, (), dtype)
        B = T.match_buffer(b, (1, 1, OC), dtype)
        C = T.match_buffer(c, (N, IW, OC), dtype)
        pad_A = T.alloc_buffer((N, IW + pad + pad, IC), dtype)
        W_unpack = T.alloc_buffer((FW, IC, OC), dtype)
        C_tmp = T.alloc_buffer((N, IW, OC), dtype)
        for n, pad_iw, ic in T.grid(N, IW + pad + pad, IC):
            with T.block("pad_A"):
                v_n, v_pad_iw, v_ic = T.axis.remap("SSS", [n, pad_iw, ic])
                T.reads(A[v_n, v_pad_iw - pad, v_ic])
                T.writes(pad_A[v_n, v_pad_iw, v_ic])
                pad_A[v_n, v_pad_iw, v_ic] = T.if_then_else(pad <= v_pad_iw and v_pad_iw < IW + pad, A[v_n, v_pad_iw - pad, v_ic], T.Cast(dtype, 0.0))
        for fw, ic, oc in T.grid(FW, IC // 2, OC):
            with T.block("unpack"):
                vfw, vic, voc = T.axis.remap("SSS", [fw, ic, oc])
                W_unpack[vfw, vic * 2, voc] = (T.float16(1.0) * (T.bitwise_and(T.shift_right(W[voc, vic, vfw], 4), 0xf) - T.int8(8))) * D[()]
                W_unpack[vfw, vic * 2 + 1, voc] = (T.float16(1.0) * (T.bitwise_and(W[voc, vic, vfw], 0xf) - T.int8(8))) * D[()]
        for n, w, oc, fw, ic in T.grid(N, IW, OC, FW, IC):
            with T.block("w4_conv_ic"):
                vn, vw, voc, vfw, vic = T.axis.remap("SSSRR", [n, w, oc, fw, ic])
                with T.init():
                    C_tmp[vn, vw, voc] = T.Cast(dtype, 0.0)
                C_tmp[vn, vw, voc] = C_tmp[vn, vw, voc] + pad_A[vn, vw * strides + vfw, vic] * W_unpack[vfw, vic, voc]
        for n, w, oc in T.grid(N, IW, OC):
            with T.block("bias"):
                vn, vw, voc = T.axis.remap("SSS", [n, w, oc])
                C[vn, vw, voc] = C_tmp[vn, vw, voc] + B[0, 0, voc]
    return w4_conv_ic