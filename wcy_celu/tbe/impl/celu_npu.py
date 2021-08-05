#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2018. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this
file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

cce extended operator builder wrapper
"""

from te import tvm

import te.lang.cce
from te.platform.fusion_manager import fusion_manager
from topi import generic



# pylint: disable=locally-disabled,unused-argument,invalid-name
@fusion_manager.register("celu")
def celu_compute(x, y, alpha=0, kernel_name="celu"):

    inp_dtype = x.dtype.lower()
    shape = x.shape

    if alpha == 0:
        if inp_dtype in ("float","float16"):
            res = te.lang.cce.vrelu(x)
        else:
            zero_tensor = te.lang.cce.broadcast(tvm.const(0, inp_dtype), shape)
            res = te.lang.cce.vmax(x, zero_tensor)
    else:
        zero_tensor = te.lang.cce.broadcast(tvm.const(0, inp_dtype), shape)
        alpha_tensor = te.lang.cce.broadcast(tvm.const(alpha, dtype=inp_dtype), shape)
        res_divide = te.lang.cce.vdiv(x, alpha_tensor)
        res_exp = te.lang.cce.vsub(te.lang.cce.vexp(res_divide), te.lang.cce.broadcast(tvm.const(1, dtype=inp_dtype), shape))
        res_temp1 = te.lang.cce.vmax(x, zero_tensor)
        res_temp2 = te.lang.cce.vmin(zero_tensor, te.lang.cce.vmuls(res_exp, alpha))
        res = te.lang.cce.vadd(res_temp1, res_temp2)

        if inp_dtype in ("int32"):
            res = te.lang.cce.round(res)

    return te.lang.cce.cast_to(res, inp_dtype)



def celu(x, y, alpha=0, kernel_name="celu"):

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")

    # check input tensor data_type
    check_list = ["float16", "float", "int32"]
    if dtype.lower() not in check_list:
        raise RuntimeError(
            "leaky relu only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    inp_dtype = dtype.lower()
    input_data_x = tvm.placeholder(shape, name="input_data_x", dtype=inp_dtype)

    with tvm.target.cce():

        res = celu_compute(input_data_x, y, alpha, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
