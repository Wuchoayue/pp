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
@fusion_manager.register("wu_relu")
def wu_relu_compute(x, y, negative_slope=0, kernel_name="wu_relu"):
    """
    compute for caffe_relu_layer_cce
    """
    inp_dtype = x.dtype.lower()
    shape = x.shape

    leaky_1 = tvm.const(0.5+0.5*negative_slope, inp_dtype)
    leaky_2 = tvm.const(0.5-0.5*negative_slope, inp_dtype)

    if inp_dtype in ("float16", "float32"):
        res_1 = te.lang.cce.vmuls(x, leaky_1)
        res_2 = te.lang.cce.vmuls(te.lang.cce.vabs(x), leaky_2)
        res = te.lang.cce.vadd(res_1, res_2)
    else:
        res_1 = te.lang.cce.vmuls(x, leaky_1)
        zero_tensor = te.lang.cce.broadcast(tvm.const(0, inp_dtype), shape)
        x_temp = te.lang.cce.vsub(zero_tensor, x)
        x_last = te.lang.cce.vmax(x_temp, x)
        res_2 = te.lang.cce.vmuls(x_last, leaky_2)
        res = te.lang.cce.vadd(res_1, res_2)

    return te.lang.cce.cast_to(res, inp_dtype)



def wu_relu(x, y, negative_slope=0, kernel_name="wu_relu"):
    """leaky_relu op for input tensor

       f(x)= x(x>=0) or negative_slope*x(x<0) equal to
       f(x)=negative_slope*x

    Parameters
    ----------
    x : TVM tensor
        input tensor has shape and dtype attributes
    y : dict
        dict with keys(shape and dtype) of output

    negative_slope : float or int
        allow non-zero slope for negative inputs to speed up optimization

    kernel_name : str
        cce kernel name, default value is "leaky_relu"

    Returns
    ------
    None
    """

    # check input tensor shape
    shape = x.get("shape")
    dtype = x.get("dtype")

    # check input tensor data_type
    check_list = ["float16", "float32", "int32", "int8"]
    if dtype.lower() not in check_list:
        raise RuntimeError(
            "leaky relu only support %s while dtype is %s"
            % (",".join(check_list), dtype))

    inp_dtype = dtype.lower()
    input_data_x = tvm.placeholder(shape, name="input_data_x", dtype=inp_dtype)

    with tvm.target.cce():

        res = wu_relu_compute(input_data_x, y, negative_slope, kernel_name)
        sch = generic.auto_schedule(res)

    config = {"name": kernel_name,
              "tensor_list": [input_data_x, res]}
    te.lang.cce.cce_build_code(sch, config)
