/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
Description: op_proto for celu operator
Author:Wuchaoyue
Create: 2021-7-23
*/
#include "./celu_npu.h"

namespace ge {

    IMPLEMT_VERIFIER(CeluNpu, CeluNpuVerify) {

        return GRAPH_SUCCESS;
    }
    IMPLEMT_INFERFUNC(CeluNpu, CeluNpuInferShape) {
        auto x_shape = op.GetInputDesc("x").GetShape().GetDims();
        DataType x_dtype = op.GetInputDesc("x").GetDataType();
        TensorDesc y_desc = op.GetOutputDesc("y");
        y_desc.SetShape(ge::Shape(x_shape));
        y_desc.SetDataType(x_dtype);
        (void)op.UpdateOutputDesc("y", y_desc);
        return GRAPH_SUCCESS;
    }
    INFER_FUNC_REG(CeluNpu, CeluNpuInferShape);
    VERIFY_FUNC_REG(CeluNpu, CeluNpuVerify);

}