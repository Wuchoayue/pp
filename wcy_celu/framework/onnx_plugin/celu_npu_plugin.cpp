/* Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: framework for celu operator
 * Author:Wuchaoyue
 * Create: 2020-7-23
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the Apache License Version 2.0.You may not use
 * this file except in compliance with the License.
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * Apache License for more details at
 * http://www.apache.org/licenses/LICENSE-2.0
 * file celu_plugin.cpp
 */


#include "ge_onnx.pb.h"
#include "register/register.h"
#include <string>
#include <vector>

namespace domi {
    Status ParseParamsCelu(const Message *op_src, ge::Operator& op_dst)
    {
        const ge::onnx::NodeProto *node = reinterpret_cast<const ge::onnx::NodeProto*>(op_src);
        if (node == nullptr) {
            // OP_LOGE(op_dst.GetName().c_str(), "Dynamic cast op_src to NodeProto failed.");
            return FAILED;
        }

        float negativeSlope = 0.0f;
        bool bFindAlpha = false;
        for (auto attr : node->attribute()) {
            if (attr.name() == "alpha") {
                bFindAlpha = true;
                negativeSlope = attr.f();
                break;
            }
        }

        if (!bFindAlpha) {
            negativeSlope = 0.0f;
        }
        op_dst.SetAttr("alpha", negativeSlope);
        return SUCCESS;
    }

    REGISTER_CUSTOM_OP("CeluNpu")
    .FrameworkType(ONNX)
    .OriginOpType("ai.onnx::11::CeluNpu")
    .ParseParamsFn(ParseParamsCelu)
    .ImplyType(ImplyType::TVM);
}