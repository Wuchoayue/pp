#include <gtest/gtest.h>
#include <vector>
#include "wu_relu_npu.h"

class WuReluNpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "wu_relu_npu test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "wu_relu_npu test TearDown" << std::endl;
    }
};

TEST_F(WuReluNpuTest, wu_relu_npu_test_case_1) {
    // [TODO] define your op here
    // ge::op::WuReluNpu wu_relu_npu_op;
    // ge::TensorDesc tensorDesc;
    // ge::Shape shape({2, 3, 4});
    // tensorDesc.SetDataType(ge::DT_FLOAT16);
    // tensorDesc.SetShape(shape);

    // [TODO] update op input here
    // wu_relu_npu_op.UpdateInputDesc("x1", tensorDesc);
    // wu_relu_npu_op.UpdateInputDesc("x2", tensorDesc);

    // [TODO] call InferShapeAndType function here
    // auto ret = wu_relu_npu_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    // auto output_desc = wu_relu_npu_op.GetOutputDesc("y");
    // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    // std::vector<int64_t> expected_output_shape = {2, 3, 4};
    // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
