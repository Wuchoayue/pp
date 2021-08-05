#include <gtest/gtest.h>
#include <vector>
#include "celu_npu.h"

class CeluNpuTest : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "celu_npu test SetUp" << std::endl;
}

    static void TearDownTestCase() {
        std::cout << "celu_npu test TearDown" << std::endl;
    }
};

TEST_F(CeluNpuTest, celu_npu_test_case_1) {
    // [TODO] define your op here
    // ge::op::CeluNpu celu_npu_op;
    // ge::TensorDesc tensorDesc;
    // ge::Shape shape({2, 3, 4});
    // tensorDesc.SetDataType(ge::DT_FLOAT16);
    // tensorDesc.SetShape(shape);

    // [TODO] update op input here
    // celu_npu_op.UpdateInputDesc("x1", tensorDesc);
    // celu_npu_op.UpdateInputDesc("x2", tensorDesc);

    // [TODO] call InferShapeAndType function here
    // auto ret = celu_npu_op.InferShapeAndType();
    // EXPECT_EQ(ret, ge::GRAPH_SUCCESS);

    // [TODO] compare dtype and shape of op output
    // auto output_desc = celu_npu_op.GetOutputDesc("y");
    // EXPECT_EQ(output_desc.GetDataType(), ge::DT_FLOAT16);
    // std::vector<int64_t> expected_output_shape = {2, 3, 4};
    // EXPECT_EQ(output_desc.GetShape().GetDims(), expected_output_shape);
}
