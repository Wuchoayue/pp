/**
* @file testcase.cpp
*
* Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#include "op_test.h"
#include "op_execute.h"
using namespace OpTest;


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_001)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_FLOAT16};
    opTestDesc.inputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_001_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_001_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_001");

}


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_002)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_INT32};
    opTestDesc.inputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_002_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_INT32};
    opTestDesc.outputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_002_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_002");

}


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_003)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_FLOAT};
    opTestDesc.inputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_003_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_FLOAT};
    opTestDesc.outputFormat = {ACL_FORMAT_NCHW};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_003_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_003");

}


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_004)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_FLOAT16};
    opTestDesc.inputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_004_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_FLOAT16};
    opTestDesc.outputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_004_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_004");

}


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_005)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_INT32};
    opTestDesc.inputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_005_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_INT32};
    opTestDesc.outputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_005_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_005");

}


OP_TEST(CeluNpu, Test_CeluNpu_001_sub_case_006)
{
    
    std::string opType = "CeluNpu";
    OpTestDesc opTestDesc(opType);
    // input parameter init
    opTestDesc.inputShape = {{1, 6, 128, 128}};
    opTestDesc.inputDataType = {ACL_FLOAT};
    opTestDesc.inputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.inputFilePath = {"test_data/data/Test_CeluNpu_001_sub_case_006_input_0"};
    // output parameter init
    opTestDesc.outputShape = {{1, 6, 128, 128}};
    opTestDesc.outputDataType = {ACL_FLOAT};
    opTestDesc.outputFormat = {ACL_FORMAT_NHWC};
    opTestDesc.outputFilePath = {"result_files/Test_CeluNpu_001_sub_case_006_output_0"};
    // attr parameter init
        OpTestAttr attr0 = {OP_FLOAT, "alpha"};
    attr0.floatAttr = 0.1;
    opTestDesc.opAttrVec.push_back(attr0);

    EXPECT_EQ_AND_RECORD(true, OpExecute(opTestDesc), opTestDesc, "Test_CeluNpu_001_sub_case_006");

}

