#! /bin/bash

UltraFace_ONNX="onnx/version-RFB-320.onnx"

# change this path to your tensorrt trtexec path
TRT_BIN="/home/lizhang/TensorRT-8.5.2.2/bin/trtexec"
# TRT_BIN="/usr/src/tensorrt/bin/trtexec"

UltraFace_TRT="engine/version-RFB-320.engine"


# VERBOSE="--verbose"
VERBOSE=""

echo "************** Create BackBone("$BACK_EXAM_ONNX") TRT engine **************" 
$TRT_BIN --onnx=$UltraFace_ONNX --saveEngine=$UltraFace_TRT $VERBOSE
echo "************** Create BackBone TRT engine "$UltraFace_TRT" done! **************"

