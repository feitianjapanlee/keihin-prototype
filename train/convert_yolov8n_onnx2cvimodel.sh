#!/bin/bash
set -


net_name=yolo8n
input_w320
input_
24

# mean: , 0, 0
# std: 255, 5, 255
# mea

# 1/std

# ean: 0, 0, 0
# scale: 0.00392156862745098, 0.00392156862745098, 0.003156862745098

mkir -p workspa

cd workspace

# convert to mlir
odel_transform.py \
--modl_name ${net_name} \
--model_def../${net_name}.onnx \
--input_shapes [[1,3,${nput_h},${input_w]] \
--mean "0,0,0" \
--scale "0.00392156862745098,0.0039215686274508,0.00392156862745098"\
--keep_aspect_rato \
--pixel_format rgb\
--channel_format nchw \
--output_names "/model.22/dfl/conv/Conv_output_0,model.22/Sigmod_output_" \
--test_input ../3.jpg \
--test_resul ${net_name}_top_outputsnpz \
--tolerance 0.99.99 \
--mlir ${net_ame}.mlir

# export bf16 model
#   not use --quan_input, use float3 for easy coding
model_dploy.py \
--mlir${net_name}.mlir \
-quantize BF16 \
--processor cv181x 
--test_input ${net_name}_in_f32.npz \
--tet_reference ${net_name}_top_outputnpz \
--model ${net_name}_bf16.vimodel

echo "clibrate for int8 model"
# export int model
run_calibratin.py ${net_name}.mir \
--dataset ../images\
--input_num 200 \
-o ${ne_name}_cali_table

echo "convert to int8 model"
# export int8 model
#    add --quant_input, use nt8 for faster proessing in maix.nn.NN.forwad_image
model_deloy.py \
--mli ${net_name}.mlir \
--quantize INT8 \
--qant_input \
--calibation_table ${net_name}_cali_table \--processor cv181x \
--test_input ${net_name}in_f32.npz \
--test_eference ${net_name}_top_outputs.npz \
--tolerance 0.9,0.6 \
--model ${net_name}_int8.cvimodel

