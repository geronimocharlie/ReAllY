��
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.3.02unknown8��
�
my_model_4/dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namemy_model_4/dense_8/kernel
�
-my_model_4/dense_8/kernel/Read/ReadVariableOpReadVariableOpmy_model_4/dense_8/kernel*
_output_shapes

:*
dtype0
�
my_model_4/dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namemy_model_4/dense_8/bias

+my_model_4/dense_8/bias/Read/ReadVariableOpReadVariableOpmy_model_4/dense_8/bias*
_output_shapes
:*
dtype0
�
my_model_4/dense_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:**
shared_namemy_model_4/dense_9/kernel
�
-my_model_4/dense_9/kernel/Read/ReadVariableOpReadVariableOpmy_model_4/dense_9/kernel*
_output_shapes

:*
dtype0
�
my_model_4/dense_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namemy_model_4/dense_9/bias

+my_model_4/dense_9/bias/Read/ReadVariableOpReadVariableOpmy_model_4/dense_9/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�

value�
B�
 B�

y
	layer

layer2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 

0
	1
2
3

0
	1
2
3
�
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables
	variables
layer_metrics
 
VT
VARIABLE_VALUEmy_model_4/dense_8/kernel'layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEmy_model_4/dense_8/bias%layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
	1

0
	1
�

regularization_losses

layers
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables
	variables
layer_metrics
WU
VARIABLE_VALUEmy_model_4/dense_9/kernel(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEmy_model_4/dense_9/bias&layer2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
�
regularization_losses

layers
layer_regularization_losses
 non_trainable_variables
!metrics
trainable_variables
	variables
"layer_metrics

0
1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_4/dense_8/kernelmy_model_4/dense_8/biasmy_model_4/dense_9/kernelmy_model_4/dense_9/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference_signature_wrapper_41429
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename-my_model_4/dense_8/kernel/Read/ReadVariableOp+my_model_4/dense_8/bias/Read/ReadVariableOp-my_model_4/dense_9/kernel/Read/ReadVariableOp+my_model_4/dense_9/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *'
f"R 
__inference__traced_save_41502
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_4/dense_8/kernelmy_model_4/dense_8/biasmy_model_4/dense_9/kernelmy_model_4/dense_9/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_restore_41524��
�
�
B__inference_dense_8_layer_call_and_return_conditional_losses_41357

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_dense_9_layer_call_and_return_conditional_losses_41383

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
B__inference_dense_8_layer_call_and_return_conditional_losses_41439

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
#__inference_signature_wrapper_41429
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *)
f$R"
 __inference__wrapped_model_413432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
 __inference__wrapped_model_41343
input_15
1my_model_4_dense_8_matmul_readvariableop_resource6
2my_model_4_dense_8_biasadd_readvariableop_resource5
1my_model_4_dense_9_matmul_readvariableop_resource6
2my_model_4_dense_9_biasadd_readvariableop_resource
identity��
(my_model_4/dense_8/MatMul/ReadVariableOpReadVariableOp1my_model_4_dense_8_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(my_model_4/dense_8/MatMul/ReadVariableOp�
my_model_4/dense_8/MatMulMatMulinput_10my_model_4/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
my_model_4/dense_8/MatMul�
)my_model_4/dense_8/BiasAdd/ReadVariableOpReadVariableOp2my_model_4_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)my_model_4/dense_8/BiasAdd/ReadVariableOp�
my_model_4/dense_8/BiasAddBiasAdd#my_model_4/dense_8/MatMul:product:01my_model_4/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
my_model_4/dense_8/BiasAdd�
(my_model_4/dense_9/MatMul/ReadVariableOpReadVariableOp1my_model_4_dense_9_matmul_readvariableop_resource*
_output_shapes

:*
dtype02*
(my_model_4/dense_9/MatMul/ReadVariableOp�
my_model_4/dense_9/MatMulMatMul#my_model_4/dense_8/BiasAdd:output:00my_model_4/dense_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
my_model_4/dense_9/MatMul�
)my_model_4/dense_9/BiasAdd/ReadVariableOpReadVariableOp2my_model_4_dense_9_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)my_model_4/dense_9/BiasAdd/ReadVariableOp�
my_model_4/dense_9/BiasAddBiasAdd#my_model_4/dense_9/MatMul:product:01my_model_4/dense_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
my_model_4/dense_9/BiasAddw
IdentityIdentity#my_model_4/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������:::::P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
B__inference_dense_9_layer_call_and_return_conditional_losses_41458

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������:::O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
|
'__inference_dense_8_layer_call_fn_41448

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_413572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_my_model_4_layer_call_fn_41414
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_my_model_4_layer_call_and_return_conditional_losses_414002
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
E__inference_my_model_4_layer_call_and_return_conditional_losses_41400
input_1
dense_8_41368
dense_8_41370
dense_9_41394
dense_9_41396
identity��dense_8/StatefulPartitionedCall�dense_9/StatefulPartitionedCall�
dense_8/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_8_41368dense_8_41370*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_413572!
dense_8/StatefulPartitionedCall�
dense_9/StatefulPartitionedCallStatefulPartitionedCall(dense_8/StatefulPartitionedCall:output:0dense_9_41394dense_9_41396*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_413832!
dense_9/StatefulPartitionedCall�
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0 ^dense_8/StatefulPartitionedCall ^dense_9/StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:���������::::2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall2B
dense_9/StatefulPartitionedCalldense_9/StatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
!__inference__traced_restore_41524
file_prefix.
*assignvariableop_my_model_4_dense_8_kernel.
*assignvariableop_1_my_model_4_dense_8_bias0
,assignvariableop_2_my_model_4_dense_9_kernel.
*assignvariableop_3_my_model_4_dense_9_bias

identity_5��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOp*assignvariableop_my_model_4_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOp*assignvariableop_1_my_model_4_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp,assignvariableop_2_my_model_4_dense_9_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp*assignvariableop_3_my_model_4_dense_9_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4�

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
�
__inference__traced_save_41502
file_prefix8
4savev2_my_model_4_dense_8_kernel_read_readvariableop6
2savev2_my_model_4_dense_8_bias_read_readvariableop8
4savev2_my_model_4_dense_9_kernel_read_readvariableop6
2savev2_my_model_4_dense_9_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const�
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_2bf32927c8fe47d0a50f91b440ae30bb/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�
value�B�B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:04savev2_my_model_4_dense_8_kernel_read_readvariableop2savev2_my_model_4_dense_8_bias_read_readvariableop4savev2_my_model_4_dense_9_kernel_read_readvariableop2savev2_my_model_4_dense_9_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*7
_input_shapes&
$: ::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
�
|
'__inference_dense_9_layer_call_fn_41467

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *K
fFRD
B__inference_dense_9_layer_call_and_return_conditional_losses_413832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������<
q_values0
StatefulPartitionedCall:0���������tensorflow/serving/predict:�;
�
	layer

layer2
regularization_losses
trainable_variables
	variables
	keras_api

signatures
#_default_save_signature
*$&call_and_return_all_conditional_losses
%__call__"�
_tf_keras_model�{"class_name": "MyModel", "name": "my_model_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}}
�

kernel
	bias

regularization_losses
trainable_variables
	variables
	keras_api
*&&call_and_return_all_conditional_losses
'__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
�

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
*(&call_and_return_all_conditional_losses
)__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_9", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
 "
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
<
0
	1
2
3"
trackable_list_wrapper
�
regularization_losses

layers
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables
	variables
layer_metrics
%__call__
#_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
+:)2my_model_4/dense_8/kernel
%:#2my_model_4/dense_8/bias
 "
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
�

regularization_losses

layers
layer_regularization_losses
non_trainable_variables
metrics
trainable_variables
	variables
layer_metrics
'__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
+:)2my_model_4/dense_9/kernel
%:#2my_model_4/dense_9/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
regularization_losses

layers
layer_regularization_losses
 non_trainable_variables
!metrics
trainable_variables
	variables
"layer_metrics
)__call__
*(&call_and_return_all_conditional_losses
&("call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
 __inference__wrapped_model_41343�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
E__inference_my_model_4_layer_call_and_return_conditional_losses_41400�
���
FullArgSpec
args�
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
*__inference_my_model_4_layer_call_fn_41414�
���
FullArgSpec
args�
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
B__inference_dense_8_layer_call_and_return_conditional_losses_41439�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_8_layer_call_fn_41448�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_dense_9_layer_call_and_return_conditional_losses_41458�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_dense_9_layer_call_fn_41467�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
2B0
#__inference_signature_wrapper_41429input_1�
 __inference__wrapped_model_41343m	0�-
&�#
!�
input_1���������
� "3�0
.
q_values"�
q_values����������
B__inference_dense_8_layer_call_and_return_conditional_losses_41439\	/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_dense_8_layer_call_fn_41448O	/�,
%�"
 �
inputs���������
� "�����������
B__inference_dense_9_layer_call_and_return_conditional_losses_41458\/�,
%�"
 �
inputs���������
� "%�"
�
0���������
� z
'__inference_dense_9_layer_call_fn_41467O/�,
%�"
 �
inputs���������
� "�����������
E__inference_my_model_4_layer_call_and_return_conditional_losses_41400y	0�-
&�#
!�
input_1���������
� "?�<
5�2
0
q_values$�!

0/q_values���������
� �
*__inference_my_model_4_layer_call_fn_41414m	0�-
&�#
!�
input_1���������
� "3�0
.
q_values"�
q_values����������
#__inference_signature_wrapper_41429x	;�8
� 
1�.
,
input_1!�
input_1���������"3�0
.
q_values"�
q_values���������