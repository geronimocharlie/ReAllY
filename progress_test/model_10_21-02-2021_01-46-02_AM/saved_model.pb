ох
њ£
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
dtypetypeИ
Њ
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
executor_typestring И
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.3.02unknown8Г∞
Т
my_model_26/dense_52/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namemy_model_26/dense_52/kernel
Л
/my_model_26/dense_52/kernel/Read/ReadVariableOpReadVariableOpmy_model_26/dense_52/kernel*
_output_shapes

:*
dtype0
К
my_model_26/dense_52/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namemy_model_26/dense_52/bias
Г
-my_model_26/dense_52/bias/Read/ReadVariableOpReadVariableOpmy_model_26/dense_52/bias*
_output_shapes
:*
dtype0
Т
my_model_26/dense_53/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_namemy_model_26/dense_53/kernel
Л
/my_model_26/dense_53/kernel/Read/ReadVariableOpReadVariableOpmy_model_26/dense_53/kernel*
_output_shapes

:*
dtype0
К
my_model_26/dense_53/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namemy_model_26/dense_53/bias
Г
-my_model_26/dense_53/bias/Read/ReadVariableOpReadVariableOpmy_model_26/dense_53/bias*
_output_shapes
:*
dtype0

NoOpNoOp
 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Е
valueы
Bш
 Bс

y
	layer

layer2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
h

kernel
	bias

trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api

0
	1
2
3

0
	1
2
3
 
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
 
XV
VARIABLE_VALUEmy_model_26/dense_52/kernel'layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEmy_model_26/dense_52/bias%layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
	1

0
	1
 
≠

trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
YW
VARIABLE_VALUEmy_model_26/dense_53/kernel(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEmy_model_26/dense_53/bias&layer2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics
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
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
Њ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1my_model_26/dense_52/kernelmy_model_26/dense_52/biasmy_model_26/dense_53/kernelmy_model_26/dense_53/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *-
f(R&
$__inference_signature_wrapper_396971
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/my_model_26/dense_52/kernel/Read/ReadVariableOp-my_model_26/dense_52/bias/Read/ReadVariableOp/my_model_26/dense_53/kernel/Read/ReadVariableOp-my_model_26/dense_53/bias/Read/ReadVariableOpConst*
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
GPU 2J 8В *(
f#R!
__inference__traced_save_397045
К
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemy_model_26/dense_52/kernelmy_model_26/dense_52/biasmy_model_26/dense_53/kernelmy_model_26/dense_53/bias*
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
GPU 2J 8В *+
f&R$
"__inference__traced_restore_397067іП
Ќ
ђ
D__inference_dense_53_layer_call_and_return_conditional_losses_397000

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
л
д
"__inference__traced_restore_397067
file_prefix0
,assignvariableop_my_model_26_dense_52_kernel0
,assignvariableop_1_my_model_26_dense_52_bias2
.assignvariableop_2_my_model_26_dense_53_kernel0
,assignvariableop_3_my_model_26_dense_53_bias

identity_5ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3…
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*’
valueЋB»B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_namesШ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesƒ
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

IdentityЂ
AssignVariableOpAssignVariableOp,assignvariableop_my_model_26_dense_52_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1±
AssignVariableOp_1AssignVariableOp,assignvariableop_1_my_model_26_dense_52_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2≥
AssignVariableOp_2AssignVariableOp.assignvariableop_2_my_model_26_dense_53_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3±
AssignVariableOp_3AssignVariableOp,assignvariableop_3_my_model_26_dense_53_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4ђ

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
°
Ў
__inference__traced_save_397045
file_prefix:
6savev2_my_model_26_dense_52_kernel_read_readvariableop8
4savev2_my_model_26_dense_52_bias_read_readvariableop:
6savev2_my_model_26_dense_53_kernel_read_readvariableop8
4savev2_my_model_26_dense_53_bias_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
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
ConstН
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_edb6dc76bcc544b99dc700df7671d426/part2	
Const_1Л
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
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename√
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*’
valueЋB»B'layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB%layer/bias/.ATTRIBUTES/VARIABLE_VALUEB(layer2/kernel/.ATTRIBUTES/VARIABLE_VALUEB&layer2/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesТ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesЪ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_my_model_26_dense_52_kernel_read_readvariableop4savev2_my_model_26_dense_52_bias_read_readvariableop6savev2_my_model_26_dense_53_kernel_read_readvariableop4savev2_my_model_26_dense_53_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
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
№
~
)__inference_dense_52_layer_call_fn_396990

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_3968942
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
В	
∞
,__inference_my_model_26_layer_call_fn_396954
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1ИҐStatefulPartitionedCall¶
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *P
fKRI
G__inference_my_model_26_layer_call_and_return_conditional_losses_3969382
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
№
~
)__inference_dense_53_layer_call_fn_397009

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallф
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_53_layer_call_and_return_conditional_losses_3969202
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
Ќ
ђ
D__inference_dense_52_layer_call_and_return_conditional_losses_396981

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
‘
®
$__inference_signature_wrapper_396971
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identity

identity_1ИҐStatefulPartitionedCallА
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *:
_output_shapes(
&:€€€€€€€€€:€€€€€€€€€*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В **
f%R#
!__inference__wrapped_model_3968802
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

IdentityТ

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ
ђ
D__inference_dense_52_layer_call_and_return_conditional_losses_396894

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ф
Ј
!__inference__wrapped_model_396880
input_17
3my_model_26_dense_52_matmul_readvariableop_resource8
4my_model_26_dense_52_biasadd_readvariableop_resource7
3my_model_26_dense_53_matmul_readvariableop_resource8
4my_model_26_dense_53_biasadd_readvariableop_resource
identity

identity_1Ић
*my_model_26/dense_52/MatMul/ReadVariableOpReadVariableOp3my_model_26_dense_52_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*my_model_26/dense_52/MatMul/ReadVariableOp≥
my_model_26/dense_52/MatMulMatMulinput_12my_model_26/dense_52/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_26/dense_52/MatMulЋ
+my_model_26/dense_52/BiasAdd/ReadVariableOpReadVariableOp4my_model_26_dense_52_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+my_model_26/dense_52/BiasAdd/ReadVariableOp’
my_model_26/dense_52/BiasAddBiasAdd%my_model_26/dense_52/MatMul:product:03my_model_26/dense_52/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_26/dense_52/BiasAddћ
*my_model_26/dense_53/MatMul/ReadVariableOpReadVariableOp3my_model_26_dense_53_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*my_model_26/dense_53/MatMul/ReadVariableOp—
my_model_26/dense_53/MatMulMatMul%my_model_26/dense_52/BiasAdd:output:02my_model_26/dense_53/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_26/dense_53/MatMulЋ
+my_model_26/dense_53/BiasAdd/ReadVariableOpReadVariableOp4my_model_26_dense_53_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+my_model_26/dense_53/BiasAdd/ReadVariableOp’
my_model_26/dense_53/BiasAddBiasAdd%my_model_26/dense_53/MatMul:product:03my_model_26/dense_53/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
my_model_26/dense_53/BiasAddy
IdentityIdentity%my_model_26/dense_52/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity}

Identity_1Identity%my_model_26/dense_53/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€:::::P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1
Ќ
ђ
D__inference_dense_53_layer_call_and_return_conditional_losses_396920

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИН
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€:::O K
'
_output_shapes
:€€€€€€€€€
 
_user_specified_nameinputs
ќ
С
G__inference_my_model_26_layer_call_and_return_conditional_losses_396938
input_1
dense_52_396905
dense_52_396907
dense_53_396931
dense_53_396933
identity

identity_1ИҐ dense_52/StatefulPartitionedCallҐ dense_53/StatefulPartitionedCallХ
 dense_52/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_52_396905dense_52_396907*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_52_layer_call_and_return_conditional_losses_3968942"
 dense_52/StatefulPartitionedCallЈ
 dense_53/StatefulPartitionedCallStatefulPartitionedCall)dense_52/StatefulPartitionedCall:output:0dense_53_396931dense_53_396933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *M
fHRF
D__inference_dense_53_layer_call_and_return_conditional_losses_3969202"
 dense_53/StatefulPartitionedCall√
IdentityIdentity)dense_52/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity«

Identity_1Identity)dense_53/StatefulPartitionedCall:output:0!^dense_52/StatefulPartitionedCall!^dense_53/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*6
_input_shapes%
#:€€€€€€€€€::::2D
 dense_52/StatefulPartitionedCall dense_52/StatefulPartitionedCall2D
 dense_53/StatefulPartitionedCall dense_53/StatefulPartitionedCall:P L
'
_output_shapes
:€€€€€€€€€
!
_user_specified_name	input_1"ЄL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*п
serving_defaultџ
;
input_10
serving_default_input_1:0€€€€€€€€€<
q_values0
StatefulPartitionedCall:0€€€€€€€€€B
value_estimate0
StatefulPartitionedCall:1€€€€€€€€€tensorflow/serving/predict:Б>
ќ
	layer

layer2
trainable_variables
	variables
regularization_losses
	keras_api

signatures
#__call__
$_default_save_signature
*%&call_and_return_all_conditional_losses"ы
_tf_keras_modelб{"class_name": "MyModel", "name": "my_model_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "MyModel"}}
о

kernel
	bias

trainable_variables
	variables
regularization_losses
	keras_api
&__call__
*'&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense_52", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_52", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 4]}}
о

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
(__call__
*)&call_and_return_all_conditional_losses"…
_tf_keras_layerѓ{"class_name": "Dense", "name": "dense_53", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_53", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [1, 2]}}
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
 "
trackable_list_wrapper
 
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
#__call__
$_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
,
*serving_default"
signature_map
-:+2my_model_26/dense_52/kernel
':%2my_model_26/dense_52/bias
.
0
	1"
trackable_list_wrapper
.
0
	1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠

trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
layer_metrics
layer_regularization_losses
metrics
&__call__
*'&call_and_return_all_conditional_losses
&'"call_and_return_conditional_losses"
_generic_user_object
-:+2my_model_26/dense_53/kernel
':%2my_model_26/dense_53/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
≠
trainable_variables

layers
	variables
non_trainable_variables
regularization_losses
 layer_metrics
!layer_regularization_losses
"metrics
(__call__
*)&call_and_return_all_conditional_losses
&)"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
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
 "
trackable_list_wrapper
 "
trackable_list_wrapper
ш2х
,__inference_my_model_26_layer_call_fn_396954ƒ
Ч≤У
FullArgSpec
argsЪ
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
я2№
!__inference__wrapped_model_396880ґ
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
У2Р
G__inference_my_model_26_layer_call_and_return_conditional_losses_396938ƒ
Ч≤У
FullArgSpec
argsЪ
jself
jx_in
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *&Ґ#
!К
input_1€€€€€€€€€
”2–
)__inference_dense_52_layer_call_fn_396990Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_52_layer_call_and_return_conditional_losses_396981Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
”2–
)__inference_dense_53_layer_call_fn_397009Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
о2л
D__inference_dense_53_layer_call_and_return_conditional_losses_397000Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
3B1
$__inference_signature_wrapper_396971input_1ѕ
!__inference__wrapped_model_396880©	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€§
D__inference_dense_52_layer_call_and_return_conditional_losses_396981\	/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_52_layer_call_fn_396990O	/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€§
D__inference_dense_53_layer_call_and_return_conditional_losses_397000\/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "%Ґ"
К
0€€€€€€€€€
Ъ |
)__inference_dense_53_layer_call_fn_397009O/Ґ,
%Ґ"
 К
inputs€€€€€€€€€
™ "К€€€€€€€€€Г
G__inference_my_model_26_layer_call_and_return_conditional_losses_396938Ј	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "}Ґz
s™p
0
q_values$К!

0/q_values€€€€€€€€€
<
value_estimate*К'
0/value_estimate€€€€€€€€€
Ъ Џ
,__inference_my_model_26_layer_call_fn_396954©	0Ґ-
&Ґ#
!К
input_1€€€€€€€€€
™ "o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€Ё
$__inference_signature_wrapper_396971і	;Ґ8
Ґ 
1™.
,
input_1!К
input_1€€€€€€€€€"o™l
.
q_values"К
q_values€€€€€€€€€
:
value_estimate(К%
value_estimate€€€€€€€€€