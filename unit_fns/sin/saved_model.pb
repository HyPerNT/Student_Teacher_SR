л■
Ї─
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
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
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8д╠
Д
Adam/dense_1135/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1135/bias/v
}
*Adam/dense_1135/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1135/bias/v*
_output_shapes
:*
dtype0
Н
Adam/dense_1135/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameAdam/dense_1135/kernel/v
Ж
,Adam/dense_1135/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1135/kernel/v*
_output_shapes
:	А*
dtype0
Е
Adam/dense_1134/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/dense_1134/bias/v
~
*Adam/dense_1134/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1134/bias/v*
_output_shapes	
:А*
dtype0
О
Adam/dense_1134/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*)
shared_nameAdam/dense_1134/kernel/v
З
,Adam/dense_1134/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1134/kernel/v* 
_output_shapes
:
АА*
dtype0
Е
Adam/dense_1133/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/dense_1133/bias/v
~
*Adam/dense_1133/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1133/bias/v*
_output_shapes	
:А*
dtype0
Н
Adam/dense_1133/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameAdam/dense_1133/kernel/v
Ж
,Adam/dense_1133/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1133/kernel/v*
_output_shapes
:	А*
dtype0
Д
Adam/dense_1135/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/dense_1135/bias/m
}
*Adam/dense_1135/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1135/bias/m*
_output_shapes
:*
dtype0
Н
Adam/dense_1135/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameAdam/dense_1135/kernel/m
Ж
,Adam/dense_1135/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1135/kernel/m*
_output_shapes
:	А*
dtype0
Е
Adam/dense_1134/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/dense_1134/bias/m
~
*Adam/dense_1134/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1134/bias/m*
_output_shapes	
:А*
dtype0
О
Adam/dense_1134/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*)
shared_nameAdam/dense_1134/kernel/m
З
,Adam/dense_1134/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1134/kernel/m* 
_output_shapes
:
АА*
dtype0
Е
Adam/dense_1133/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:А*'
shared_nameAdam/dense_1133/bias/m
~
*Adam/dense_1133/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1133/bias/m*
_output_shapes	
:А*
dtype0
Н
Adam/dense_1133/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*)
shared_nameAdam/dense_1133/kernel/m
Ж
,Adam/dense_1133/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1133/kernel/m*
_output_shapes
:	А*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
v
dense_1135/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namedense_1135/bias
o
#dense_1135/bias/Read/ReadVariableOpReadVariableOpdense_1135/bias*
_output_shapes
:*
dtype0

dense_1135/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*"
shared_namedense_1135/kernel
x
%dense_1135/kernel/Read/ReadVariableOpReadVariableOpdense_1135/kernel*
_output_shapes
:	А*
dtype0
w
dense_1134/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_namedense_1134/bias
p
#dense_1134/bias/Read/ReadVariableOpReadVariableOpdense_1134/bias*
_output_shapes	
:А*
dtype0
А
dense_1134/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
АА*"
shared_namedense_1134/kernel
y
%dense_1134/kernel/Read/ReadVariableOpReadVariableOpdense_1134/kernel* 
_output_shapes
:
АА*
dtype0
w
dense_1133/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:А* 
shared_namedense_1133/bias
p
#dense_1133/bias/Read/ReadVariableOpReadVariableOpdense_1133/bias*
_output_shapes	
:А*
dtype0

dense_1133/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А*"
shared_namedense_1133/kernel
x
%dense_1133/kernel/Read/ReadVariableOpReadVariableOpdense_1133/kernel*
_output_shapes
:	А*
dtype0
|
serving_default_input_206Placeholder*'
_output_shapes
:         *
dtype0*
shape:         
о
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_206dense_1133/kerneldense_1133/biasdense_1134/kerneldense_1134/biasdense_1135/kerneldense_1135/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В */
f*R(
&__inference_signature_wrapper_21546238

NoOpNoOp
°5
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*│5
valueй5Bж5 BЯ5
█
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
ж
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
е
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator* 
ж
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias*
е
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_random_generator* 
ж
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias*
.
0
1
$2
%3
34
45*
.
0
1
$2
%3
34
45*
* 
░
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
:trace_0
;trace_1
<trace_2
=trace_3* 
6
>trace_0
?trace_1
@trace_2
Atrace_3* 
* 
░
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_ratemtmu$mv%mw3mx4myvzv{$v|%v}3v~4v*

Gserving_default* 

0
1*

0
1*
* 
У
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Mtrace_0* 

Ntrace_0* 
a[
VARIABLE_VALUEdense_1133/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1133/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

Ttrace_0
Utrace_1* 

Vtrace_0
Wtrace_1* 
* 

$0
%1*

$0
%1*
* 
У
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

]trace_0* 

^trace_0* 
a[
VARIABLE_VALUEdense_1134/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1134/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
С
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

dtrace_0
etrace_1* 

ftrace_0
gtrace_1* 
* 

30
41*

30
41*
* 
У
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses*

mtrace_0* 

ntrace_0* 
a[
VARIABLE_VALUEdense_1135/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1135/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
'
0
1
2
3
4*

o0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
p	variables
q	keras_api
	rtotal
	scount*

r0
s1*

p	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1133/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1133/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1134/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1134/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1135/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1135/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1133/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1133/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1134/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1134/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Д~
VARIABLE_VALUEAdam/dense_1135/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
Аz
VARIABLE_VALUEAdam/dense_1135/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Х

StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_1133/kernel/Read/ReadVariableOp#dense_1133/bias/Read/ReadVariableOp%dense_1134/kernel/Read/ReadVariableOp#dense_1134/bias/Read/ReadVariableOp%dense_1135/kernel/Read/ReadVariableOp#dense_1135/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/dense_1133/kernel/m/Read/ReadVariableOp*Adam/dense_1133/bias/m/Read/ReadVariableOp,Adam/dense_1134/kernel/m/Read/ReadVariableOp*Adam/dense_1134/bias/m/Read/ReadVariableOp,Adam/dense_1135/kernel/m/Read/ReadVariableOp*Adam/dense_1135/bias/m/Read/ReadVariableOp,Adam/dense_1133/kernel/v/Read/ReadVariableOp*Adam/dense_1133/bias/v/Read/ReadVariableOp,Adam/dense_1134/kernel/v/Read/ReadVariableOp*Adam/dense_1134/bias/v/Read/ReadVariableOp,Adam/dense_1135/kernel/v/Read/ReadVariableOp*Adam/dense_1135/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
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
GPU 2J 8В **
f%R#
!__inference__traced_save_21546549
Ь
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1133/kerneldense_1133/biasdense_1134/kerneldense_1134/biasdense_1135/kerneldense_1135/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_1133/kernel/mAdam/dense_1133/bias/mAdam/dense_1134/kernel/mAdam/dense_1134/bias/mAdam/dense_1135/kernel/mAdam/dense_1135/bias/mAdam/dense_1133/kernel/vAdam/dense_1133/bias/vAdam/dense_1134/kernel/vAdam/dense_1134/bias/vAdam/dense_1135/kernel/vAdam/dense_1135/bias/v*%
Tin
2*
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
GPU 2J 8В *-
f(R&
$__inference__traced_restore_21546634╔╨
р
g
I__inference_dropout_929_layer_call_and_return_conditional_losses_21545989

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
л
J
.__inference_dropout_928_layer_call_fn_21546363

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21545965a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
 	
h
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546053

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╒
И
&__inference_signature_wrapper_21546238
	input_206
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCallя
StatefulPartitionedCallStatefulPartitionedCall	input_206unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *,
f'R%
#__inference__wrapped_model_21545936o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
■9
р

!__inference__traced_save_21546549
file_prefix0
,savev2_dense_1133_kernel_read_readvariableop.
*savev2_dense_1133_bias_read_readvariableop0
,savev2_dense_1134_kernel_read_readvariableop.
*savev2_dense_1134_bias_read_readvariableop0
,savev2_dense_1135_kernel_read_readvariableop.
*savev2_dense_1135_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_dense_1133_kernel_m_read_readvariableop5
1savev2_adam_dense_1133_bias_m_read_readvariableop7
3savev2_adam_dense_1134_kernel_m_read_readvariableop5
1savev2_adam_dense_1134_bias_m_read_readvariableop7
3savev2_adam_dense_1135_kernel_m_read_readvariableop5
1savev2_adam_dense_1135_bias_m_read_readvariableop7
3savev2_adam_dense_1133_kernel_v_read_readvariableop5
1savev2_adam_dense_1133_bias_v_read_readvariableop7
3savev2_adam_dense_1134_kernel_v_read_readvariableop5
1savev2_adam_dense_1134_bias_v_read_readvariableop7
3savev2_adam_dense_1135_kernel_v_read_readvariableop5
1savev2_adam_dense_1135_bias_v_read_readvariableop
savev2_const

identity_1ИвMergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/partБ
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Й
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▓
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHб
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B ╫

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_1133_kernel_read_readvariableop*savev2_dense_1133_bias_read_readvariableop,savev2_dense_1134_kernel_read_readvariableop*savev2_dense_1134_bias_read_readvariableop,savev2_dense_1135_kernel_read_readvariableop*savev2_dense_1135_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_dense_1133_kernel_m_read_readvariableop1savev2_adam_dense_1133_bias_m_read_readvariableop3savev2_adam_dense_1134_kernel_m_read_readvariableop1savev2_adam_dense_1134_bias_m_read_readvariableop3savev2_adam_dense_1135_kernel_m_read_readvariableop1savev2_adam_dense_1135_bias_m_read_readvariableop3savev2_adam_dense_1133_kernel_v_read_readvariableop1savev2_adam_dense_1133_bias_v_read_readvariableop3savev2_adam_dense_1134_kernel_v_read_readvariableop1savev2_adam_dense_1134_bias_v_read_readvariableop3savev2_adam_dense_1135_kernel_v_read_readvariableop1savev2_adam_dense_1135_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Л
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*╔
_input_shapes╖
┤: :	А:А:
АА:А:	А:: : : : : : : :	А:А:
АА:А:	А::	А:А:
АА:А:	А:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::%!

_output_shapes
:	А:!

_output_shapes	
:А:&"
 
_output_shapes
:
АА:!

_output_shapes	
:А:%!

_output_shapes
:	А: 

_output_shapes
::

_output_shapes
: 
з

√
H__inference_dense_1133_layer_call_and_return_conditional_losses_21546358

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Ьf
░
$__inference__traced_restore_21546634
file_prefix5
"assignvariableop_dense_1133_kernel:	А1
"assignvariableop_1_dense_1133_bias:	А8
$assignvariableop_2_dense_1134_kernel:
АА1
"assignvariableop_3_dense_1134_bias:	А7
$assignvariableop_4_dense_1135_kernel:	А0
"assignvariableop_5_dense_1135_bias:&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: ?
,assignvariableop_13_adam_dense_1133_kernel_m:	А9
*assignvariableop_14_adam_dense_1133_bias_m:	А@
,assignvariableop_15_adam_dense_1134_kernel_m:
АА9
*assignvariableop_16_adam_dense_1134_bias_m:	А?
,assignvariableop_17_adam_dense_1135_kernel_m:	А8
*assignvariableop_18_adam_dense_1135_bias_m:?
,assignvariableop_19_adam_dense_1133_kernel_v:	А9
*assignvariableop_20_adam_dense_1133_bias_v:	А@
,assignvariableop_21_adam_dense_1134_kernel_v:
АА9
*assignvariableop_22_adam_dense_1134_bias_v:	А?
,assignvariableop_23_adam_dense_1135_kernel_v:	А8
*assignvariableop_24_adam_dense_1135_bias_v:
identity_26ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_3вAssignVariableOp_4вAssignVariableOp_5вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9М
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*▓
valueиBеB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHд
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B а
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOpAssignVariableOp"assignvariableop_dense_1133_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_1133_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1134_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1134_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:У
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_1135_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:С
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_1135_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:Л
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Н
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ч
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_dense_1133_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_1133_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1134_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1134_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_1135_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_1135_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_1133_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_1133_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1134_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1134_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Э
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_1135_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ы
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_1135_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ї
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: т
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
╧	
·
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
┐*
Щ
#__inference__wrapped_model_21545936
	input_206S
@sin_2_layers_512_nodes_dense_1133_matmul_readvariableop_resource:	АP
Asin_2_layers_512_nodes_dense_1133_biasadd_readvariableop_resource:	АT
@sin_2_layers_512_nodes_dense_1134_matmul_readvariableop_resource:
ААP
Asin_2_layers_512_nodes_dense_1134_biasadd_readvariableop_resource:	АS
@sin_2_layers_512_nodes_dense_1135_matmul_readvariableop_resource:	АO
Asin_2_layers_512_nodes_dense_1135_biasadd_readvariableop_resource:
identityИв8sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOpв7sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOpв8sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOpв7sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOpв8sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOpв7sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOp╣
7sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOpReadVariableOp@sin_2_layers_512_nodes_dense_1133_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0▒
(sin_2-layers_512-nodes/dense_1133/MatMulMatMul	input_206?sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╖
8sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOpReadVariableOpAsin_2_layers_512_nodes_dense_1133_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▌
)sin_2-layers_512-nodes/dense_1133/BiasAddBiasAdd2sin_2-layers_512-nodes/dense_1133/MatMul:product:0@sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АХ
&sin_2-layers_512-nodes/dense_1133/ReluRelu2sin_2-layers_512-nodes/dense_1133/BiasAdd:output:0*
T0*(
_output_shapes
:         Аа
+sin_2-layers_512-nodes/dropout_928/IdentityIdentity4sin_2-layers_512-nodes/dense_1133/Relu:activations:0*
T0*(
_output_shapes
:         А║
7sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOpReadVariableOp@sin_2_layers_512_nodes_dense_1134_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0▄
(sin_2-layers_512-nodes/dense_1134/MatMulMatMul4sin_2-layers_512-nodes/dropout_928/Identity:output:0?sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         А╖
8sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOpReadVariableOpAsin_2_layers_512_nodes_dense_1134_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0▌
)sin_2-layers_512-nodes/dense_1134/BiasAddBiasAdd2sin_2-layers_512-nodes/dense_1134/MatMul:product:0@sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АХ
&sin_2-layers_512-nodes/dense_1134/ReluRelu2sin_2-layers_512-nodes/dense_1134/BiasAdd:output:0*
T0*(
_output_shapes
:         Аа
+sin_2-layers_512-nodes/dropout_929/IdentityIdentity4sin_2-layers_512-nodes/dense_1134/Relu:activations:0*
T0*(
_output_shapes
:         А╣
7sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOpReadVariableOp@sin_2_layers_512_nodes_dense_1135_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0█
(sin_2-layers_512-nodes/dense_1135/MatMulMatMul4sin_2-layers_512-nodes/dropout_929/Identity:output:0?sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         ╢
8sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOpReadVariableOpAsin_2_layers_512_nodes_dense_1135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0▄
)sin_2-layers_512-nodes/dense_1135/BiasAddBiasAdd2sin_2-layers_512-nodes/dense_1135/MatMul:product:0@sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Б
IdentityIdentity2sin_2-layers_512-nodes/dense_1135/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         е
NoOpNoOp9^sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOp8^sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOp9^sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOp8^sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOp9^sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOp8^sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2t
8sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOp8sin_2-layers_512-nodes/dense_1133/BiasAdd/ReadVariableOp2r
7sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOp7sin_2-layers_512-nodes/dense_1133/MatMul/ReadVariableOp2t
8sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOp8sin_2-layers_512-nodes/dense_1134/BiasAdd/ReadVariableOp2r
7sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOp7sin_2-layers_512-nodes/dense_1134/MatMul/ReadVariableOp2t
8sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOp8sin_2-layers_512-nodes/dense_1135/BiasAdd/ReadVariableOp2r
7sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOp7sin_2-layers_512-nodes/dense_1135/MatMul/ReadVariableOp:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
═
Ы
-__inference_dense_1135_layer_call_fn_21546441

inputs
unknown:	А
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
 	
h
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546385

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Т.
│
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546338

inputs<
)dense_1133_matmul_readvariableop_resource:	А9
*dense_1133_biasadd_readvariableop_resource:	А=
)dense_1134_matmul_readvariableop_resource:
АА9
*dense_1134_biasadd_readvariableop_resource:	А<
)dense_1135_matmul_readvariableop_resource:	А8
*dense_1135_biasadd_readvariableop_resource:
identityИв!dense_1133/BiasAdd/ReadVariableOpв dense_1133/MatMul/ReadVariableOpв!dense_1134/BiasAdd/ReadVariableOpв dense_1134/MatMul/ReadVariableOpв!dense_1135/BiasAdd/ReadVariableOpв dense_1135/MatMul/ReadVariableOpЛ
 dense_1133/MatMul/ReadVariableOpReadVariableOp)dense_1133_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0А
dense_1133/MatMulMatMulinputs(dense_1133/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЙ
!dense_1133/BiasAdd/ReadVariableOpReadVariableOp*dense_1133_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_1133/BiasAddBiasAdddense_1133/MatMul:product:0)dense_1133/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
dense_1133/ReluReludense_1133/BiasAdd:output:0*
T0*(
_output_shapes
:         А^
dropout_928/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?Ф
dropout_928/dropout/MulMuldense_1133/Relu:activations:0"dropout_928/dropout/Const:output:0*
T0*(
_output_shapes
:         Аf
dropout_928/dropout/ShapeShapedense_1133/Relu:activations:0*
T0*
_output_shapes
:е
0dropout_928/dropout/random_uniform/RandomUniformRandomUniform"dropout_928/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0g
"dropout_928/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<╦
 dropout_928/dropout/GreaterEqualGreaterEqual9dropout_928/dropout/random_uniform/RandomUniform:output:0+dropout_928/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АИ
dropout_928/dropout/CastCast$dropout_928/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АО
dropout_928/dropout/Mul_1Muldropout_928/dropout/Mul:z:0dropout_928/dropout/Cast:y:0*
T0*(
_output_shapes
:         АМ
 dense_1134/MatMul/ReadVariableOpReadVariableOp)dense_1134_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ч
dense_1134/MatMulMatMuldropout_928/dropout/Mul_1:z:0(dense_1134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЙ
!dense_1134/BiasAdd/ReadVariableOpReadVariableOp*dense_1134_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_1134/BiasAddBiasAdddense_1134/MatMul:product:0)dense_1134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
dense_1134/ReluReludense_1134/BiasAdd:output:0*
T0*(
_output_shapes
:         А^
dropout_929/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?Ф
dropout_929/dropout/MulMuldense_1134/Relu:activations:0"dropout_929/dropout/Const:output:0*
T0*(
_output_shapes
:         Аf
dropout_929/dropout/ShapeShapedense_1134/Relu:activations:0*
T0*
_output_shapes
:е
0dropout_929/dropout/random_uniform/RandomUniformRandomUniform"dropout_929/dropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0g
"dropout_929/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<╦
 dropout_929/dropout/GreaterEqualGreaterEqual9dropout_929/dropout/random_uniform/RandomUniform:output:0+dropout_929/dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         АИ
dropout_929/dropout/CastCast$dropout_929/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         АО
dropout_929/dropout/Mul_1Muldropout_929/dropout/Mul:z:0dropout_929/dropout/Cast:y:0*
T0*(
_output_shapes
:         АЛ
 dense_1135/MatMul/ReadVariableOpReadVariableOp)dense_1135_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_1135/MatMulMatMuldropout_929/dropout/Mul_1:z:0(dense_1135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!dense_1135/BiasAdd/ReadVariableOpReadVariableOp*dense_1135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
dense_1135/BiasAddBiasAdddense_1135/MatMul:product:0)dense_1135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         j
IdentityIdentitydense_1135/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp"^dense_1133/BiasAdd/ReadVariableOp!^dense_1133/MatMul/ReadVariableOp"^dense_1134/BiasAdd/ReadVariableOp!^dense_1134/MatMul/ReadVariableOp"^dense_1135/BiasAdd/ReadVariableOp!^dense_1135/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2F
!dense_1133/BiasAdd/ReadVariableOp!dense_1133/BiasAdd/ReadVariableOp2D
 dense_1133/MatMul/ReadVariableOp dense_1133/MatMul/ReadVariableOp2F
!dense_1134/BiasAdd/ReadVariableOp!dense_1134/BiasAdd/ReadVariableOp2D
 dense_1134/MatMul/ReadVariableOp dense_1134/MatMul/ReadVariableOp2F
!dense_1135/BiasAdd/ReadVariableOp!dense_1135/BiasAdd/ReadVariableOp2D
 dense_1135/MatMul/ReadVariableOp dense_1135/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р	
Ш
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546255

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 	
h
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546432

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
 	
h
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546086

inputs
identityИR
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *¤JБ?e
dropout/MulMulinputsdropout/Const:output:0*
T0*(
_output_shapes
:         АC
dropout/ShapeShapeinputs*
T0*
_output_shapes
:Н
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*(
_output_shapes
:         А*
dtype0[
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *
╫#<з
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*(
_output_shapes
:         Аp
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*(
_output_shapes
:         Аj
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*(
_output_shapes
:         АZ
IdentityIdentitydropout/Mul_1:z:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
g
I__inference_dropout_928_layer_call_and_return_conditional_losses_21545965

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
¤
g
.__inference_dropout_929_layer_call_fn_21546415

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546053p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Щ	
Ы
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546171
	input_206
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCall	input_206unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546139o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
╝
╞
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546008

inputs&
dense_1133_21545955:	А"
dense_1133_21545957:	А'
dense_1134_21545979:
АА"
dense_1134_21545981:	А&
dense_1135_21546002:	А!
dense_1135_21546004:
identityИв"dense_1133/StatefulPartitionedCallв"dense_1134/StatefulPartitionedCallв"dense_1135/StatefulPartitionedCall 
"dense_1133/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1133_21545955dense_1133_21545957*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954ц
dropout_928/PartitionedCallPartitionedCall+dense_1133/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21545965Э
"dense_1134/StatefulPartitionedCallStatefulPartitionedCall$dropout_928/PartitionedCall:output:0dense_1134_21545979dense_1134_21545981*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978ц
dropout_929/PartitionedCallPartitionedCall+dense_1134/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21545989Ь
"dense_1135/StatefulPartitionedCallStatefulPartitionedCall$dropout_929/PartitionedCall:output:0dense_1135_21546002dense_1135_21546004*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001z
IdentityIdentity+dense_1135/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╡
NoOpNoOp#^dense_1133/StatefulPartitionedCall#^dense_1134/StatefulPartitionedCall#^dense_1135/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2H
"dense_1133/StatefulPartitionedCall"dense_1133/StatefulPartitionedCall2H
"dense_1134/StatefulPartitionedCall"dense_1134/StatefulPartitionedCall2H
"dense_1135/StatefulPartitionedCall"dense_1135/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧	
·
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546451

inputs1
matmul_readvariableop_resource:	А-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
л
J
.__inference_dropout_929_layer_call_fn_21546410

inputs
identity╡
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21545989a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:         А"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
¤
g
.__inference_dropout_928_layer_call_fn_21546368

inputs
identityИвStatefulPartitionedCall┼
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546086p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Р	
Ш
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546272

inputs
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCallЭ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546139o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┼
╔
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546192
	input_206&
dense_1133_21546174:	А"
dense_1133_21546176:	А'
dense_1134_21546180:
АА"
dense_1134_21546182:	А&
dense_1135_21546186:	А!
dense_1135_21546188:
identityИв"dense_1133/StatefulPartitionedCallв"dense_1134/StatefulPartitionedCallв"dense_1135/StatefulPartitionedCallВ
"dense_1133/StatefulPartitionedCallStatefulPartitionedCall	input_206dense_1133_21546174dense_1133_21546176*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954ц
dropout_928/PartitionedCallPartitionedCall+dense_1133/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21545965Э
"dense_1134/StatefulPartitionedCallStatefulPartitionedCall$dropout_928/PartitionedCall:output:0dense_1134_21546180dense_1134_21546182*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978ц
dropout_929/PartitionedCallPartitionedCall+dense_1134/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21545989Ь
"dense_1135/StatefulPartitionedCallStatefulPartitionedCall$dropout_929/PartitionedCall:output:0dense_1135_21546186dense_1135_21546188*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001z
IdentityIdentity+dense_1135/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         ╡
NoOpNoOp#^dense_1133/StatefulPartitionedCall#^dense_1134/StatefulPartitionedCall#^dense_1135/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2H
"dense_1133/StatefulPartitionedCall"dense_1133/StatefulPartitionedCall2H
"dense_1134/StatefulPartitionedCall"dense_1134/StatefulPartitionedCall2H
"dense_1135/StatefulPartitionedCall"dense_1135/StatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
╤
Э
-__inference_dense_1134_layer_call_fn_21546394

inputs
unknown:
АА
	unknown_0:	А
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
з

√
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954

inputs1
matmul_readvariableop_resource:	А.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╞
│
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546298

inputs<
)dense_1133_matmul_readvariableop_resource:	А9
*dense_1133_biasadd_readvariableop_resource:	А=
)dense_1134_matmul_readvariableop_resource:
АА9
*dense_1134_biasadd_readvariableop_resource:	А<
)dense_1135_matmul_readvariableop_resource:	А8
*dense_1135_biasadd_readvariableop_resource:
identityИв!dense_1133/BiasAdd/ReadVariableOpв dense_1133/MatMul/ReadVariableOpв!dense_1134/BiasAdd/ReadVariableOpв dense_1134/MatMul/ReadVariableOpв!dense_1135/BiasAdd/ReadVariableOpв dense_1135/MatMul/ReadVariableOpЛ
 dense_1133/MatMul/ReadVariableOpReadVariableOp)dense_1133_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0А
dense_1133/MatMulMatMulinputs(dense_1133/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЙ
!dense_1133/BiasAdd/ReadVariableOpReadVariableOp*dense_1133_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_1133/BiasAddBiasAdddense_1133/MatMul:product:0)dense_1133/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
dense_1133/ReluReludense_1133/BiasAdd:output:0*
T0*(
_output_shapes
:         Аr
dropout_928/IdentityIdentitydense_1133/Relu:activations:0*
T0*(
_output_shapes
:         АМ
 dense_1134/MatMul/ReadVariableOpReadVariableOp)dense_1134_matmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0Ч
dense_1134/MatMulMatMuldropout_928/Identity:output:0(dense_1134/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АЙ
!dense_1134/BiasAdd/ReadVariableOpReadVariableOp*dense_1134_biasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0Ш
dense_1134/BiasAddBiasAdddense_1134/MatMul:product:0)dense_1134/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аg
dense_1134/ReluReludense_1134/BiasAdd:output:0*
T0*(
_output_shapes
:         Аr
dropout_929/IdentityIdentitydense_1134/Relu:activations:0*
T0*(
_output_shapes
:         АЛ
 dense_1135/MatMul/ReadVariableOpReadVariableOp)dense_1135_matmul_readvariableop_resource*
_output_shapes
:	А*
dtype0Ц
dense_1135/MatMulMatMuldropout_929/Identity:output:0(dense_1135/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         И
!dense_1135/BiasAdd/ReadVariableOpReadVariableOp*dense_1135_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ч
dense_1135/BiasAddBiasAdddense_1135/MatMul:product:0)dense_1135/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         j
IdentityIdentitydense_1135/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         Ы
NoOpNoOp"^dense_1133/BiasAdd/ReadVariableOp!^dense_1133/MatMul/ReadVariableOp"^dense_1134/BiasAdd/ReadVariableOp!^dense_1134/MatMul/ReadVariableOp"^dense_1135/BiasAdd/ReadVariableOp!^dense_1135/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2F
!dense_1133/BiasAdd/ReadVariableOp!dense_1133/BiasAdd/ReadVariableOp2D
 dense_1133/MatMul/ReadVariableOp dense_1133/MatMul/ReadVariableOp2F
!dense_1134/BiasAdd/ReadVariableOp!dense_1134/BiasAdd/ReadVariableOp2D
 dense_1134/MatMul/ReadVariableOp dense_1134/MatMul/ReadVariableOp2F
!dense_1135/BiasAdd/ReadVariableOp!dense_1135/BiasAdd/ReadVariableOp2D
 dense_1135/MatMul/ReadVariableOp dense_1135/MatMul/ReadVariableOp:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┬
Т
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546139

inputs&
dense_1133_21546121:	А"
dense_1133_21546123:	А'
dense_1134_21546127:
АА"
dense_1134_21546129:	А&
dense_1135_21546133:	А!
dense_1135_21546135:
identityИв"dense_1133/StatefulPartitionedCallв"dense_1134/StatefulPartitionedCallв"dense_1135/StatefulPartitionedCallв#dropout_928/StatefulPartitionedCallв#dropout_929/StatefulPartitionedCall 
"dense_1133/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1133_21546121dense_1133_21546123*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954Ў
#dropout_928/StatefulPartitionedCallStatefulPartitionedCall+dense_1133/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546086е
"dense_1134/StatefulPartitionedCallStatefulPartitionedCall,dropout_928/StatefulPartitionedCall:output:0dense_1134_21546127dense_1134_21546129*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978Ь
#dropout_929/StatefulPartitionedCallStatefulPartitionedCall+dense_1134/StatefulPartitionedCall:output:0$^dropout_928/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546053д
"dense_1135/StatefulPartitionedCallStatefulPartitionedCall,dropout_929/StatefulPartitionedCall:output:0dense_1135_21546133dense_1135_21546135*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001z
IdentityIdentity+dense_1135/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp#^dense_1133/StatefulPartitionedCall#^dense_1134/StatefulPartitionedCall#^dense_1135/StatefulPartitionedCall$^dropout_928/StatefulPartitionedCall$^dropout_929/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2H
"dense_1133/StatefulPartitionedCall"dense_1133/StatefulPartitionedCall2H
"dense_1134/StatefulPartitionedCall"dense_1134/StatefulPartitionedCall2H
"dense_1135/StatefulPartitionedCall"dense_1135/StatefulPartitionedCall2J
#dropout_928/StatefulPartitionedCall#dropout_928/StatefulPartitionedCall2J
#dropout_929/StatefulPartitionedCall#dropout_929/StatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╬
Ь
-__inference_dense_1133_layer_call_fn_21546347

inputs
unknown:	А
	unknown_0:	А
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:         А`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
л

№
H__inference_dense_1134_layer_call_and_return_conditional_losses_21546405

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
╦
Х
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546213
	input_206&
dense_1133_21546195:	А"
dense_1133_21546197:	А'
dense_1134_21546201:
АА"
dense_1134_21546203:	А&
dense_1135_21546207:	А!
dense_1135_21546209:
identityИв"dense_1133/StatefulPartitionedCallв"dense_1134/StatefulPartitionedCallв"dense_1135/StatefulPartitionedCallв#dropout_928/StatefulPartitionedCallв#dropout_929/StatefulPartitionedCallВ
"dense_1133/StatefulPartitionedCallStatefulPartitionedCall	input_206dense_1133_21546195dense_1133_21546197*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1133_layer_call_and_return_conditional_losses_21545954Ў
#dropout_928/StatefulPartitionedCallStatefulPartitionedCall+dense_1133/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546086е
"dense_1134/StatefulPartitionedCallStatefulPartitionedCall,dropout_928/StatefulPartitionedCall:output:0dense_1134_21546201dense_1134_21546203*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978Ь
#dropout_929/StatefulPartitionedCallStatefulPartitionedCall+dense_1134/StatefulPartitionedCall:output:0$^dropout_928/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:         А* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8В *R
fMRK
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546053д
"dense_1135/StatefulPartitionedCallStatefulPartitionedCall,dropout_929/StatefulPartitionedCall:output:0dense_1135_21546207dense_1135_21546209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8В *Q
fLRJ
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546001z
IdentityIdentity+dense_1135/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         Б
NoOpNoOp#^dense_1133/StatefulPartitionedCall#^dense_1134/StatefulPartitionedCall#^dense_1135/StatefulPartitionedCall$^dropout_928/StatefulPartitionedCall$^dropout_929/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 2H
"dense_1133/StatefulPartitionedCall"dense_1133/StatefulPartitionedCall2H
"dense_1134/StatefulPartitionedCall"dense_1134/StatefulPartitionedCall2H
"dense_1135/StatefulPartitionedCall"dense_1135/StatefulPartitionedCall2J
#dropout_928/StatefulPartitionedCall#dropout_928/StatefulPartitionedCall2J
#dropout_929/StatefulPartitionedCall#dropout_929/StatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
р
g
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546420

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
р
g
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546373

inputs

identity_1O
IdentityIdentityinputs*
T0*(
_output_shapes
:         А\

Identity_1IdentityIdentity:output:0*
T0*(
_output_shapes
:         А"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:         А:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs
Щ	
Ы
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546023
	input_206
unknown:	А
	unknown_0:	А
	unknown_1:
АА
	unknown_2:	А
	unknown_3:	А
	unknown_4:
identityИвStatefulPartitionedCallа
StatefulPartitionedCallStatefulPartitionedCall	input_206unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8В *]
fXRV
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546008o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         `
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:         
#
_user_specified_name	input_206
л

№
H__inference_dense_1134_layer_call_and_return_conditional_losses_21545978

inputs2
matmul_readvariableop_resource:
АА.
biasadd_readvariableop_resource:	А
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
АА*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         Аs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:А*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         АQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:         Аb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:         Аw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:         А: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:         А
 
_user_specified_nameinputs"╡	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*▒
serving_defaultЭ
?
	input_2062
serving_default_input_206:0         >

dense_11350
StatefulPartitionedCall:0         tensorflow/serving/predict:еб
ї
layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
layer_with_weights-2
layer-4
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
╗
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
╝
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
╗
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses

$kernel
%bias"
_tf_keras_layer
╝
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses
,_random_generator"
_tf_keras_layer
╗
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses

3kernel
4bias"
_tf_keras_layer
J
0
1
$2
%3
34
45"
trackable_list_wrapper
J
0
1
$2
%3
34
45"
trackable_list_wrapper
 "
trackable_list_wrapper
╩
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Щ
:trace_0
;trace_1
<trace_2
=trace_32о
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546023
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546255
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546272
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546171┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z:trace_0z;trace_1z<trace_2z=trace_3
Е
>trace_0
?trace_1
@trace_2
Atrace_32Ъ
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546298
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546338
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546192
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546213┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z>trace_0z?trace_1z@trace_2zAtrace_3
╨B═
#__inference__wrapped_model_21545936	input_206"Ш
С▓Н
FullArgSpec
argsЪ 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
┐
Biter

Cbeta_1

Dbeta_2
	Edecay
Flearning_ratemtmu$mv%mw3mx4myvzv{$v|%v}3v~4v"
	optimizer
,
Gserving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Hnon_trainable_variables

Ilayers
Jmetrics
Klayer_regularization_losses
Llayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ё
Mtrace_02╘
-__inference_dense_1133_layer_call_fn_21546347в
Щ▓Х
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
annotationsк *
 zMtrace_0
М
Ntrace_02я
H__inference_dense_1133_layer_call_and_return_conditional_losses_21546358в
Щ▓Х
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
annotationsк *
 zNtrace_0
$:"	А2dense_1133/kernel
:А2dense_1133/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
Onon_trainable_variables

Players
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
═
Ttrace_0
Utrace_12Ц
.__inference_dropout_928_layer_call_fn_21546363
.__inference_dropout_928_layer_call_fn_21546368│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zTtrace_0zUtrace_1
Г
Vtrace_0
Wtrace_12╠
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546373
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546385│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zVtrace_0zWtrace_1
"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
н
Xnon_trainable_variables

Ylayers
Zmetrics
[layer_regularization_losses
\layer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
ё
]trace_02╘
-__inference_dense_1134_layer_call_fn_21546394в
Щ▓Х
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
annotationsк *
 z]trace_0
М
^trace_02я
H__inference_dense_1134_layer_call_and_return_conditional_losses_21546405в
Щ▓Х
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
annotationsк *
 z^trace_0
%:#
АА2dense_1134/kernel
:А2dense_1134/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
н
_non_trainable_variables

`layers
ametrics
blayer_regularization_losses
clayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
═
dtrace_0
etrace_12Ц
.__inference_dropout_929_layer_call_fn_21546410
.__inference_dropout_929_layer_call_fn_21546415│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zdtrace_0zetrace_1
Г
ftrace_0
gtrace_12╠
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546420
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546432│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zftrace_0zgtrace_1
"
_generic_user_object
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
н
hnon_trainable_variables

ilayers
jmetrics
klayer_regularization_losses
llayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
ё
mtrace_02╘
-__inference_dense_1135_layer_call_fn_21546441в
Щ▓Х
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
annotationsк *
 zmtrace_0
М
ntrace_02я
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546451в
Щ▓Х
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
annotationsк *
 zntrace_0
$:"	А2dense_1135/kernel
:2dense_1135/bias
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
'
o0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
НBК
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546023	input_206"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546255inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
КBЗ
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546272inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
НBК
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546171	input_206"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546298inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
еBв
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546338inputs"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
иBе
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546192	input_206"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
иBе
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546213	input_206"┐
╢▓▓
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
╧B╠
&__inference_signature_wrapper_21546238	input_206"Ф
Н▓Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
сB▐
-__inference_dense_1133_layer_call_fn_21546347inputs"в
Щ▓Х
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
annotationsк *
 
№B∙
H__inference_dense_1133_layer_call_and_return_conditional_losses_21546358inputs"в
Щ▓Х
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
annotationsк *
 
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
єBЁ
.__inference_dropout_928_layer_call_fn_21546363inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
.__inference_dropout_928_layer_call_fn_21546368inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546373inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546385inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
сB▐
-__inference_dense_1134_layer_call_fn_21546394inputs"в
Щ▓Х
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
annotationsк *
 
№B∙
H__inference_dense_1134_layer_call_and_return_conditional_losses_21546405inputs"в
Щ▓Х
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
annotationsк *
 
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
єBЁ
.__inference_dropout_929_layer_call_fn_21546410inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
єBЁ
.__inference_dropout_929_layer_call_fn_21546415inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546420inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ОBЛ
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546432inputs"│
к▓ж
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
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
сB▐
-__inference_dense_1135_layer_call_fn_21546441inputs"в
Щ▓Х
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
annotationsк *
 
№B∙
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546451inputs"в
Щ▓Х
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
annotationsк *
 
N
p	variables
q	keras_api
	rtotal
	scount"
_tf_keras_metric
.
r0
s1"
trackable_list_wrapper
-
p	variables"
_generic_user_object
:  (2total
:  (2count
):'	А2Adam/dense_1133/kernel/m
#:!А2Adam/dense_1133/bias/m
*:(
АА2Adam/dense_1134/kernel/m
#:!А2Adam/dense_1134/bias/m
):'	А2Adam/dense_1135/kernel/m
": 2Adam/dense_1135/bias/m
):'	А2Adam/dense_1133/kernel/v
#:!А2Adam/dense_1133/bias/v
*:(
АА2Adam/dense_1134/kernel/v
#:!А2Adam/dense_1134/bias/v
):'	А2Adam/dense_1135/kernel/v
": 2Adam/dense_1135/bias/vЬ
#__inference__wrapped_model_21545936u$%342в/
(в%
#К 
	input_206         
к "7к4
2

dense_1135$К!

dense_1135         й
H__inference_dense_1133_layer_call_and_return_conditional_losses_21546358]/в,
%в"
 К
inputs         
к "&в#
К
0         А
Ъ Б
-__inference_dense_1133_layer_call_fn_21546347P/в,
%в"
 К
inputs         
к "К         Ак
H__inference_dense_1134_layer_call_and_return_conditional_losses_21546405^$%0в-
&в#
!К
inputs         А
к "&в#
К
0         А
Ъ В
-__inference_dense_1134_layer_call_fn_21546394Q$%0в-
&в#
!К
inputs         А
к "К         Ай
H__inference_dense_1135_layer_call_and_return_conditional_losses_21546451]340в-
&в#
!К
inputs         А
к "%в"
К
0         
Ъ Б
-__inference_dense_1135_layer_call_fn_21546441P340в-
&в#
!К
inputs         А
к "К         л
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546373^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ л
I__inference_dropout_928_layer_call_and_return_conditional_losses_21546385^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Г
.__inference_dropout_928_layer_call_fn_21546363Q4в1
*в'
!К
inputs         А
p 
к "К         АГ
.__inference_dropout_928_layer_call_fn_21546368Q4в1
*в'
!К
inputs         А
p
к "К         Ал
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546420^4в1
*в'
!К
inputs         А
p 
к "&в#
К
0         А
Ъ л
I__inference_dropout_929_layer_call_and_return_conditional_losses_21546432^4в1
*в'
!К
inputs         А
p
к "&в#
К
0         А
Ъ Г
.__inference_dropout_929_layer_call_fn_21546410Q4в1
*в'
!К
inputs         А
p 
к "К         АГ
.__inference_dropout_929_layer_call_fn_21546415Q4в1
*в'
!К
inputs         А
p
к "К         Ан
&__inference_signature_wrapper_21546238В$%34?в<
в 
5к2
0
	input_206#К 
	input_206         "7к4
2

dense_1135$К!

dense_1135         ├
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546192k$%34:в7
0в-
#К 
	input_206         
p 

 
к "%в"
К
0         
Ъ ├
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546213k$%34:в7
0в-
#К 
	input_206         
p

 
к "%в"
К
0         
Ъ └
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546298h$%347в4
-в*
 К
inputs         
p 

 
к "%в"
К
0         
Ъ └
T__inference_sin_2-layers_512-nodes_layer_call_and_return_conditional_losses_21546338h$%347в4
-в*
 К
inputs         
p

 
к "%в"
К
0         
Ъ Ы
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546023^$%34:в7
0в-
#К 
	input_206         
p 

 
к "К         Ы
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546171^$%34:в7
0в-
#К 
	input_206         
p

 
к "К         Ш
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546255[$%347в4
-в*
 К
inputs         
p 

 
к "К         Ш
9__inference_sin_2-layers_512-nodes_layer_call_fn_21546272[$%347в4
-в*
 К
inputs         
p

 
к "К         