/**
  ******************************************************************************
  * @file    network.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2025-12-27T22:51:11+0900
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */


#include "network.h"
#include "network_data.h"

#include "ai_platform.h"
#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "core_convert.h"

#include "layers.h"



#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_network
 
#undef AI_NETWORK_MODEL_SIGNATURE
#define AI_NETWORK_MODEL_SIGNATURE     "0x10acf7421e948db9fefeb11031e22a26"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     ""
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "2025-12-27T22:51:11+0900"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_NETWORK_N_BATCHES
#define AI_NETWORK_N_BATCHES         (1)

static ai_ptr g_network_activations_map[1] = AI_C_ARRAY_INIT;
static ai_ptr g_network_weights_map[1] = AI_C_ARRAY_INIT;



/**  Array declarations section  **********************************************/
/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_input_layer0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 9216, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 35344, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 15488, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6400, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  gemm_10_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 6, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 144, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4608, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 18432, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  gemm_10_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 409600, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  gemm_10_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_weights_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 384, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 6, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 548, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3008, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6144, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2880, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 7168, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch1_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 2560, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  gemm_10_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 6720, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  gemm_11_scratch0_array, AI_ARRAY_FORMAT_S16,
  NULL, NULL, 94, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  nl_12_scratch0_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 31, AI_STATIC)

/**  Array metadata declarations section  *************************************/
/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002871324075385928f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_0_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002871324075385928f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_0_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 16,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021272541489452124f, 0.0016984634567052126f, 0.0020810398273169994f, 0.0013306086184456944f, 0.0024348339065909386f, 0.001960537862032652f, 0.002268432406708598f, 0.0017916170181706548f, 0.0019684245344251394f, 0.0018184265354648232f, 0.0031948506366461515f, 0.002148489700630307f, 0.0020202116575092077f, 0.0019750487990677357f, 0.0015671203145757318f, 0.0019183820113539696f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006413483526557684f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006413483526557684f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_2_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 32,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.002550331875681877f, 0.002108293818309903f, 0.0022528567351400852f, 0.002568464493378997f, 0.0027253299485892057f, 0.0021486803889274597f, 0.0019186976132914424f, 0.0017936008516699076f, 0.0028375692199915648f, 0.002690403489395976f, 0.001753288204781711f, 0.002843436086550355f, 0.003678614739328623f, 0.0022179170046001673f, 0.0019983905367553234f, 0.0025545088574290276f, 0.003128188429400325f, 0.0021094258408993483f, 0.0018669619457796216f, 0.001702256384305656f, 0.0019022340420633554f, 0.0015870587667450309f, 0.002907020039856434f, 0.002934967400506139f, 0.002096917014569044f, 0.002685186453163624f, 0.0025009838864207268f, 0.002238810993731022f, 0.0017658870201557875f, 0.0023228488862514496f, 0.003057172056287527f, 0.0020034899935126305f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03625316917896271f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_scratch1_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03625316917896271f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_4_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0021238261833786964f, 0.001922985422424972f, 0.002315069315955043f, 0.00245667970739305f, 0.0017757912864908576f, 0.002051437506452203f, 0.0016716554528102279f, 0.0029282644391059875f, 0.002933062147349119f, 0.0036962381564080715f, 0.001986070303246379f, 0.002352654468268156f, 0.0012044282630085945f, 0.002033720724284649f, 0.0026948931626975536f, 0.001987235387787223f, 0.0016923469956964254f, 0.0015537647996097803f, 0.002793355379253626f, 0.0023955840151757f, 0.0015316811623051763f, 0.0031830966472625732f, 0.0034942333586513996f, 0.0019103080267086625f, 0.002154591726139188f, 0.002587550785392523f, 0.0016093860613182187f, 0.002517266198992729f, 0.0020485660061240196f, 0.0019309712806716561f, 0.002136913361027837f, 0.0023166413884609938f, 0.0025337308179587126f, 0.0016634499188512564f, 0.0033672593999654055f, 0.0019836248829960823f, 0.0015261454973369837f, 0.002385936211794615f, 0.0015690579311922193f, 0.002333749318495393f, 0.0018049138598144054f, 0.0028918611351400614f, 0.0029187174513936043f, 0.002658606506884098f, 0.002118069911375642f, 0.002915286459028721f, 0.0014557503163814545f, 0.002332499483600259f, 0.0023282524198293686f, 0.0023087181616574526f, 0.0018909525824710727f, 0.0032449495047330856f, 0.0020843318197876215f, 0.0014926610747352242f, 0.0037808190099895f, 0.002154100453481078f, 0.003648483194410801f, 0.0018022112781181931f, 0.0029191020876169205f, 0.0019483949290588498f, 0.0016937772743403912f, 0.0027996767312288284f, 0.002026753965765238f, 0.0014414764009416103f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_10_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.2522430419921875f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_10_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 64,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023514837957918644f, 0.0025568148121237755f, 0.001493490650318563f, 0.0002868656883947551f, 0.0004888396942988038f, 0.002640918828547001f, 0.00048142115701921284f, 0.002964239101856947f, 0.00041796904406510293f, 0.0007492240401916206f, 0.00028649024898186326f, 0.00030740504735149443f, 0.0021610604599118233f, 0.002393616596236825f, 0.002368174260482192f, 0.0019499544287100434f, 0.0029214899986982346f, 0.002870122669264674f, 0.002693583955988288f, 0.0023976683150976896f, 0.002223177580162883f, 0.0022893256973475218f, 0.001900090603157878f, 0.00036851532058790326f, 0.0025012572295963764f, 0.0022031962871551514f, 0.001051991363056004f, 0.00031742569990456104f, 0.0002869482559617609f, 0.0025128074921667576f, 0.002389936475083232f, 0.0029776287265121937f, 0.0022881501354277134f, 0.0028101233765482903f, 0.0002871327451430261f, 0.0021769744344055653f, 0.0022454175632447004f, 0.0006840704591013491f, 0.0009717662469483912f, 0.0038260831497609615f, 0.0020132078789174557f, 0.0020925034768879414f, 0.0024619426112622023f, 0.002726820996031165f, 0.0018648699624463916f, 0.0004917257465422153f, 0.0032229567877948284f, 0.0021718244533985853f, 0.003010747954249382f, 0.0025225100107491016f, 0.002721902448683977f, 0.0005633281543850899f, 0.0019331647781655192f, 0.0026629611384123564f, 0.0004819864116143435f, 0.0023271343670785427f, 0.0019201065879315138f, 0.002883515553548932f, 0.002182366792112589f, 0.002200827933847904f, 0.0022367823403328657f, 0.0015226354589685798f, 0.000491259794216603f, 0.0004726263869088143f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.6563528776168823f),
    AI_PACK_INTQ_ZP(-8)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(gemm_11_weights_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 6,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0026360307820141315f, 0.0035278124269098043f, 0.002866367343813181f, 0.002785132499411702f, 0.0033729514107108116f, 0.0030814995989203453f),
    AI_PACK_INTQ_ZP(0, 0, 0, 0, 0, 0)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(nl_12_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_input_layer0_output_array_intq, AI_STATIC_CONST,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.003921568859368563f),
    AI_PACK_INTQ_ZP(-128)))

/**  Tensor declarations section  *********************************************/
/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  0, 0x0,
  AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_0_bias_array, NULL)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 47, 47), AI_STRIDE_INIT(4, 1, 1, 16, 752),
  1, &conv2d_0_output_array, &conv2d_0_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  2, 0x0,
  AI_SHAPE_INIT(4, 1, 548, 1, 1), AI_STRIDE_INIT(4, 1, 1, 548, 548),
  1, &conv2d_0_scratch0_array, NULL)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch1, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 94, 2), AI_STRIDE_INIT(4, 1, 1, 16, 1504),
  1, &conv2d_0_scratch1_array, &conv2d_0_scratch1_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 3, 3, 16), AI_STRIDE_INIT(4, 1, 1, 16, 48),
  1, &conv2d_0_weights_array, &conv2d_0_weights_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  5, 0x0,
  AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_2_bias_array, NULL)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 22, 22), AI_STRIDE_INIT(4, 1, 1, 32, 704),
  1, &conv2d_2_output_array, &conv2d_2_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  7, 0x0,
  AI_SHAPE_INIT(4, 1, 6144, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6144, 6144),
  1, &conv2d_2_scratch0_array, NULL)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch1, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 45, 2), AI_STRIDE_INIT(4, 1, 1, 32, 1440),
  1, &conv2d_2_scratch1_array, &conv2d_2_scratch1_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 16, 3, 3, 32), AI_STRIDE_INIT(4, 1, 16, 512, 1536),
  1, &conv2d_2_weights_array, &conv2d_2_weights_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  10, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_4_bias_array, NULL)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  11, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 1, 1, 64, 640),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output0, AI_STATIC,
  12, 0x1,
  AI_SHAPE_INIT(4, 1, 6400, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6400, 6400),
  1, &conv2d_4_output_array, &conv2d_4_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  13, 0x0,
  AI_SHAPE_INIT(4, 1, 7168, 1, 1), AI_STRIDE_INIT(4, 1, 1, 7168, 7168),
  1, &conv2d_4_scratch0_array, NULL)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch1, AI_STATIC,
  14, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 20, 2), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_4_scratch1_array, &conv2d_4_scratch1_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  15, 0x1,
  AI_SHAPE_INIT(4, 32, 3, 3, 64), AI_STRIDE_INIT(4, 1, 32, 2048, 6144),
  1, &conv2d_4_weights_array, &conv2d_4_weights_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  gemm_10_bias, AI_STATIC,
  16, 0x0,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &gemm_10_bias_array, NULL)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  gemm_10_output, AI_STATIC,
  17, 0x1,
  AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &gemm_10_output_array, &gemm_10_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  gemm_10_scratch0, AI_STATIC,
  18, 0x0,
  AI_SHAPE_INIT(4, 1, 6720, 1, 1), AI_STRIDE_INIT(4, 2, 2, 13440, 13440),
  1, &gemm_10_scratch0_array, NULL)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  gemm_10_weights, AI_STATIC,
  19, 0x1,
  AI_SHAPE_INIT(4, 6400, 64, 1, 1), AI_STRIDE_INIT(4, 1, 6400, 409600, 409600),
  1, &gemm_10_weights_array, &gemm_10_weights_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_bias, AI_STATIC,
  20, 0x0,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 4, 4, 24, 24),
  1, &gemm_11_bias_array, NULL)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_output, AI_STATIC,
  21, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &gemm_11_output_array, &gemm_11_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_scratch0, AI_STATIC,
  22, 0x0,
  AI_SHAPE_INIT(4, 1, 94, 1, 1), AI_STRIDE_INIT(4, 2, 2, 188, 188),
  1, &gemm_11_scratch0_array, NULL)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  gemm_11_weights, AI_STATIC,
  23, 0x1,
  AI_SHAPE_INIT(4, 64, 6, 1, 1), AI_STRIDE_INIT(4, 1, 64, 384, 384),
  1, &gemm_11_weights_array, &gemm_11_weights_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_output, AI_STATIC,
  24, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6, 6),
  1, &nl_12_output_array, &nl_12_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  nl_12_scratch0, AI_STATIC,
  25, 0x0,
  AI_SHAPE_INIT(4, 1, 31, 1, 1), AI_STRIDE_INIT(4, 4, 4, 124, 124),
  1, &nl_12_scratch0_array, NULL)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_input_layer0_output, AI_STATIC,
  26, 0x1,
  AI_SHAPE_INIT(4, 1, 1, 96, 96), AI_STRIDE_INIT(4, 1, 1, 1, 96),
  1, &serving_default_input_layer0_output_array, &serving_default_input_layer0_output_array_intq)



/**  Layer declarations section  **********************************************/



AI_STATIC_CONST ai_i32 nl_12_nl_params_data[] = { 1409507072, 26, -31 };
AI_ARRAY_OBJ_DECLARE(
    nl_12_nl_params, AI_ARRAY_FORMAT_S32,
    nl_12_nl_params_data, nl_12_nl_params_data, 3, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &nl_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  nl_12_layer, 12,
  SM_TYPE, 0x0, NULL,
  sm, forward_sm_integer,
  &nl_12_chain,
  NULL, &nl_12_layer, AI_STATIC, 
  .nl_params = &nl_12_nl_params, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_11_weights, &gemm_11_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_11_layer, 11,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_11_chain,
  NULL, &nl_12_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  gemm_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output0),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_10_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &gemm_10_weights, &gemm_10_bias),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &gemm_10_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  gemm_10_layer, 10,
  DENSE_TYPE, 0x0, NULL,
  dense, forward_dense_integer_SSSA_ch,
  &gemm_10_chain,
  NULL, &gemm_11_layer, AI_STATIC, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_4_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_4_scratch0, &conv2d_4_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 5,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_4_chain,
  NULL, &gemm_10_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_2_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_2_scratch0, &conv2d_2_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 3,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool,  forward_conv2d_deep_3x3_sssa8_ch_nl_pool,
  &conv2d_2_chain,
  NULL, &conv2d_4_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &serving_default_input_layer0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_0_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 3, &conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &conv2d_0_scratch0, &conv2d_0_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 1,
  OPTIMIZED_CONV2D_TYPE, 0x0, NULL,
  conv2d_nl_pool, forward_conv2d_sssa8_ch_nl_pool,
  &conv2d_0_chain,
  NULL, &conv2d_2_layer, AI_STATIC, 
  .groups = 1, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = AI_HANDLE_PTR(pool_func_mp_array_integer_INT8), 
  .in_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
  .out_ch_format = AI_LAYER_FORMAT_CHANNEL_LAST_VALID, 
)


#if (AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 433896, 1, 1),
    433896, NULL, NULL),
  AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
    AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 45072, 1, 1),
    45072, NULL, NULL),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_12_output),
  &conv2d_0_layer, 0x662a7721, NULL)

#else

AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 433896, 1, 1),
      433896, NULL, NULL)
  ),
  AI_BUFFER_ARRAY_OBJ_INIT_STATIC(
  	AI_FLAG_NONE, 1,
    AI_BUFFER_INIT(AI_FLAG_NONE,  AI_BUFFER_FORMAT_U8,
      AI_BUFFER_SHAPE_INIT(AI_SHAPE_BCWH, 4, 1, 45072, 1, 1),
      45072, NULL, NULL)
  ),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_IN_NUM, &serving_default_input_layer0_output),
  AI_TENSOR_LIST_IO_OBJ_INIT(AI_FLAG_NONE, AI_NETWORK_OUT_NUM, &nl_12_output),
  &conv2d_0_layer, 0x662a7721, NULL)

#endif	/*(AI_TOOLS_API_VERSION < AI_TOOLS_API_VERSION_1_5)*/



/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_activations(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_activations_map(g_network_activations_map, 1, params)) {
    /* Updating activations (byte) offsets */
    
    serving_default_input_layer0_output_array.data = AI_PTR(g_network_activations_map[0] + 27776);
    serving_default_input_layer0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 27776);
    conv2d_0_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 36992);
    conv2d_0_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 36992);
    conv2d_0_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 37540);
    conv2d_0_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 37540);
    conv2d_0_output_array.data = AI_PTR(g_network_activations_map[0] + 704);
    conv2d_0_output_array.data_start = AI_PTR(g_network_activations_map[0] + 704);
    conv2d_2_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 36048);
    conv2d_2_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 36048);
    conv2d_2_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 42192);
    conv2d_2_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 42192);
    conv2d_2_output_array.data = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_2_output_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    conv2d_4_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 15488);
    conv2d_4_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 15488);
    conv2d_4_scratch1_array.data = AI_PTR(g_network_activations_map[0] + 22656);
    conv2d_4_scratch1_array.data_start = AI_PTR(g_network_activations_map[0] + 22656);
    conv2d_4_output_array.data = AI_PTR(g_network_activations_map[0] + 25216);
    conv2d_4_output_array.data_start = AI_PTR(g_network_activations_map[0] + 25216);
    gemm_10_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_10_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_10_output_array.data = AI_PTR(g_network_activations_map[0] + 13440);
    gemm_10_output_array.data_start = AI_PTR(g_network_activations_map[0] + 13440);
    gemm_11_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    gemm_11_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    gemm_11_output_array.data = AI_PTR(g_network_activations_map[0] + 188);
    gemm_11_output_array.data_start = AI_PTR(g_network_activations_map[0] + 188);
    nl_12_scratch0_array.data = AI_PTR(g_network_activations_map[0] + 0);
    nl_12_scratch0_array.data_start = AI_PTR(g_network_activations_map[0] + 0);
    nl_12_output_array.data = AI_PTR(g_network_activations_map[0] + 124);
    nl_12_output_array.data_start = AI_PTR(g_network_activations_map[0] + 124);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_ACTIVATIONS);
  return false;
}




/******************************************************************************/
AI_DECLARE_STATIC
ai_bool network_configure_weights(
  ai_network* net_ctx, const ai_network_params* params)
{
  AI_ASSERT(net_ctx)

  if (ai_platform_get_weights_map(g_network_weights_map, 1, params)) {
    /* Updating weights (byte) offsets */
    
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 0);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(g_network_weights_map[0] + 144);
    conv2d_0_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 144);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(g_network_weights_map[0] + 208);
    conv2d_2_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 208);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(g_network_weights_map[0] + 4816);
    conv2d_2_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 4816);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(g_network_weights_map[0] + 4944);
    conv2d_4_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 4944);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(g_network_weights_map[0] + 23376);
    conv2d_4_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 23376);
    gemm_10_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_10_weights_array.data = AI_PTR(g_network_weights_map[0] + 23632);
    gemm_10_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 23632);
    gemm_10_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_10_bias_array.data = AI_PTR(g_network_weights_map[0] + 433232);
    gemm_10_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 433232);
    gemm_11_weights_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_weights_array.data = AI_PTR(g_network_weights_map[0] + 433488);
    gemm_11_weights_array.data_start = AI_PTR(g_network_weights_map[0] + 433488);
    gemm_11_bias_array.format |= AI_FMT_FLAG_CONST;
    gemm_11_bias_array.data = AI_PTR(g_network_weights_map[0] + 433872);
    gemm_11_bias_array.data_start = AI_PTR(g_network_weights_map[0] + 433872);
    return true;
  }
  AI_ERROR_TRAP(net_ctx, INIT_FAILED, NETWORK_WEIGHTS);
  return false;
}


/**  PUBLIC APIs SECTION  *****************************************************/



AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_network_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 18615568,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .params            = AI_STRUCT_INIT,
      .activations       = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x662a7721,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}



AI_API_ENTRY
ai_bool ai_network_get_report(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if (report && net_ctx)
  {
    ai_network_report r = {
      .model_name        = AI_NETWORK_MODEL_NAME,
      .model_signature   = AI_NETWORK_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = AI_STRUCT_INIT,

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 18615568,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .map_signature     = AI_MAGIC_SIGNATURE,
      .map_weights       = AI_STRUCT_INIT,
      .map_activations   = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x662a7721,
    };

    if (!ai_platform_api_get_network_report(network, &r)) return false;

    *report = r;
    return true;
  }
  return false;
}


AI_API_ENTRY
ai_error ai_network_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}


AI_API_ENTRY
ai_error ai_network_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    AI_CONTEXT_OBJ(&AI_NET_OBJ_INSTANCE),
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}


AI_API_ENTRY
ai_error ai_network_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[])
{
  ai_error err;
  ai_network_params params;

  err = ai_network_create(network, AI_NETWORK_DATA_CONFIG);
  if (err.type != AI_ERROR_NONE) {
    return err;
  }
  
  if (ai_network_data_params_get(&params) != true) {
    err = ai_network_get_error(*network);
    return err;
  }
#if defined(AI_NETWORK_DATA_ACTIVATIONS_COUNT)
  /* set the addresses of the activations buffers */
  for (ai_u16 idx=0; activations && idx<params.map_activations.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_activations, idx, activations[idx]);
  }
#endif
#if defined(AI_NETWORK_DATA_WEIGHTS_COUNT)
  /* set the addresses of the weight buffers */
  for (ai_u16 idx=0; weights && idx<params.map_weights.size; idx++) {
    AI_BUFFER_ARRAY_ITEM_SET_ADDRESS(&params.map_weights, idx, weights[idx]);
  }
#endif
  if (ai_network_init(*network, &params) != true) {
    err = ai_network_get_error(*network);
  }
  return err;
}


AI_API_ENTRY
ai_buffer* ai_network_inputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_inputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_buffer* ai_network_outputs_get(ai_handle network, ai_u16 *n_buffer)
{
  if (network == AI_HANDLE_NULL) {
    network = (ai_handle)&AI_NET_OBJ_INSTANCE;
    AI_NETWORK_OBJ(network)->magic = AI_MAGIC_CONTEXT_TOKEN;
  }
  return ai_platform_outputs_get(network, n_buffer);
}


AI_API_ENTRY
ai_handle ai_network_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}


AI_API_ENTRY
ai_bool ai_network_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = AI_NETWORK_OBJ(ai_platform_network_init(network, params));
  ai_bool ok = true;

  if (!net_ctx) return false;
  ok &= network_configure_weights(net_ctx, params);
  ok &= network_configure_activations(net_ctx, params);

  ok &= ai_platform_network_post_init(network);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_network_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}


AI_API_ENTRY
ai_i32 ai_network_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}



#undef AI_NETWORK_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

