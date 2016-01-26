/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: mmInstructions.h 1826 2015-05-07 01:47:56Z ueshiba $
 */
/*!
  \file		simd.h
  \brief	CPUのSIMD命令に関連するクラスと関数の定義と実装
*/
#if !defined(__TU_SIMD_SIMD_H)
#define __TU_SIMD_SIMD_H

#include "TU/simd/config.h"

#if defined(SIMD)
#  include "TU/simd/vec.h"
#  include "TU/simd/allocator.h"

#  include "TU/simd/load_store.h"
#  include "TU/simd/zero.h"
#  include "TU/simd/cast.h"
#  include "TU/simd/insert_extract.h"
#  include "TU/simd/shift.h"
#  include "TU/simd/bit_shift.h"
#  include "TU/simd/dup.h"
#  include "TU/simd/cvt.h"
#  include "TU/simd/logical.h"
#  include "TU/simd/compare.h"
#  include "TU/simd/select.h"
#  include "TU/simd/arithmetic.h"
#  include "TU/simd/misc.h"
#  include "TU/simd/transform.h"
#  include "TU/simd/lookup.h"

#  include "TU/simd/load_iterator.h"
#  include "TU/simd/store_iterator.h"
#  include "TU/simd/cvtdown_iterator.h"
#  include "TU/simd/cvtup_iterator.h"
#  include "TU/simd/shift_iterator.h"
#  include "TU/simd/row_vec_iterator.h"
#endif

#endif	// !__TU_SIMD_SIMD_H
