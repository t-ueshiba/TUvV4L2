/*!
  \file		lookup.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルに対する雑多な関数の定義
*/
#if !defined(TU_SIMD_MISC_H)
#define TU_SIMD_MISC_H

#include "TU/simd/vec.h"
#if defined(MMX)
#  include "TU/simd/x86/shuffle.h"
#  include "TU/simd/x86/svml.h"
#endif

#endif	// !TU_SIMD_MISC_H
