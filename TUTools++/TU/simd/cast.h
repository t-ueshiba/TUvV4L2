/*!
  \file		cast.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトル間のキャスト関数の定義
*/
#if !defined(TU_SIMD_CAST_H)
#define TU_SIMD_CAST_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Cast operators							*
************************************************************************/
//! S型の成分を持つベクトルからT型の成分を持つベクトルへのキャストを行なう．
template <class T, class S> vec<T>	cast(vec<S> x)			;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/cast.h"
#elif defined(NEON)
#  include "TU/simd/arm/cast.h"
#endif

#endif	// !TU_SIMD_CAST_H
