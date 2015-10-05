/*
 *  $Id$
 */
#if !defined(__TU_SIMD_BIT_SHIFT_H)
#define __TU_SIMD_BIT_SHIFT_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Bitwise shift operators						*
************************************************************************/
//! 整数ベクトルの左シフトを行う．
/*!
  \param x	整数ベクトル
  \param n	シフトするビット数
  \return	シフト後の整数ベクトル
*/
template <class T> static vec<T>	operator <<(vec<T> x, int n)	;

//! 整数ベクトルの算術右シフトを行なう．
/*!
  \param x	整数ベクトル
  \param n	シフトするビット数
  \return	シフト後の整数ベクトル
*/
template <class T> static vec<T>	operator >>(vec<T> x, int n)	;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/bit_shift.h"
#elif defined(NEON)
#  include "TU/simd/arm/bit_shift.h"
#endif

#endif	// !__TU_SIMD_BIT_SHIFT_H
