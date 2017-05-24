/*
 *  $Id$
 */
#if !defined(TU_SIMD_INSERT_EXTRACT_H)
#define TU_SIMD_INSERT_EXTRACT_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Inserting/Extracting elements					*
************************************************************************/
//! ベクトルの指定された位置に成分を挿入する．
/*!
  \param I	挿入する位置を指定するindex
  \param x	ベクトル
  \return	成分を挿入されたベクトル
*/
template <size_t I, class T> vec<T>	insert(vec<T> x, int val)	;

//! ベクトルから指定された位置から成分を取り出す．
/*!
  \param I	取り出す位置を指定するindex
  \param x	ベクトル
  \return	取り出された成分
*/
template <size_t I, class T> int	extract(vec<T> x)		;

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/insert_extract.h"
#elif defined(NEON)
#  include "TU/simd/arm/insert_extract.h"
#endif

#endif	// !TU_SIMD_INSERT_EXTRACT_H
