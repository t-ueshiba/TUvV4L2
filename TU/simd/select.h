/*!
  \file		select.h
  \author	Toshio UESHIBA
  \brief	マスクに基づいて2つのSIMDベクトルから成分を選択する関数の定義
*/
#if !defined(TU_SIMD_SELECT_H)
#define TU_SIMD_SELECT_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Selection								*
************************************************************************/
//! 2つのベクトル中の成分のいずれかをマスク値に応じて選択する．
/*!
 \param mask	マスク
 \param x	ベクトル
 \param y	ベクトル
 \return	maskにおいて1が立っている成分はxから，そうでない成分は
		yからそれぞれ選択して生成されたベクトル
*/
template <class T>
vec<T>	select(vec<mask_type<T> > mask, vec<T> x, vec<T> y)		;
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/select.h"
#elif defined(NEON)
#  include "TU/simd/arm/select.h"
#endif

#endif	// !TU_SIMD_SELECT_H
