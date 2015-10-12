/*
 *  $Id$
 */
#if !defined(__TU_SIMD_SELECT_H)
#define __TU_SIMD_SELECT_H

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
#  include "TU/simd/intel/select.h"
#elif defined(NEON)
#  include "TU/simd/arm/select.h"
#endif

#endif	// !__TU_SIMD_SELECT_H
