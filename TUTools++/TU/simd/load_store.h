/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOAD_STORE_H)
#define __TU_SIMD_LOAD_STORE_H

#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Load/Store								*
************************************************************************/
//! メモリからベクトルをロードする．
/*!
  \param p	ロード元のメモリアドレス
  \return	ロードされたベクトル
*/
template <bool ALIGNED=false, class T> vec<T>	load(const T* p)	;

//! メモリにベクトルをストアする．
/*!
  \param p	ストア先のメモリアドレス
  \param x	ストアされるベクトル
*/
template <bool ALIGNED=false, class T> void	store(T* p, vec<T> x)	;
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/load_store.h"
#elif defined(NEON)
#  include "TU/simd/arm/load_store.h"
#endif

#endif	// !__TU_SIMD_LOAD_STORE_H
