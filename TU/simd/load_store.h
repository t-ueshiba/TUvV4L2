/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOAD_STORE_H)
#define __TU_SIMD_LOAD_STORE_H

#include "TU/iterator.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  functions for supporting memory alignment				*
************************************************************************/
template <class T> inline T*
ceil(T* p)
{
    constexpr size_t	vsize = vec<typename std::remove_cv<T>::type>::size;
    const size_t	n = (reinterpret_cast<size_t>(p) - 1) / vsize;
    
    return reinterpret_cast<T*>(vsize * (n + 1));
}
    
template <class T> inline T*
floor(T* p)
{
    constexpr size_t	vsize = vec<typename std::remove_cv<T>::type>::size;
    const size_t	n = reinterpret_cast<size_t>(p) / vsize;
    
    return reinterpret_cast<T*>(vsize * n);
}

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
#  include "TU/simd/x86/load_store.h"
#elif defined(NEON)
#  include "TU/simd/arm/load_store.h"
#endif

#endif	// !__TU_SIMD_LOAD_STORE_H
