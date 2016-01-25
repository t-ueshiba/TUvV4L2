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
*  functions for supporting memory alignment				*
************************************************************************/
template <class ITER> inline ITER
ceil(ITER iter)
{
    return iter;
}
    
template <class T> inline T*
ceil(T* p)
{
    constexpr size_t	vsize = vec<typename std::remove_cv<T>::type>::size;
    const size_t	n = (reinterpret_cast<size_t>(p) - 1) / vsize;
    
    return reinterpret_cast<T*>(vsize * (n + 1));
}
    
template <class ITER> inline ITER
floor(ITER iter)
{
    return iter;
}
    
template <class T> inline T*
floor(T* p)
{
    constexpr size_t	vsize = vec<typename std::remove_cv<T>::type>::size;
    const size_t	n = reinterpret_cast<size_t>(p) / vsize;
    
    return reinterpret_cast<T*>(vsize * n);
}

template <class T> inline auto
begin(T& x) -> decltype(ceil(std::begin(x)))
{
    return ceil(std::begin(x));
}
    
template <class T> inline auto
cbegin(const T& x) -> decltype(ceil(std::cbegin(x)))
{
    return ceil(std::begin(x));
}
    
template <class T> inline auto
end(T& x) -> decltype(floor(std::end(x)))
{
    return floor(std::end(x));
}
    
template <class T> inline auto
cend(const T& x) -> decltype(floor(std::cend(x)))
{
    return floor(std::cend(x));
}
    
template <class T> inline auto
rbegin(T& x) -> decltype(floor(std::rbegin(x)))
{
    return floor(std::rbegin(x));
}
    
template <class T> inline auto
crbegin(const T& x) -> decltype(floor(std::crbegin(x)))
{
    return floor(std::rbegin(x));
}
    
template <class T> inline auto
rend(T& x) -> decltype(ceil(std::rend(x)))
{
    return ceil(std::rend(x));
}
    
template <class T> inline auto
crend(const T& x) -> decltype(ceil(std::crend(x)))
{
    return ceil(std::crend(x));
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
