/*
 *  $Id$
 */
#if !defined(TU_SIMD_LOAD_STORE_H)
#define TU_SIMD_LOAD_STORE_H

#include <memory>
#include <boost/iterator/iterator_adaptor.hpp>
#include "TU/pair.h"
#include "TU/simd/vec.h"
#include "TU/simd/pack.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  functions for supporting memory alignment				*
************************************************************************/
  /*
template <class T> inline T*
begin(T* p)
{
    constexpr size_t	vsize = sizeof(vec<typename std::remove_cv<T>::type>);
    size_t		space = 2*vsize - 1;
    void*		q = const_cast<void*>(p);
    
    return reinterpret_cast<T*>(std::align(vsize, vsize, q, space));
}

template <class T> inline T*
end(T* p)
{
    constexpr size_t	vsize = sizeof(vec<typename std::remove_cv<T>::type>);
    p = reinterpret_cast<T*>((char*)p - vsize + 1);

    return begin(p);
}
  */
/************************************************************************
*  ptr<T>								*
************************************************************************/
template <class T>
class ptr : public boost::iterator_adaptor<ptr<T>, T*>
{
  private:
    using	super = boost::iterator_adaptor<ptr, T*>;

  public:
    ptr(T* p=nullptr)	:super(p)	{}

    T*	get()		const	{ return super::base(); }
    operator T*()	const	{ return super::base(); }
};
    
/************************************************************************
*  Load/Store								*
************************************************************************/
//! メモリからベクトルをロードする．
/*!
  \param p	ロード元のメモリアドレス
  \return	ロードされたベクトル
*/
template <bool ALIGNED=false, size_t N=1, class T> pack<T, N>
load(const T* p)							;

template <bool ALIGNED, size_t N, class T>
inline std::enable_if_t<(N > 1), pack<T, N> >
load(const T* p)
{
    constexpr size_t	D = vec<T>::size * (N >> 1);
    
    return std::make_pair(load<ALIGNED, (N >> 1)>(p),
			  load<ALIGNED, (N >> 1)>(p + D));
}
    
//! メモリにベクトルをストアする．
/*!
  \param p	ストア先のメモリアドレス
  \param x	ストアされるベクトル
*/
template <bool ALIGNED=false, class T> void
store(T* p, vec<T> x)							;

template <bool ALIGNED=false, class T, class PACK> inline void
store(T* p, const std::pair<PACK, PACK>& x)
{
    constexpr size_t	D = vec<T>::size * pair_traits<PACK>::size;
    
    store(p,	 x.first);
    store(p + D, x.second);
}
    
}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/load_store.h"
#elif defined(NEON)
#  include "TU/simd/arm/load_store.h"
#endif

#endif	// !TU_SIMD_LOAD_STORE_H
