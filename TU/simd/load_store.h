/*!
  \file		load_store.h
  \author	Toshio UESHIBA
  \brief	メモリとSIMDベクトル間のデータ転送を行う関数の定義
*/
#if !defined(TU_SIMD_LOAD_STORE_H)
#define TU_SIMD_LOAD_STORE_H

#include <memory>
#include <boost/iterator/iterator_adaptor.hpp>
#include "TU/simd/vec.h"
#include "TU/simd/pack.h"

namespace std
{
#if !defined(__clang__) && (__GNUG__ <= 4)
inline void*
align(std::size_t alignment,
      std::size_t size, void*& ptr, std::size_t& space) noexcept
{
    std::uintptr_t	pn	= reinterpret_cast<std::uintptr_t>(ptr);
    std::uintptr_t	aligned = (pn + alignment - 1) & -alignment;
    std::size_t		padding = aligned - pn;

    if (space < size + padding)
	return nullptr;

    space -= padding;
    return ptr = reinterpret_cast<void*>(aligned);
}
#endif
}

namespace TU
{
namespace simd
{
/************************************************************************
*  functions for memory alignment					*
************************************************************************/
template <class T> inline auto
ceil(T* p)
{
    constexpr size_t	vsize = sizeof(vec<std::remove_cv_t<T> >);
    size_t		space = 2*vsize - 1;
    void*		q     = const_cast<void*>(p);
    
    return reinterpret_cast<T*>(std::align(vsize, vsize, q, space));
}

template <class T> inline auto
floor(T* p)
{
    constexpr size_t	vsize = sizeof(vec<std::remove_cv_t<T> >);

    return ceil(reinterpret_cast<T*>(reinterpret_cast<char*>(p) - vsize + 1));
}

/************************************************************************
*  iterator_wrapper<ITER>						*
************************************************************************/
template <class ITER>
class iterator_wrapper
    : public boost::iterator_adaptor<iterator_wrapper<ITER>, ITER>
{
  private:
    using	super = boost::iterator_adaptor<iterator_wrapper, ITER>;

  public:
		iterator_wrapper(ITER iter)	:super(iter)	{}
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		iterator_wrapper(ITER_ iter)	:super(iter)	{}

    ITER	get()			const	{ return super::base(); }
		operator ITER()		const	{ return super::base(); }
};

template <class ITER> inline iterator_wrapper<ITER>
make_iterator_wrapper(ITER iter)
{
    return {iter};
}
    
template <class ITER> inline ITER
make_accessor(iterator_wrapper<ITER> iter)
{
    return iter.get();
}

namespace detail
{
  template <class ITER>
  struct vsize
  {
      constexpr static size_t	value = std::iterator_traits<ITER>::value_type
								  ::size;
  };
  template <class T>
  struct vsize<T*>
  {
      constexpr static size_t	value = vec<std::remove_cv_t<T> >::size;
  };
}

template <class ITER> constexpr inline size_t
make_terminator(size_t n)
{
    return (n ? (n - 1)/detail::vsize<ITER>::value + 1 : 0);
}
    
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
