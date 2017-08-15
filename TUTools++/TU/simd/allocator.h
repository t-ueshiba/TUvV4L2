/*!
  \file		allocator.h
  \author	Toshio UESHIBA
  \brief	SIMD演算のためのアロケータクラスの定義と実装
*/
#if !defined(TU_SIMD_ALLOCATOR_H)
#define TU_SIMD_ALLOCATOR_H

#include <new>			// for std::bad_alloc()
#include <type_traits>
#include <stdlib.h>
#include <boost/iterator/iterator_adaptor.hpp>
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  iterator_wrapper<ITER, ALIGNED>					*
************************************************************************/
//! 反復子をラップして名前空間 TU::simd に取り込むためのクラス
/*!
  本クラスの目的は，以下の2つである．
    (1)	iterator_value<ITER> がSIMDベクトル型となるあらゆる反復子を
	その機能を保持したまま名前を変更することにより，TU/algorithm.h にある
	関数のオーバーロード版(本ファイルで定義)を呼び出せるようにすること．
    (2)	ITER が const T* 型または T* 型のとき，それぞれ load_iterator<T, true>
	store_iterator<T, true> を生成できるようにすること．本クラスでラップ
	されたポインタが指すアドレスは sizeof(vec<T>) にalignされているものと
	みなされる．
	
  \param ITER	ラップする反復子の型
*/ 
template <class ITER, bool ALIGNED>
class iterator_wrapper
    : public boost::iterator_adaptor<iterator_wrapper<ITER, ALIGNED>, ITER>
{
  public:
    using element_type	= typename std::iterator_traits<ITER>::value_type;
    template <class U_>
    struct rebind
    {
	using type	= iterator_wrapper<U_*, ALIGNED>;
    };
    
  private:
    using super		= boost::iterator_adaptor<iterator_wrapper, ITER>;

  public:
    iterator_wrapper(ITER iter)	:super(iter)	{}
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
    iterator_wrapper(ITER_ iter) :super(iter)	{}
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
    iterator_wrapper(iterator_wrapper<ITER_, ALIGNED> iter)
	:super(iter.base())			{}

    operator ITER()			const	{ return super::base(); }
};

/************************************************************************
*  class allocator<T>							*
************************************************************************/
//! SIMD演算を実行するためのメモリ領域を確保するアロケータを表すクラス
/*!
  T が算術型の場合は sizeof(vec<T>) バイトに，T = vec<T_> の場合は
  sizeof(vec<T_>) バイトに，それぞれalignされた領域を返す．
  \param T	メモリ領域の要素の型
*/
template <class T, bool ALIGNED=true>
class allocator
{
  public:
    using value_type	= T;
    using pointer	= iterator_wrapper<T*, ALIGNED>;
    using const_pointer	= iterator_wrapper<const T*, ALIGNED>;

  private:
    template <class T_>
    struct align
    {
	constexpr static size_t	value = sizeof(vec<T_>);
    };
    template <class T_>
    struct align<vec<T_> >
    {
	constexpr static size_t	value = sizeof(vec<T_>);
    };
    template <class... T_>
    struct align<std::tuple<T_...> >
    {
	constexpr static size_t	value = sizeof(vec<std::common_type_t<T_...> >);
    };
    template <class... T_>
    struct align<std::tuple<vec<T_>...> >
    {
	constexpr static size_t	value = sizeof(vec<std::common_type_t<T_...> >);
    };

  public:
    constexpr static size_t	Alignment = align<T>::value;
    
  public:
		allocator()						{}

    pointer	allocate(std::size_t n)
		{
		    if (n == 0)
			return nullptr;

		    void*	p;
		    if (posix_memalign(&p, Alignment, n*sizeof(T)))
			throw std::bad_alloc();
			
		    return static_cast<T*>(p);
		}
    void	deallocate(pointer p, std::size_t)
		{
		    if (p != nullptr)
			free(p);
		}
};
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ALLOCATOR_H)


