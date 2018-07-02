/*!
  \file		allocator.h
  \author	Toshio UESHIBA
  \brief	SIMD演算のためのアロケータクラスの定義と実装
*/
#if !defined(TU_SIMD_ALLOCATOR_H)
#define TU_SIMD_ALLOCATOR_H

#include <new>			// for std::bad_alloc()
#include <cstdlib>
#include "TU/simd/iterator_wrapper.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class allocator<T>							*
************************************************************************/
//! SIMD演算を実行するためのメモリ領域を確保するアロケータを表すクラス
/*!
  T が算術型の場合は sizeof(vec<T>) バイトに，T = vec<T_> の場合は
  sizeof(vec<T_>) バイトに，それぞれalignされた領域を返す．
  \param T	メモリ領域の要素の型
*/
template <class T>
class allocator
{
  public:
    using value_type	= T;
    using pointer	= iterator_wrapper<T*, true>;
    using const_pointer	= iterator_wrapper<const T*, true>;

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
