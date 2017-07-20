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
#include "TU/simd/vec.h"

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
    struct align<std::tuple<vec<T_>...> >
    {
	constexpr static size_t	value = sizeof(vec<std::common_type_t<T_...> >);
    };
    
  public:
		allocator()						{}

    T*		allocate(std::size_t n)
		{
		    if (n == 0)
			return nullptr;

		    void*	p;
		    if (posix_memalign(&p, align<T>::value, n*sizeof(T)))
			throw std::bad_alloc();
			
		    return static_cast<T*>(p);
		}
    void	deallocate(T* p, std::size_t)
		{
		    if (p != nullptr)
			free(p);
		}
};
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ALLOCATOR_H)


