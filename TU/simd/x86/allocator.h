/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_ALLOCATOR_H)
#define __TU_SIMD_X86_ALLOCATOR_H

#include <limits>
#include <new>			// for std::bad_alloc()
#include <type_traits>

namespace TU
{
namespace simd
{
/************************************************************************
*  class allocator<T>							*
************************************************************************/
template <class T>
class allocator
{
  public:
    using value_type	= T;

    template <class T_>
    struct align
    {
	constexpr static size_t	value = sizeof(T_);
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

    static T*	allocate(std::size_t n)
		{
		    if (n == 0)
			return nullptr;
			    
		    auto	p = static_cast<T*>(
					_mm_malloc(sizeof(value_type)*n,
						   align<T>::value));
		    if (p == nullptr)
			throw std::bad_alloc();
		    return p;
		}
    static void	deallocate(T* p, std::size_t)
		{
		    if (p != nullptr)
			_mm_free(p);
		}
};
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_ALLOCATOR_H
