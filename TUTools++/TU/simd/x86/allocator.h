/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_ALLOCATOR_H)
#define __TU_SIMD_X86_ALLOCATOR_H

#include <new>			// for std::bad_alloc()
#include <type_traits>
#include <stdlib.h>

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
#endif	// !__TU_SIMD_X86_ALLOCATOR_H
