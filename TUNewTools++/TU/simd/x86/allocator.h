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
    typedef T		value_type;
    typedef T*		pointer;
    typedef const T*	const_pointer;
    typedef T&		reference;
    typedef const T&	const_reference;
    typedef size_t	size_type;
    typedef ptrdiff_t	difference_type;

    template <class T_>	struct rebind	{ typedef allocator<T_>	other; };

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
    
  public:
			allocator()				{}
    template <class U>	allocator(const allocator<U>&)		{}

    static pointer	allocate(size_type n,
				 typename std::allocator<void>
					     ::const_pointer=nullptr)
			{
			    if (n == 0)
				return nullptr;
			    
			    pointer	p = static_cast<pointer>(
						_mm_malloc(sizeof(value_type)*n,
							   align<T>::value));
			    if (p == nullptr)
				throw std::bad_alloc();
			    return p;
			}
    static void		deallocate(pointer p, size_type)
			{
			    if (p != nullptr)
				_mm_free(p);
			}
    static void		construct(pointer p, const_reference val)
			{
			    new(p) value_type(val);
			}
    static void		destroy(pointer p)
			{
			    p->~value_type();
			}
    constexpr
    static size_type	max_size()
			{
			    return std::numeric_limits<size_type>::max()
				 / sizeof(value_type);
			}
    static pointer	address(reference r)			{ return &r; }
    static const_pointer
			address(const_reference r)		{ return &r; }
};
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_ALLOCATOR_H
