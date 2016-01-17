/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_ALLOCATOR_H)
#define __TU_SIMD_X86_ALLOCATOR_H

#include <limits>
#include <new>			// for std::bad_alloc()
#include <immintrin.h>

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

    template <class U>	struct rebind	{ typedef allocator<U>	other; };
    
  public:
			allocator()					{}
    template <class U>	allocator(const allocator<U>&)			{}
			~allocator()					{}

    pointer		allocate(size_type n,
				 typename std::allocator<void>
					     ::const_pointer=nullptr)
			{
			    if (n == 0)
				return nullptr;
			    
			    pointer	p = static_cast<pointer>(
						_mm_malloc(sizeof(value_type)*n,
							   sizeof(value_type)));
			    if (p == nullptr)
				throw std::bad_alloc();
			    return p;
			}
    void		deallocate(pointer p, size_type)
			{
			    if (p != nullptr)
				_mm_free(p);
			}
    void		construct(pointer p, const_reference val)
			{
			    new(p) value_type(val);
			}
    void		destroy(pointer p)
			{
			    p->~value_type();
			}
    size_type		max_size() const
			{
			    return std::numeric_limits<size_type>::max()
				 / sizeof(value_type);
			}
    pointer		address(reference r)		const	{ return &r; }
    const_pointer	address(const_reference r)	const	{ return &r; }
};
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_ALLOCATOR_H
