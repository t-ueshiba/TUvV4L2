/*
 *  $Id$
 */
#if !defined(__TU_SIMD_MULTIPLEX_ITERATOR_H)
#define __TU_SIMD_MULTIPLEX_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
template <size_t N, class ITER>
class multiplex_iterator
    : public boost::iterator_adaptor<
		 multiplex_iterator<N, ITER>,
		 ITER,
		 pack<typename iterator_value<ITER>::element_type, N>,
		 boost::single_pass_traversal_tag,
		 pack<typename iterator_value<ITER>::element_type, N> >
{
  private:
    typedef typename iterator_value<ITER>::element_type		element_type;
    typedef boost::iterator_adaptor<multiplex_iterator,
				    ITER,
				    pack<element_type, N>,
				    boost::single_pass_traversal_tag,
				    pack<element_type, N> >	super;

    template <size_t N_, class=void>
    struct dereference_impl
    {
	static pack<element_type, N_>
	exec(ITER& iter)
	{
	    const auto&	x = dereference_impl<(N_ >> 1)>::exec(iter);
	    const auto&	y = dereference_impl<(N_ >> 1)>::exec(iter);
	    return std::make_pair(x, y);
	}
    };
    template <class DUMMY_>
    struct dereference_impl<1, DUMMY_>
    {
	static iterator_value<ITER>
	exec(ITER& iter)
	{
	    const auto	x = *iter;
	    ++iter;
	    return x;
	}
    };
    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;
    
  public:
    multiplex_iterator(const ITER& iter)	:super(iter)		{}

  private:
    reference	dereference() const
		{
		    return dereference_impl<N>::exec(
			       const_cast<multiplex_iterator*>(this)
			       ->super::base_reference());
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};

template <size_t N, class ITER> inline multiplex_iterator<N, ITER>
make_multiplex_iterator(ITER iter)
{
    return multiplex_iterator<N, ITER>(iter);
}

template <class... ITERS> inline auto
make_iterator_tuple(ITERS... iters)
    ->decltype(boost::make_tuple(
		   make_multiplex_iterator<
		       lcm(iterator_value<ITERS>::size...)/
		       iterator_value<ITERS>::size>(iters)...))
{
    return boost::make_tuple(
	make_multiplex_iterator<lcm(iterator_value<ITERS>::size...)/
				iterator_value<ITERS>::size>(iters)...);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_MULTIPLEX_ITERATOR_H
