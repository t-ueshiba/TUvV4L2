/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVTDOWN_MASK_ITERATOR_H)
#define __TU_SIMD_CVTDOWN_MASK_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt_mask.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvtdown_mask_iterator<T, ITER>					*
************************************************************************/
//! SIMDマスクベクトルを出力する反復子を介して複数のSIMDマスクベクトルを読み込み，それをより小さな成分を持つSIMDマスクベクトルに変換する反復子
/*!
  \param T	変換先のSIMDマスクベクトルの成分の型
  \param ITER	SIMDマスクベクトルを出力する反復子
*/
template <class T, class ITER>
class cvtdown_mask_iterator
    : public boost::iterator_adaptor<
		 cvtdown_mask_iterator<T, ITER>,		// self
		 ITER,						// base
		 tuple_replace<iterator_value<ITER>, vec<T> >,
		 boost::single_pass_traversal_tag,
		 tuple_replace<iterator_value<ITER>, vec<T> > >
{
  private:
    typedef tuple_head<iterator_value<ITER> >			src_vec;
    typedef boost::iterator_adaptor<
		cvtdown_mask_iterator,
		ITER,
		tuple_replace<iterator_value<ITER>, vec<T> >,
		boost::single_pass_traversal_tag,
		tuple_replace<iterator_value<ITER>, vec<T> > >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
		cvtdown_mask_iterator(const ITER& iter)	:super(iter)	{}

  private:
    template <class T_>
    typename std::enable_if<(vec<T_>::size == src_vec::size), vec<T_> >::type
		cvtdown()
		{
		    auto	x = *super::base();
		    ++super::base_reference();
		    return cvt_mask<T_>(x);
		}
    template <class T_>
    typename std::enable_if<(vec<T_>::size > src_vec::size), vec<T_> >::type
		cvtdown()
		{
		    using A = cvt_mask_above_type<
				  T_, typename src_vec::element_type>;

		    auto	x = cvtdown<A>();
		    auto	y = cvtdown<A>();
		    return cvt_mask<T_>(x, y);
		}

    reference	dereference() const
		{
		    return
			const_cast<cvtdown_mask_iterator*>(this)->cvtdown<T>();
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvtdown_mask_iterator<T, ITER>
make_cvtdown_mask_iterator(ITER iter)
{
    return cvtdown_mask_iterator<T, ITER>(iter);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVTDOWN_MASK_ITERATOR_H
