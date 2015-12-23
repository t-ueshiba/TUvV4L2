/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVTDOWN_ITERATOR_H)
#define __TU_SIMD_CVTDOWN_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvtdown_iterator<T, ITER, MASK>				*
************************************************************************/
//! SIMDベクトルを出力する反復子を介して複数のSIMDベクトルを読み込み，それをより小さな成分を持つSIMDベクトルに変換する反復子
/*!
  \param T	変換先のSIMDベクトルの成分の型
  \param ITER	SIMDベクトルを出力する反復子
*/
template <class T, class ITER, bool MASK=false>
class cvtdown_iterator
    : public boost::iterator_adaptor<
		 cvtdown_iterator<T, ITER, MASK>,
		 ITER,
		 tuple_replace<iterator_value<ITER>, vec<T> >,
		 boost::single_pass_traversal_tag,
		 tuple_replace<iterator_value<ITER>, vec<T> > >
{
  private:
    typedef typename tuple_head<iterator_value<ITER> >::element_type	S;
    typedef boost::iterator_adaptor<
		cvtdown_iterator,
		ITER,
		tuple_replace<iterator_value<ITER>, vec<T> >,
		boost::single_pass_traversal_tag,
		tuple_replace<iterator_value<ITER>, vec<T> > >		super;

  public:
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;

    friend class	boost::iterator_core_access;

  public:
		cvtdown_iterator(const ITER& iter)	:super(iter)	{}

  private:
    template <class T_>
    typename std::enable_if<(vec<T_>::size == vec<S>::size), vec<T_> >::type
		cvtdown()
		{
		    auto	x = *super::base();
		    ++super::base_reference();
		    return cvt<T_, false, MASK>(x);
		}
    template <class T_>
    typename std::enable_if<(vec<T_>::size > vec<S>::size), vec<T_> >::type
		cvtdown()
		{
		    using A = cvt_above_type<T_, S, MASK>;
	  
		    auto	x = cvtdown<A>();
		    auto	y = cvtdown<A>();
		    return cvt<T_, MASK>(x, y);
		}

    reference	dereference() const
		{		    
		    return const_cast<cvtdown_iterator*>(this)->cvtdown<T>();
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvtdown_iterator<T, ITER, false>
make_cvtdown_iterator(ITER iter)
{
    return cvtdown_iterator<T, ITER, false>(iter);
}

template <class T, class ITER>
using cvtdown_mask_iterator = cvtdown_iterator<T, ITER, true>;
    
template <class T, class ITER> cvtdown_mask_iterator<T, ITER>
make_cvtdown_mask_iterator(ITER iter)
{
    return cvtdown_mask_iterator<T, ITER>(iter);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVTDOWN_ITERATOR_H
