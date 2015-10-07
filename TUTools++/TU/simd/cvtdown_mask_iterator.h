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
		 cvtdown_mask_iterator<T, ITER>,	// self
		 ITER,					// base
		 tuple_replace<iterator_value<ITER>, vec<T> >,
		 boost::single_pass_traversal_tag,
		 tuple_replace<iterator_value<ITER>, vec<T> > >
{
  private:
    typedef boost::iterator_adaptor<
		cvtdown_mask_iterator,
		ITER,
		tuple_replace<iterator_value<ITER>, vec<T> >,
		boost::single_pass_traversal_tag,
		tuple_replace<iterator_value<ITER>, vec<T> > >
							super;
    typedef iterator_value<ITER>			elementary_vec;
    typedef typename tuple_head<elementary_vec>::element_type
							element_type;
    typedef complementary_mask_type<element_type>	complementary_type;
    typedef tuple_replace<elementary_vec, vec<complementary_type> >
							complementary_vec;
    typedef typename std::conditional<
		std::is_floating_point<element_type>::value,
		complementary_type, element_type>::type	integral_type;
    typedef tuple_replace<elementary_vec, vec<integral_type> >
							integral_vec;
    typedef typename std::conditional<
		std::is_signed<integral_type>::value,
		unsigned_type<integral_type>,
		signed_type<integral_type> >::type	flipped_type;
    typedef tuple_replace<elementary_vec, vec<flipped_type> >
							flipped_vec;
    typedef lower_type<flipped_type>			flipped_lower_type;
    typedef tuple_replace<elementary_vec, vec<flipped_lower_type> >
							flipped_lower_vec;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
		cvtdown_mask_iterator(ITER const& iter)	:super(iter)	{}

  private:
    void	cvtdown(elementary_vec& x)
		{
		    x = *super::base();
		    ++super::base_reference();
		}
    void	cvtdown(complementary_vec& x)
		{
		    elementary_vec	y;
		    cvtdown(y);
		    x = cvt_mask<complementary_type>(y);
		}
    void	cvtdown(flipped_vec& x)
		{
		    integral_vec	y;
		    cvtdown(y);
		    x = cvt_mask<flipped_type>(y);
		}
    void	cvtdown(flipped_lower_vec& x)
		{
		    integral_vec	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<flipped_lower_type>(y, z);
		}
    template <class VEC_>
    void	cvtdown(VEC_& x)
		{
		    typedef typename
			tuple_head<VEC_>::element_type	S;
		    typedef upper_type<S>		upper_type;
		    
		    tuple_replace<elementary_vec, vec<upper_type> >	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<S>(y, z);
		}

    reference	dereference() const
		{
		    reference	x;
		    const_cast<cvtdown_mask_iterator*>(this)->cvtdown(x);
		    return x;
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
