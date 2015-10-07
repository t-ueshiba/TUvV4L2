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
*  class cvtdown_iterator<T, ITER>					*
************************************************************************/
//! SIMDベクトルを出力する反復子を介して複数のSIMDベクトルを読み込み，それをより小さな成分を持つSIMDベクトルに変換する反復子
/*!
  \param T	変換先のSIMDベクトルの成分の型
  \param ITER	SIMDベクトルを出力する反復子
*/
template <class T, class ITER>
class cvtdown_iterator
    : public boost::iterator_adaptor<
		 cvtdown_iterator<T, ITER>,		// self
		 ITER,					// base
		 tuple_replace<iterator_value<ITER>, vec<T> >,
		 boost::single_pass_traversal_tag,
		 tuple_replace<iterator_value<ITER>, vec<T> > >
{
  private:
    typedef boost::iterator_adaptor<
		cvtdown_iterator,
		ITER,
		tuple_replace<iterator_value<ITER>, vec<T> >,
		boost::single_pass_traversal_tag,
		tuple_replace<iterator_value<ITER>, vec<T> > >
							super;
    typedef iterator_value<ITER>			elementary_vec;

    typedef typename tuple_head<elementary_vec>::element_type
							element_type;
    typedef simd::complementary_type<element_type>	complementary_type;
    typedef tuple_replace<elementary_vec, vec<complementary_type> >
							complementary_vec;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
		cvtdown_iterator(ITER const& iter)	:super(iter)	{}

  private:
    void	cvtdown(elementary_vec& x)
		{
		    x = *super::base();
		    ++super::base_reference();
		}
    void	cvtdown(complementary_vec& x)
		{
		    cvtdown(x,
			    std::integral_constant<
			        bool, (vec<complementary_type>::size ==
				       vec<element_type>::size)>());
		}
    void	cvtdown(complementary_vec& x, std::true_type)
		{
		    elementary_vec	y;
		    cvtdown(y);
		    x = cvt<complementary_type>(y);
		}
    void	cvtdown(complementary_vec& x, std::false_type)
		{
		    elementary_vec	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt<complementary_type>(y, z);
		}
    template <class VEC_>
    void	cvtdown(VEC_& x)
		{
		    typedef typename
			tuple_head<VEC_>::element_type	S;
		    typedef upper_type<S>		upper_type;
		    typedef typename
			std::conditional<std::is_floating_point<S>::value,
					 upper_type,
					 signed_type<upper_type> >::type
							signed_upper_type;
		    
		    tuple_replace<elementary_vec, vec<signed_upper_type> > y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt<S>(y, z);
		}

    reference	dereference() const
		{
		    reference	x;
		    const_cast<cvtdown_iterator*>(this)->cvtdown(x);
		    return x;
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvtdown_iterator<T, ITER>
make_cvtdown_iterator(ITER iter)
{
    return cvtdown_iterator<T, ITER>(iter);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVTDOWN_ITERATOR_H
