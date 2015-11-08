/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVT_MASK_ITERATOR_H)
#define __TU_SIMD_CVT_MASK_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt_mask.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvt_mask_iterator<T, ITER>					*
************************************************************************/
//! SIMDマスクベクトルを出力する反復子を介して1つまたは複数のSIMDマスクベクトルを読み込み，それを指定された成分を持つSIMDマスクベクトルに変換する反復子
/*!
  \param T	変換先のSIMDマスクベクトルの成分の型
  \param ITER	SIMDマスクベクトルを出力する反復子
*/
template <class T, class ITER>
class cvt_mask_iterator
    : public boost::iterator_adaptor<cvt_mask_iterator<T, ITER>,
				     ITER,
				     pack_target<T, iterator_value<ITER> >,
				     boost::single_pass_traversal_tag,
				     pack_target<T, iterator_value<ITER> > >
{
  private:
    typedef iterator_value<ITER>			src_vec;
    typedef pack_target<T, src_vec>			dst_vec;
    typedef boost::iterator_adaptor<cvt_mask_iterator,
				    ITER,
				    dst_vec,
				    boost::single_pass_traversal_tag,
				    dst_vec>		super;
    typedef typename std::conditional<
		(pack_vec<src_vec>::size <= vec<T>::size),
		src_vec, dst_vec>::type			upmost_vec;
    typedef pack_element<upmost_vec>			upmost_type;
    typedef simd::complementary_mask_type<upmost_type>	complementary_type;
    typedef pack_target<complementary_type, upmost_vec>	complementary_vec;

  public:
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;

    friend class	boost::iterator_core_access;

  public:
		cvt_mask_iterator(const ITER& iter)	:super(iter)	{}

  private:
    upmost_vec	up(const upmost_vec& x) const
		{
		    return x;
		}
    upmost_vec	up(const complementary_vec& x) const
		{
		    return detail::cvtup_mask<upmost_type>(x);
		}
    template <class VEC_>
    upmost_vec	up(const VEC_& x) const
		{
		    typedef simd::upper_type<pack_element<VEC_> > upper_type;
		    
		    return up(detail::cvtup_mask<upper_type>(x));
		}

    void	down(upmost_vec& x)
		{
		    x = *super::base();
		    ++super::base_reference();
		}
    void	down(complementary_vec& x)
		{
		    upmost_vec	y;
		    down(y);
		    x = cvt_mask<complementary_type>(y);
		}
    template <class VEC_>
    void	down(VEC_& x)
		{
		    typedef pack_element<VEC_>			element_type;
		    typedef simd::upper_type<element_type>	upper_type;
		    
		    pack_target<upper_type, upmost_vec>	y, z;
		    down(y);
		    down(z);
		    x = cvt_mask<element_type>(y, z);
		}

    reference	dereference() const
		{
		    return const_cast<cvt_mask_iterator*>(this)
			->dereference(std::is_same<upmost_vec, src_vec>());
		}
    reference	dereference(std::true_type)
		{
		    reference	x;
		    const_cast<cvt_mask_iterator*>(this)->down(x);
		    return x;
		}
    reference	dereference(std::false_type)
		{
		    auto	x = up(*super::base());
		    ++super::base_reference();
		    return x;
		}

    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvt_mask_iterator<T, ITER>
make_cvt_mask_iterator(ITER iter)
{
    return cvt_mask_iterator<T, ITER>(iter);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVT_MASK_ITERATOR_H
