/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVT_ITERATOR_H)
#define __TU_SIMD_CVT_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/pack.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvt_iterator<T, ITER, MASK>					*
************************************************************************/
//! SIMDベクトルを出力する反復子を介して1つまたは複数のSIMDベクトルを読み込み，指定された成分を持つSIMDベクトルに変換する反復子
/*!
  \param S	変換先のSIMDベクトルの成分の型
  \param ITER	SIMDベクトルを出力する反復子
*/
template <class T, class ITER, bool MASK=false>
class cvt_iterator
    : public boost::iterator_adaptor<cvt_iterator<T, ITER, MASK>,
				     ITER,
				     pack_target<T, iterator_value<ITER> >,
				     boost::single_pass_traversal_tag,
				     pack_target<T, iterator_value<ITER> > >
{
  private:
    using src_type	= iterator_value<ITER>;
    using dst_type	= pack_target<T, src_type>;
    using super		= boost::iterator_adaptor<
			      cvt_iterator,
			      ITER,
			      dst_type,
			      boost::single_pass_traversal_tag,
			      dst_type>;

  public:
    using		typename super::difference_type;
    using		typename super::reference;

    friend class	boost::iterator_core_access;

  public:
		cvt_iterator(const ITER& iter)	:super(iter)	{}
    
  private:
    template <class T_>
    std::enable_if_t<(vec<T_>::size == src_type::size), vec<T_> >
		cvtdown()
		{
		    auto	x = *super::base();
		    ++super::base_reference();
		    return cvt<T_, false, MASK>(x);
		}
    template <class T_>
    std::enable_if_t<(vec<T_>::size > src_type::size), vec<T_> >
		cvtdown()
		{
		    using A = cvt_above_type<
				  T_ , typename src_type::element_type, MASK>;
	  
		    auto	x = cvtdown<A>();
		    auto	y = cvtdown<A>();
		    return cvt<T_, MASK>(x, y);
		}

    reference	dereference() const
		{
		    return const_cast<cvt_iterator*>(this)
			->dereference(
			    std::integral_constant<
				bool, (vec<T>::size > src_type::size)>());
		}
    reference	dereference(std::true_type)
		{
		    return cvtdown<T>();
		}
    reference	dereference(std::false_type)
		{
		    auto	x = cvt_pack<T, MASK>(*super::base());
		    ++super::base_reference();
		    return x;
		}

    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}
};
    
template <class T, class ITER> cvt_iterator<T, ITER>
make_cvt_iterator(ITER iter)
{
    return cvt_iterator<T, ITER>(iter);
}

template <class T, class ITER>
using cvt_mask_iterator = cvt_iterator<T, ITER, true>;
    
template <class T, class ITER> cvt_mask_iterator<T, ITER>
make_cvt_mask_iterator(ITER iter)
{
    return cvt_mask_iterator<T, ITER>(iter);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVT_ITERATOR_H
