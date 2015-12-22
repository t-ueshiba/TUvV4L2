/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVTUP_MASK_ITERATOR_H)
#define __TU_SIMD_CVTUP_MASK_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt_mask.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvtup_mask_iterator<ITER>					*
************************************************************************/
namespace detail
{
  template <class ITER>
  class cvtup_mask_proxy
  {
    public:
      typedef iterator_value<ITER>				value_type;
      typedef typename tuple_head<value_type>::element_type	element_type;
      typedef cvtup_mask_proxy					self;

    private:
      template <class OP_, class VEC_>
      typename std::enable_if<(tuple_head<value_type>::size ==
			       tuple_head<VEC_>::size)>::type
		cvtup(const VEC_& x)
		{
		    OP_()(*_iter, cvt_mask<element_type>(x));
		    ++_iter;
		}
      template <class OP_, class VEC_>
      typename std::enable_if<(tuple_head<value_type>::size <
			       tuple_head<VEC_>::size)>::type
		cvtup(const VEC_& x)
		{
		    using U = cvt_mask_upper_type<
				  element_type,
				  typename tuple_head<VEC_>::element_type>;

		    cvtup<OP_>(cvt_mask<U, 0>(x));
		    cvtup<OP_>(cvt_mask<U, 1>(x));
		}

    public:
      cvtup_mask_proxy(ITER const& iter) :_iter(const_cast<ITER&>(iter)) {}
	
      template <class VEC_>
      self&	operator =(const VEC_& x)
		{
		    cvtup<assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator &=(const VEC_& x)
		{
		    cvtup<bit_and_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator |=(const VEC_& x)
		{
		    cvtup<bit_or_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator ^=(const VEC_& x)
		{
		    cvtup<bit_xor_assign>(x);
		    return *this;
		}
	
    private:
      ITER&	_iter;
  };
}

//! SIMDベクトルを受け取ってより大きな成分を持つ複数のSIMDベクトルに変換し，それらを指定された反復子を介して書き込む反復子
/*!
  \param ITER	変換されたSIMDベクトルの書き込み先を指す反復子
*/
template <class ITER>
class cvtup_mask_iterator
    : public boost::iterator_adaptor<
		 cvtup_mask_iterator<ITER>,
		 ITER,
		 typename detail::cvtup_mask_proxy<ITER>::value_type,
		 boost::single_pass_traversal_tag,
		 detail::cvtup_mask_proxy<ITER> >
{
  private:
    typedef boost::iterator_adaptor<
		cvtup_mask_iterator,
		ITER,
		typename detail::cvtup_mask_proxy<ITER>::value_type,
		boost::single_pass_traversal_tag,
		detail::cvtup_mask_proxy<ITER> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
    cvtup_mask_iterator(const ITER& iter)	:super(iter)		{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(const cvtup_mask_iterator& iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

template <class ITER> cvtup_mask_iterator<ITER>
make_cvtup_mask_iterator(ITER iter)
{
    return cvtup_mask_iterator<ITER>(iter);
}

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVTUP_MASK_ITERATOR_H
