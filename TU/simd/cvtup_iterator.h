/*!
  \file		cvtup_iterator.h
  \author	Toshio UESHIBA
  \brief	より大きな成分を持つSIMDベクトルへの型変換を行う反復子の定義
*/
#if !defined(TU_SIMD_CVTUP_ITERATOR_H)
#define TU_SIMD_CVTUP_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvtup_iterator<ITER, MASK>					*
************************************************************************/
namespace detail
{
  template <class ITER, bool MASK>
  class cvtup_proxy
  {
    public:
      typedef iterator_value<ITER>				value_type;
      typedef cvtup_proxy					self;

    private:
      typedef typename tuple_head<value_type>::element_type	T;

    private:
      template <class OP_, class VEC_>
      typename std::enable_if<(vec<T>::size == tuple_head<VEC_>::size)>::type
		cvtup(const VEC_& x)
		{
		    OP_()(*_iter, cvt<T, false, MASK>(x));
		    ++_iter;
		}
      template <class OP_, class VEC_>
      typename std::enable_if<(vec<T>::size < tuple_head<VEC_>::size)>::type
		cvtup(const VEC_& x)
		{
		    using U = cvt_upper_type<
				 T, typename tuple_head<VEC_>::element_type,
				 MASK>;
		    
		    cvtup<OP_>(cvt<U, false, MASK>(x));
		    cvtup<OP_>(cvt<U, true,  MASK>(x));
		}

    public:
      cvtup_proxy(const ITER& iter) :_iter(const_cast<ITER&>(iter)) {}
	
      template <class VEC_>
      self&	operator =(const VEC_& x)
		{
		    cvtup<assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator +=(const VEC_& x)
		{
		    cvtup<plus_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator -=(const VEC_& x)
		{
		    cvtup<minus_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator *=(const VEC_& x)
		{
		    cvtup<multiplies_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator /=(const VEC_& x)
		{
		    cvtup<divides_assign>(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator %=(const VEC_& x)
		{
		    cvtup<modulus_assign>(x);
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
template <class ITER, bool MASK=false>
class cvtup_iterator
    : public boost::iterator_adaptor<
		 cvtup_iterator<ITER, MASK>,
		 ITER,
		 typename detail::cvtup_proxy<ITER, MASK>::value_type,
		 boost::single_pass_traversal_tag,
		 detail::cvtup_proxy<ITER, MASK> >
{
  private:
    typedef boost::iterator_adaptor<
		cvtup_iterator,
		ITER,
		typename detail::cvtup_proxy<ITER, MASK>::value_type,
		boost::single_pass_traversal_tag,
		detail::cvtup_proxy<ITER, MASK> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
    cvtup_iterator(const ITER& iter)	:super(iter)			{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(const cvtup_iterator& iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

template <class ITER> cvtup_iterator<ITER, false>
make_cvtup_iterator(ITER iter)
{
    return cvtup_iterator<ITER, false>(iter);
}

template <class ITER>
using cvtup_mask_iterator = cvtup_iterator<ITER, true>;
    
template <class ITER> cvtup_mask_iterator<ITER>
make_cvtup_mask_iterator(ITER iter)
{
    return cvtup_mask_iterator<ITER>(iter);
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_CVTUP_ITERATOR_H
