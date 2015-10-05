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
    // xがcons型のとき cvt_mask<S>(x) の結果もcons型になるので，
    // iterator_value<ITER> がtuple型のときはそれをcons型に直したものを
    // value_typeとしておかないと，cvupの最終ステップで cvtup(value_type)
    // を呼び出せない．
      typedef tuple_replace<iterator_value<ITER> >		value_type;
      typedef typename tuple_head<value_type>::element_type	element_type;
      typedef cvtup_mask_proxy				self;

    private:
      typedef typename std::iterator_traits<ITER>::reference
							reference;
      typedef typename type_traits<element_type>::complementary_mask_type
							complementary_type;
      typedef tuple_replace<value_type, vec<complementary_type> >
							complementary_vec;
      typedef typename std::conditional<
	  std::is_floating_point<element_type>::value,
	  complementary_type, element_type>::type	integral_type;
      typedef typename std::conditional<
	  std::is_signed<integral_type>::value,
	  typename type_traits<integral_type>::unsigned_type,
	  typename type_traits<integral_type>::signed_type>::type
							flipped_type;
      typedef tuple_replace<value_type, vec<flipped_type> >
							flipped_vec;
      typedef typename type_traits<flipped_type>::lower_type
							flipped_lower_type;
      typedef tuple_replace<value_type, vec<flipped_lower_type> >
							flipped_lower_vec;
	
    private:
      template <class OP_>
      void	cvtup(value_type x)
		{
		    OP_()(*_iter, x);
		    ++_iter;
		}
      template <class OP_>
      void	cvtup(complementary_vec x)
		{
		    cvtup<OP_>(cvt_mask<element_type>(x));
		}
      template <class OP_>
      void	cvtup(flipped_vec x)
		{
		    cvtup<OP_>(cvt_mask<integral_type>(x));
		}
      template <class OP_>
      void	cvtup(flipped_lower_vec x)
		{
		    cvtup<OP_>(cvt_mask<integral_type, 0>(x));
		    cvtup<OP_>(cvt_mask<integral_type, 1>(x));
		}
      template <class OP_, class VEC_>
      void	cvtup(VEC_ x)
		{
		    typedef
			typename tuple_head<VEC_>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;

		    cvtup<OP_>(cvt_mask<upper_type, 0>(x));
		    cvtup<OP_>(cvt_mask<upper_type, 1>(x));
		}

    public:
      cvtup_mask_proxy(ITER const& iter) :_iter(const_cast<ITER&>(iter)) {}
	
      template <class VEC_>
      self&	operator =(VEC_ x)
		{
		    cvtup<assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator &=(VEC_ x)
		{
		    cvtup<bit_and_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator |=(VEC_ x)
		{
		    cvtup<bit_or_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator ^=(VEC_ x)
		{
		    cvtup<bit_xor_assign<reference, value_type> >(x);
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
    cvtup_mask_iterator(ITER const& iter)	:super(iter)		{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(cvtup_mask_iterator const& iter) const
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
