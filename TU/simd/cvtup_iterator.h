/*
 *  $Id$
 */
#if !defined(__TU_SIMD_CVTUP_ITERATOR_H)
#define __TU_SIMD_CVTUP_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class cvtup_iterator<ITER>						*
************************************************************************/
namespace detail
{
  template <class ITER>
  class cvtup_proxy
  {
    public:
    // xがcons型のとき cvt_mask<S>(x) の結果もcons型になるので，
    // iterator_value<ITER> がtuple型のときはそれをcons型に直したものを
    // value_typeとしておかないと，cvupの最終ステップで cvtup(value_type)
    // を呼び出せない．
      typedef tuple_replace<iterator_value<ITER> >		value_type;
      typedef typename tuple_head<value_type>::element_type	element_type;
      typedef cvtup_proxy					self;

    private:
      typedef typename std::iterator_traits<ITER>::reference
							reference;
      typedef simd::complementary_type<element_type>	complementary_type;
      typedef tuple_replace<value_type, vec<complementary_type> >
							complementary_vec;
      typedef typename std::conditional<
		  std::is_floating_point<element_type>::value,
		  complementary_type,
		  element_type>::type			integral_type;
      typedef simd::unsigned_type<simd::lower_type<integral_type> >
							unsigned_lower_type;
      typedef tuple_replace<value_type, vec<unsigned_lower_type> >
							unsigned_lower_vec;
	
    private:
      template <class OP_>
      void	cvtup(value_type x)
		{
		    OP_()(*_iter, x);
		    ++_iter;
		}
      template <class OP_>
      void	cvtup(unsigned_lower_vec x)
		{
		    cvtup<OP_>(cvt<integral_type, 0>(x));
		    cvtup<OP_>(cvt<integral_type, 1>(x));
		}
      template <class OP_>
      void	cvtup(complementary_vec x)
		{
		    cvtup<OP_>(x,
			       std::integral_constant<
				   bool, (vec<complementary_type>::size ==
					  vec<element_type>::size)>());
		}
      template <class OP_>
      void	cvtup(complementary_vec x, std::true_type)
		{
		    cvtup<OP_>(cvt<element_type>(x));
		}
      template <class OP_>
      void	cvtup(complementary_vec x, std::false_type)
		{
		    cvtup<OP_>(cvt<element_type, 0>(x));
		    cvtup<OP_>(cvt<element_type, 1>(x));
		}
      template <class OP_, class VEC_>
      void	cvtup(VEC_ x)
		{
		    typedef upper_type<
			typename tuple_head<VEC_>::element_type> upper_type;

		    cvtup<OP_>(cvt<upper_type, 0>(x));
		    cvtup<OP_>(cvt<upper_type, 1>(x));
		}

    public:
      cvtup_proxy(const ITER& iter) :_iter(const_cast<ITER&>(iter)) {}
	
      template <class VEC_>
      self&	operator =(VEC_ x)
		{
		    cvtup<assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator +=(VEC_ x)
		{
		    cvtup<plus_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator -=(VEC_ x)
		{
		    cvtup<minus_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator *=(VEC_ x)
		{
		    cvtup<multiplies_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator /=(VEC_ x)
		{
		    cvtup<divides_assign<reference, value_type> >(x);
		    return *this;
		}
      template <class VEC_>
      self&	operator %=(VEC_ x)
		{
		    cvtup<modulus_assign<reference, value_type> >(x);
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
class cvtup_iterator
    : public boost::iterator_adaptor<
		 cvtup_iterator<ITER>,
		 ITER,
		 typename detail::cvtup_proxy<ITER>::value_type,
		 boost::single_pass_traversal_tag,
		 detail::cvtup_proxy<ITER> >
{
  private:
    typedef boost::iterator_adaptor<
		cvtup_iterator,
		ITER,
		typename detail::cvtup_proxy<ITER>::value_type,
		boost::single_pass_traversal_tag,
		detail::cvtup_proxy<ITER> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::reference		reference;

    friend class	boost::iterator_core_access;

  public:
    cvtup_iterator(ITER const& iter)	:super(iter)			{}

  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type)			{}
    void		increment()					{}
    void		decrement()					{}
    difference_type	distance_to(cvtup_iterator const& iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

template <class ITER> cvtup_iterator<ITER>
make_cvtup_iterator(ITER iter)
{
    return cvtup_iterator<ITER>(iter);
}

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_CVTUP_ITERATOR_H
