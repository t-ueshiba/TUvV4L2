/*
 *  $Id$
 */
#if !defined(__TU_SIMD_STORE_ITERATOR_H)
#define __TU_SIMD_STORE_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/load_store.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class store_iterator<ITER, ALIGNED>					*
************************************************************************/
namespace detail
{
  template <class ITER, bool ALIGNED=false>
  class store_proxy
  {
    public:
      typedef typename std::iterator_traits<ITER>::value_type	element_type;
      typedef vec<element_type>					value_type;
      typedef store_proxy					self;
	
    public:
      store_proxy(ITER iter)		:_iter(iter)			{}

		operator value_type() const
		{
		    return load<ALIGNED>(_iter);
		}
      self&	operator =(value_type val)
		{
		    store<ALIGNED>(_iter, val);
		    return *this;
		}
      self&	operator +=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) + val);
		}
      self&	operator -=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) - val);
		}
      self&	operator *=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) * val);
		}
      self&	operator /=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) / val);
		}
      self&	operator %=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) % val);
		}
      self&	operator &=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) & val);
		}
      self&	operator |=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) | val);
		}
      self&	operator ^=(value_type val)
		{
		    return operator =(load<ALIGNED>(_iter) ^ val);
		}

    private:
      ITER 	_iter;
  };
}	// namespace detail

//! 反復子が指す書き込み先にSIMDベクトルを書き込む反復子
/*!
  \param ITER		SIMDベクトルの書き込み先を指す反復子の型
  \param ALIGNED	書き込み先アドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class ITER, bool ALIGNED=false>
class store_iterator
    : public boost::iterator_adaptor<
		store_iterator<ITER, ALIGNED>,
		ITER,
		typename detail::store_proxy<ITER, ALIGNED>::value_type,
  // boost::use_default とすると libc++ で std::fill() に適用できない
		typename std::iterator_traits<ITER>::iterator_category,
		detail::store_proxy<ITER, ALIGNED> >
{
  private:
    typedef boost::iterator_adaptor<
		store_iterator,
		ITER,
		typename detail::store_proxy<ITER, ALIGNED>::value_type,
		typename std::iterator_traits<ITER>::iterator_category,
		detail::store_proxy<ITER, ALIGNED> >		super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::reference		reference;
    
    friend class	boost::iterator_core_access;

  public:
    store_iterator(ITER iter)	:super(iter)	{}
    store_iterator(value_type* p)
	:super(reinterpret_cast<ITER>(p))	{}

    value_type		operator ()() const
			{
			    return load<ALIGNED>(super::base());
			}
    
  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference() += n * value_type::size;
			}
    void		increment()
			{
			    super::base_reference() += value_type::size;
			}
    void		decrement()
			{
			    super::base_reference() -= value_type::size;
			}
    difference_type	distance_to(store_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

namespace detail
{
  template <bool ALIGNED>
  struct storer
  {
      template <class ITER_> store_iterator<ITER_, ALIGNED>
      operator ()(const ITER_& iter) const
      {
	  return store_iterator<ITER_, ALIGNED>(iter);
      }
  };
}	// namespace detail

template <class ITER_TUPLE, bool ALIGNED>
class store_iterator<zip_iterator<ITER_TUPLE>, ALIGNED>
    : public zip_iterator<decltype(tuple_transform(std::declval<ITER_TUPLE>(),
						   detail::storer<ALIGNED>()))>
{
  private:
    using super = zip_iterator<decltype(tuple_transform(
					    std::declval<ITER_TUPLE>(),
					    detail::storer<ALIGNED>()))>;

    struct load
    {
	template <class ITER_> auto
	operator ()(const ITER_& iter) const -> decltype(iter())
	{
	    return iter();
	}
    };

  public:
    using base_type	= ITER_TUPLE;
    using value_type	= decltype(tuple_transform(std::declval<super>()
						   .get_iterator_tuple(),
						   load()));
    
  public:
    store_iterator(zip_iterator<ITER_TUPLE> const& iter)
	:super(tuple_transform(iter.get_iterator_tuple(),
			       detail::storer<ALIGNED>()))		{}
    store_iterator(super const& iter)	:super(iter)			{}

    base_type	base() const
		{
		    return tuple_transform(super::get_iterator_tuple(),
					   [](auto iter)
					   { return iter.base(); });
		}
    
    value_type	operator ()() const
		{
		    return tuple_transform(super::get_iterator_tuple(), load());
		}
};

template <bool ALIGNED=false, class ITER> store_iterator<ITER, ALIGNED>
make_store_iterator(ITER iter)
{
    return {iter};
}

template <bool ALIGNED=false, class T> store_iterator<T*, ALIGNED>
make_store_iterator(vec<T>* p)
{
    return {p};
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_STORE_ITERATOR_H
