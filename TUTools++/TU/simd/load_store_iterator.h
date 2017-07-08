/*
 *  $Id$
 */
#if !defined(TU_SIMD_LOAD_STORE_ITERATOR_H)
#define TU_SIMD_LOAD_STORE_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/load_store.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class load_iterator<T, ALIGNED>					*
************************************************************************/
//! 反復子が指すアドレスからSIMDベクトルを読み込む反復子
/*!
  \param T		SIMDベクトルの成分の型
  \param ALIGNED	読み込み元のアドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class T, bool ALIGNED=false>
class load_iterator
    : public boost::iterator_adaptor<load_iterator<T, ALIGNED>,
				     const T*,
				     vec<T>,
				     boost::use_default,
				     vec<T> >
{
  private:
    using element_type	= T;
    using super		= boost::iterator_adaptor<load_iterator,
						  const T*,
						  vec<T>,
						  boost::use_default,
						  vec<T> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::value_type;
    using	typename super::reference;
    
  public:
    load_iterator(const T* p=nullptr)	:super(p)	{}
    load_iterator(const value_type* p)
	:super(reinterpret_cast<const T*>(p))		{}
	       
  private:
    reference		dereference() const
			{
			    return load<ALIGNED>(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference()
				+= n * difference_type(value_type::size);
			}
    void		increment()
			{
			    super::base_reference()
				+= difference_type(value_type::size);
			}
    void		decrement()
			{
			    super::base_reference()
				-= difference_type(value_type::size);
			}
    difference_type	distance_to(load_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / difference_type(value_type::size);
			}
};

template <bool ALIGNED=false, class T> inline load_iterator<T, false>
make_accessor(const T* p)
{
    return {p};
}

template <class T> inline load_iterator<T, true>
make_accessor(ptr<const T> p)
{
    return {p};
}
    
/************************************************************************
*  class store_iterator<T, ALIGNED>					*
************************************************************************/
namespace detail
{
  template <class T, bool ALIGNED=false>
  class store_proxy
  {
    public:
      using element_type = T;
      using value_type	 = decltype(load<ALIGNED>(std::declval<const T*>()));
      
	
    public:
      store_proxy(T* p)		:_p(p)			{}

			operator value_type() const
			{
			    return load<ALIGNED>(_p);
			}
      store_proxy&	operator =(value_type val)
			{
			    store<ALIGNED>(_p, val);
			    return *this;
			}
      store_proxy&	operator +=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) + val);
			}
      store_proxy&	operator -=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) - val);
			}
      store_proxy&	operator *=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) * val);
			}
      store_proxy&	operator /=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) / val);
			}
      store_proxy&	operator %=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) % val);
			}
      store_proxy&	operator &=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) & val);
			}
      store_proxy&	operator |=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) | val);
			}
      store_proxy&	operator ^=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) ^ val);
			}

    private:
      T* 	_p;
  };
}	// namespace detail

//! 反復子が指す書き込み先にSIMDベクトルを書き込む反復子
/*!
  \param T		SIMDベクトルの成分の型
  \param ALIGNED	書き込み先アドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class T, bool ALIGNED=false>
class store_iterator
    : public boost::iterator_adaptor<
		store_iterator<T, ALIGNED>,
		T*,
		typename detail::store_proxy<T, ALIGNED>::value_type,
  // boost::use_default とすると libc++ で std::fill() に適用できない
		iterator_category<T*>,
		detail::store_proxy<T, ALIGNED> >
{
  private:
    using super	= boost::iterator_adaptor<
		      store_iterator,
		      T*,
		      typename detail::store_proxy<T, ALIGNED>::value_type,
		      iterator_category<T*>,
		      detail::store_proxy<T, ALIGNED> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::value_type;
    using	typename super::reference;
    
  public:
    store_iterator(T* p=nullptr)	:super(p)	{}
    store_iterator(value_type* p)
	:super(reinterpret_cast<T*>(p))			{}

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
			    super::base_reference()
				+= n * difference_type(value_type::size);
			}
    void		increment()
			{
			    super::base_reference()
				+= difference_type(value_type::size);
			}
    void		decrement()
			{
			    super::base_reference()
				-= difference_type(value_type::size);
			}
    difference_type	distance_to(store_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / diffence_type(value_type::size);
			}
};

template <class T> inline store_iterator<T, false>
make_accessor(T* p)
{
    return {p};
}

template <class T> inline store_iterator<T, true>
make_accessor(ptr<T> p)
{
    return {p};
}
    
template <class ITER_TUPLE> inline auto
make_accessor(zip_iterator<ITER_TUPLE> zip_iter)
{
    return make_zip_iterator(tuple_transform([](auto iter)
					     { return make_accessor(iter); },
					     zip_iter.get_iterator_tuple()));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_LOAD_STORE_ITERATOR_H
