/*
 *  $Id$
 */
#if !defined(__TU_SIMD_LOAD_ITERATOR_H)
#define __TU_SIMD_LOAD_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/load_store.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class load_iterator<ITER, ALIGNED>					*
************************************************************************/
//! 反復子が指すアドレスからSIMDベクトルを読み込む反復子
/*!
  \param ITER		SIMDベクトルの読み込み元を指す反復子の型
  \param ALIGNED	読み込み元のアドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class ITER, bool ALIGNED=false>
class load_iterator : public boost::iterator_adaptor<
			load_iterator<ITER, ALIGNED>,
			ITER,
			vec<typename std::iterator_traits<ITER>::value_type>,
			boost::use_default,
			vec<typename std::iterator_traits<ITER>::value_type> >
{
  private:
    typedef typename std::iterator_traits<ITER>::value_type	element_type;
    typedef boost::iterator_adaptor<load_iterator,
				    ITER,
				    vec<element_type>,
				    boost::use_default,
				    vec<element_type> >		super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::reference		reference;
    
    friend class	boost::iterator_core_access;

    
  public:
    load_iterator(ITER iter)	:super(iter)	{}
    load_iterator(const value_type* p)
	:super(reinterpret_cast<ITER>(p))	{}
	       
  private:
    reference		dereference() const
			{
			    return load<ALIGNED>(super::base());
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
    difference_type	distance_to(load_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / value_type::size;
			}
};

namespace detail
{
  template <bool ALIGNED>
  struct loader
  {
      template <class ITER_> load_iterator<ITER_, ALIGNED>
      operator ()(const ITER_& iter) const
      {
	  return load_iterator<ITER_, ALIGNED>(iter);
      }
  };
}	// namespace detail
    
template <class ITER_TUPLE, bool ALIGNED>
class load_iterator<fast_zip_iterator<ITER_TUPLE>, ALIGNED>
    : public fast_zip_iterator<decltype(boost::tuples::cons_transform(
					    detail::loader<ALIGNED>(),
					    std::declval<ITER_TUPLE>()))>
{
  private:
    typedef fast_zip_iterator<decltype(boost::tuples::cons_transform(
					   detail::loader<ALIGNED>(),
					   std::declval<ITER_TUPLE>()))> super;

    struct base_iterator
    {
	template <class ITER_> auto
	operator ()(const ITER_& iter) const -> decltype(iter.base())
	{
	    return iter.base();
	}
    };

  public:
    typedef ITER_TUPLE	base_type;
    
  public:
    load_iterator(const fast_zip_iterator<ITER_TUPLE>& iter)
	:super(boost::tuples::cons_transform(detail::loader<ALIGNED>(),
					     iter.get_iterator_tuple())){}
    load_iterator(const super& iter)	:super(iter)			{}

    base_type	base() const
		{
		    return boost::tuples::cons_transform(
			       base_iterator(), super::get_iterator_tuple());
		}
};

template <bool ALIGNED=false, class ITER> load_iterator<ITER, ALIGNED>
make_load_iterator(ITER iter)
{
    return load_iterator<ITER, ALIGNED>(iter);
}

template <bool ALIGNED=false, class T> load_iterator<const T*, ALIGNED>
make_load_iterator(const vec<T>* p)
{
    return load_iterator<const T*, ALIGNED>(p);
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_LOAD_ITERATOR_H
