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
class load_iterator
    : public boost::iterator_adaptor<load_iterator<ITER, ALIGNED>,
				     ITER,
				     vec<iterator_value<ITER> >,
				     boost::use_default,
				     vec<iterator_value<ITER> > >
{
  private:
    typedef iterator_value<ITER>				element_type;
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
class load_iterator<zip_iterator<ITER_TUPLE>, ALIGNED>
    : public zip_iterator<decltype(tuple_transform(std::declval<ITER_TUPLE>(),
						   detail::loader<ALIGNED>()))>
{
  private:
    using super = zip_iterator<decltype(tuple_transform(
					    std::declval<ITER_TUPLE>(),
					    detail::loader<ALIGNED>()))>;

  public:
    using base_type = ITER_TUPLE;
    
  public:
    load_iterator(const zip_iterator<ITER_TUPLE>& iter)
	:super(tuple_transform(iter.get_iterator_tuple(),
			       detail::loader<ALIGNED>()))		{}
    load_iterator(const super& iter)	:super(iter)			{}

    base_type	base() const
		{
		    return tuple_transform(super::get_iterator_tuple(),
					   [](auto iter)
					   { return iter.base(); });
		}
};

template <bool ALIGNED=false, class ITER> load_iterator<ITER, ALIGNED>
make_load_iterator(ITER iter)
{
    return {iter};
}

template <bool ALIGNED=false, class T> load_iterator<const T*, ALIGNED>
make_load_iterator(const vec<T>* p)
{
    return {p};
}
    
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_LOAD_ITERATOR_H
