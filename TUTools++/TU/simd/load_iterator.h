/*
 *  $Id$
 */
#if !defined(TU_SIMD_LOAD_ITERATOR_H)
#define TU_SIMD_LOAD_ITERATOR_H

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
    using element_type	= iterator_value<ITER>;
    using super		= boost::iterator_adaptor<load_iterator,
						  ITER,
						  vec<element_type>,
						  boost::use_default,
						  vec<element_type> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::value_type;
    using	typename super::reference;
    
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

template <bool ALIGNED=false, class ITER> inline load_iterator<ITER, ALIGNED>
make_load_iterator(ITER iter)
{
    return {iter};
}

template <bool ALIGNED=false, class T> inline load_iterator<const T*, ALIGNED>
make_load_iterator(const vec<T>* p)
{
    return {p};
}

template <bool ALIGNED=false, class ITER_TUPLE> inline auto
make_load_iterator(zip_iterator<ITER_TUPLE> zip_iter)
{
    return make_zip_iterator(
	       tuple_transform([](auto iter)
			       { return make_load_iterator<ALIGNED>(iter); },
			       zip_iter.get_iterator_tuple()));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_LOAD_ITERATOR_H
