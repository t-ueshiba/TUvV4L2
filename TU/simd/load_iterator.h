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

template <bool ALIGNED=false, class T> inline load_iterator<T, ALIGNED>
make_load_iterator(const T* p)
{
    return {p};
}

template <bool ALIGNED=false, class T> inline load_iterator<T, ALIGNED>
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
