/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ROW_VEC_ITERATOR_H)
#define __TU_SIMD_ROW_VEC_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/vec.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  struct htuple2vec<T>							*
************************************************************************/
//! 同じ型の成分を持つtupleをSIMDベクトルに変換する関数オブジェクト
/*!
  \param T	成分の型
*/
template <class T>
struct htuple2vec
{
    typedef vec<T>	result_type;

    template <class... T_>
    std::enable_if_t<sizeof...(T_) == vec<T>::size, result_type>
		operator ()(const std::tuple<T_...>& t) const
		{
		    return exec(t, std::make_index_sequence<vec<T>::size>());
		}

  private:
    template <class TUPLE, size_t ...IDX>
    result_type	exec(const TUPLE& t, std::index_sequence<IDX...>) const
		{
		    return result_type(std::get<IDX>(t)...);
		}
};

/************************************************************************
*  class row_vec_iterator<T, ROW>					*
************************************************************************/
template <class T, class ROW>
class row_vec_iterator
    : public boost::iterator_adaptor<
		 row_vec_iterator<T, ROW>,
		 row_iterator<boost::transform_iterator<
				  htuple2vec<T>,
				  subiterator<fast_zip_iterator<
						  boost::htuple<
						      ROW, vec<T>::size> > > >,
			      fast_zip_iterator<
				  boost::htuple<ROW, vec<T>::size> >,
			      htuple2vec<T> > >
{
  private:
    typedef fast_zip_iterator<boost::htuple<ROW, vec<T>::size> >
							row_zip_iterator;
    typedef boost::iterator_adaptor<
		row_vec_iterator,
		row_iterator<
		    boost::transform_iterator<
			htuple2vec<T>,
			subiterator<row_zip_iterator> >,
		    row_zip_iterator,
		    htuple2vec<T> > >			super;

  public:
    typedef typename super::difference_type		difference_type;

    friend class	boost::iterator_core_access;
    
  public:
    row_vec_iterator(const ROW& row)
	:super(make_row_transform_iterator(
		   make_fast_zip_iterator(
		       boost::make_contiguous_htuple<vec<T>::size>(row)),
		   htuple2vec<T>()))					{}

    void		advance(difference_type n)
			{
			    super::base_reference() += n * vec<T>::size;
			}
    void		increment()
			{
			    super::base_reference() += vec<T>::size;
			}
    void		decrement()
			{
			    super::base_reference() -= vec<T>::size;
			}
    difference_type	distance_to(row_vec_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / vec<T>::size;
			}
};

template <class T, class ROW> inline row_vec_iterator<T, ROW>
make_row_vec_iterator(const ROW& row)
{
    return row_vec_iterator<T, ROW>(row);
}

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ROW_VEC_ITERATOR_H
