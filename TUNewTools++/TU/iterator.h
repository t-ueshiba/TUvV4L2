/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TU_ITERATOR_H
#define __TU_ITERATOR_H

#include <cstddef>			// for size_t
#include <type_traits>			// for std::result_of<F(ARGS...)>
#include <utility>			// for std::declval<T>
#include <iterator>
#include <boost/iterator/transform_iterator.hpp>

namespace std
{
#if __cplusplus < 201700L
/************************************************************************
*  function std::size(T)						*
************************************************************************/
template <class T> inline size_t
size(const T& x)
{
    return x.size();
}
template <class T, size_t N> inline constexpr size_t
size(const T (&array)[N]) noexcept
{
    return N;
}
#endif

#if __cplusplus < 201402L    
/************************************************************************
*  std::[rbegin|rend|cbegin|cend|crbegin|crend](T)			*
************************************************************************/
template <class T> inline auto
rbegin(const T& x) -> decltype(x.rbegin())
{
    return x.rbegin();
}
    
template <class T> inline auto
rbegin(T& x) -> decltype(x.rbegin())
{
    return x.rbegin();
}
    
template <class T> inline auto
rend(const T& x) -> decltype(x.rend())
{
    return x.rend();
}
    
template <class T> inline auto
rend(T& x) -> decltype(x.rend())
{
    return x.rend();
}
    
template <class T> inline auto
cbegin(const T& x) -> decltype(std::begin(x))
{
    return std::begin(x);
}
    
template <class T> inline auto
cend(const T& x) -> decltype(std::end(x))
{
    return std::end(x);
}

template <class T> inline auto
crbegin(const T& x) -> decltype(std::rbegin(x))
{
    return std::rbegin(x);
}
    
template <class T> inline auto
crend(const T& x) -> decltype(std::rend(x))
{
    return std::rend(x);
}
#endif
}	// namespace std

//! libTUTools++ のクラスや関数等を収める名前空間
namespace TU
{
/************************************************************************
*  make_mbr_iterator<ITER, T>						*
************************************************************************/
//! T型のメンバ変数を持つオブジェクトへの反復子からそのメンバに直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
  \param mbr	iterが指すオブジェクトのメンバへのポインタ
*/
template <class ITER, class T> inline auto
make_mbr_iterator(const ITER& iter,
		  T std::iterator_traits<ITER>::value_type::* mbr)
{
    return boost::make_transform_iterator(
	       iter,
	       std::function<
		   typename std::conditional<
		       std::is_same<
			   typename std::iterator_traits<ITER>::pointer,
			   typename std::iterator_traits<ITER>::value_type*>
			   ::value,
	           T&, const T&>::type
	       (typename std::iterator_traits<ITER>::reference)>(
		   std::mem_fn(mbr)));
}

//! std::pairへの反復子からその第1要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_first_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter,
			     &std::iterator_traits<ITER>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_second_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter,
			     &std::iterator_traits<ITER>::value_type::second);
}
    
/************************************************************************
*  transform_iterator2<FUNC, ITER0, ITER1>				*
************************************************************************/
template <class FUNC, class ITER0, class ITER1>
class transform_iterator2
    : public boost::iterator_adaptor<
		 transform_iterator2<FUNC, ITER0, ITER1>,
		 ITER0,
		 typename std::result_of<FUNC(
		     typename std::iterator_traits<ITER0>::reference,
		     typename std::iterator_traits<ITER1>::reference)>::type,
		 boost::use_default,
		 typename std::result_of<FUNC(
		     typename std::iterator_traits<ITER0>::reference,
		     typename std::iterator_traits<ITER1>::reference)>::type>
{
  private:
    using ref	= typename std::result_of<FUNC(
		      typename std::iterator_traits<ITER0>::reference,
		      typename std::iterator_traits<ITER1>::reference)>::type;
    using super	= boost::iterator_adaptor<transform_iterator2,
					  ITER0,
					  ref,
					  boost::use_default,
					  ref>;

  public:
    using	typename super::difference_type;
    using	typename super::reference;

    friend	class boost::iterator_core_access;
	
  public:
		transform_iterator2(ITER0 iter0, ITER1 iter1, FUNC func)
		    :super(iter0), _iter(iter1), _func(func)
		{
		}
	
  private:
    reference	dereference() const
		{
		    return _func(*super::base(), *_iter);
		}
    void	advance(difference_type n)
		{
		    super::base_reference() += n;
		    _iter += n;
		}
    void	increment()
		{
		    ++super::base_reference();
		    ++_iter;
		}
    void	decrement()
		{
		    --super::base_reference();
		    --_iter;
		}
	
  private:
    ITER1	_iter;	//!< 第2引数となる式の実体を指す反復子
    FUNC	_func;	//!< 2項演算子
};

template <class FUNC, class ITER0, class ITER1>
inline transform_iterator2<FUNC, ITER0, ITER1>
make_transform_iterator2(ITER0 iter0, ITER1 iter1, FUNC func)
{
    return {iter0, iter1, func};
}
    
/************************************************************************
*  alias subiterator<ITER>						*
************************************************************************/
template <class ITER>
using subiterator	= decltype(std::begin(*std::declval<ITER>()));

/************************************************************************
*  class row2col<ROW>							*
************************************************************************/
//! 行への参照を与えられると予め指定された列indexに対応する要素への参照を返す関数オブジェクト
/*!
  \param ROW	行を指す反復子
*/ 
template <class ROW>
class row2col
{
  public:
    using argument_type	= typename std::iterator_traits<ROW>::reference;
    using result_type	= typename std::iterator_traits<subiterator<ROW> >
				      ::reference;
    
  public:
		row2col(size_t col)	:_col(col)			{}
    
    result_type	operator ()(argument_type row) const
		{
		    return *(std::begin(row) + _col);
		}
    
  private:
    size_t	_col;	//!< 列を指定するindex
};

/************************************************************************
*  alias vertical_iterator<ROW>						*
************************************************************************/
template <class ROW>
using vertical_iterator = boost::transform_iterator<
			      row2col<ROW>, ROW,
			      boost::use_default,
			      typename std::iterator_traits<subiterator<ROW> >
					  ::value_type>;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(ROW row, size_t col)
{
    return {row, {col}};
}

}	// namespace TU
#endif	// !__TU_ITERATOR_H
