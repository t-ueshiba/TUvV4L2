/*!
  \file		iterator.h
  \author	Toshio UESHIBA
  \brief	各種反復子の定義と実装
*/
#ifndef TU_ITERATOR_H
#define TU_ITERATOR_H

#include <iterator>
#include <functional>			// for std::function
#include <boost/iterator/transform_iterator.hpp>
#include "TU/tuple.h"

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

template <class ITER> inline auto
make_reverse_iterator(ITER iter)
{
    return reverse_iterator<ITER>(iter);
}
#endif
}	// namespace std

namespace TU
{
/************************************************************************
*  type aliases								*
************************************************************************/
//! 反復子が指す型
template <class ITER>
using iterator_value	  = typename std::iterator_traits<ITER>::value_type;

//! 反復子が指す型への参照
template <class ITER>
using iterator_reference  = typename std::iterator_traits<ITER>::reference;

//! 反復子が指す型へのポインタ
template <class ITER>
using iterator_pointer	  = typename std::iterator_traits<ITER>::pointer;
    
//! 2つの反復子間の差を表す型
template <class ITER>
using iterator_difference = typename std::iterator_traits<ITER>
					::difference_type;
//! 反復子のカテゴリ
template <class ITER>
using iterator_category	  = typename std::iterator_traits<ITER>
					::iterator_category;

/************************************************************************
*  class zip_iterator<ITER_TUPLE>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
    // assignment_iterator<FUNC, ITER> のように dereference すると
    // その base iterator への参照を内包する proxy を返す反復子もあるので，
    // 引数は const ITER_& 型にする．もしも ITER_ 型にすると，呼出側から
    // コピーされたローカルな反復子 iter への参照を内包する proxy を
    // 返してしまい，dangling reference が生じる．
      template <class ITER_>
      decltype(auto)	operator ()(const ITER_& iter) const
			{
			    return *iter;
			}
  };
}	// namespace detail
    
template <class ITER_TUPLE>
class zip_iterator : public boost::iterator_facade<
			zip_iterator<ITER_TUPLE>,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			iterator_category<tuple_head<ITER_TUPLE> >,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>()))>
{
  private:
    using super = boost::iterator_facade<
			zip_iterator,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>())),
			iterator_category<tuple_head<ITER_TUPLE> >,
			decltype(tuple_transform(detail::generic_dereference(),
						 std::declval<ITER_TUPLE>()))>;
    friend	class boost::iterator_core_access;
    
  public:
    using	typename super::reference;
    using	typename super::difference_type;
    
  public:
		zip_iterator(ITER_TUPLE iter_tuple)
		    :_iter_tuple(iter_tuple)				{}
    template <class ITER_TUPLE_,
	      std::enable_if_t<
		  std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value>*
	      = nullptr>
		zip_iterator(const zip_iterator<ITER_TUPLE_>& iter)
		    :_iter_tuple(iter.get_iterator_tuple())		{}

    const auto&	get_iterator_tuple()	const	{ return _iter_tuple; }
    
  private:
    reference	dereference() const
		{
		    return tuple_transform(detail::generic_dereference(),
					   _iter_tuple);
		}
    template <class ITER_TUPLE_>
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value, bool>
		equal(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			== std::get<0>(_iter_tuple);
		}
    void	increment()
		{
		    ++_iter_tuple;
		}
    void	decrement()
		{
		    --_iter_tuple;
		}
    void	advance(difference_type n)
		{
		    _iter_tuple += n;
		}
    template <class ITER_TUPLE_>
    std::enable_if_t<std::is_convertible<ITER_TUPLE_, ITER_TUPLE>::value,
		     difference_type>
		distance_to(const zip_iterator<ITER_TUPLE_>& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
		  	 - std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class ITER_TUPLE> inline zip_iterator<ITER_TUPLE>
make_zip_iterator(ITER_TUPLE iter_tuple)
{
    return {iter_tuple};
}

/************************************************************************
*  type alias: decayed_iterator_value<ITER>				*
************************************************************************/
namespace detail
{
  template <class ITER>
  struct decayed_iterator_value
  {
      using type = iterator_value<ITER>;
  };
  template <class... ITER>
  struct decayed_iterator_value<zip_iterator<std::tuple<ITER...> > >
  {
      using type = std::tuple<typename decayed_iterator_value<ITER>::type...>;
  };
}	// namespace detail

//! 反復子が指す型を返す．
/*!
  zip_iterator<ITER_TUPLE>::value_type はITER_TUPLE中の各反復子が指す値への
  参照のtupleの型であるが，decayed_iterator_value<zip_iterator<ITER_TUPLE> >
  は，ITER_TUPLE中の各反復子が指す値そのもののtupleの型を返す．
  \param ITER	反復子
*/
template <class ITER>
using decayed_iterator_value = typename detail::decayed_iterator_value<ITER>
					      ::type;

/************************************************************************
*  TU::[begin|end|rbegin|rend](std::tuple<T...>)			*
************************************************************************/
namespace detail
{
  struct generic_begin
  {
      template <class T>
      auto	operator ()(T&& x) const
		{
		    using	std::begin;
		    return begin(std::forward<T>(x));
		}
  };
  struct generic_end
  {
      template <class T>
      auto	operator ()(T&& x) const
		{
		    using	std::end;
		    return end(std::forward<T>(x));
		}
  };
}
    
template <class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
begin(TUPLE&& t)
{
  // icpc-17.0.4 のバグ対策のため，lambdaを用いずに実装
    return TU::make_zip_iterator(tuple_transform(detail::generic_begin(),
						 std::forward<TUPLE>(t)));
}

template <class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
end(TUPLE&& t)
{
  // icpc-17.0.4 のバグ対策のため，lambdaを用いずに実装
    return TU::make_zip_iterator(tuple_transform(detail::generic_end(),
						 std::forward<TUPLE>(t)));
}

template <class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
rbegin(TUPLE&& t)
{
    return std::make_reverse_iterator(end(std::forward<TUPLE>(t)));
}

template <class TUPLE,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr> inline auto
rend(TUPLE&& t)
{
    return std::make_reverse_iterator(begin(std::forward<TUPLE>(t)));
}

template <class... T> inline auto
cbegin(const std::tuple<T...>& t)
{
    return begin(t);
}

template <class... T> inline auto
cend(const std::tuple<T...>& t)
{
    return end(t);
}

template <class... T> inline auto
crbegin(const std::tuple<T...>& t)
{
    return rbegin(t);
}

template <class... T> inline auto
crend(const std::tuple<T...>& t)
{
    return rend(t);
}

template <class... T> inline auto
size(const std::tuple<T...>& t)
{
    using std::size;
    return size(std::get<0>(t));
}

/************************************************************************
*  make_transform_iterator1<FUNC, ITER>					*
************************************************************************/
template <class FUNC, class ITER> inline boost::transform_iterator<FUNC, ITER>
make_transform_iterator1(const ITER& iter, FUNC func)
{
    return {iter, func};
}

/************************************************************************
*  make_mbr_iterator<ITER, T>						*
************************************************************************/
//! T型のメンバ変数を持つオブジェクトへの反復子からそのメンバに直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
  \param mbr	iterが指すオブジェクトのメンバへのポインタ
*/
template <class ITER, class T> inline auto
make_mbr_iterator(const ITER& iter, T iterator_value<ITER>::* mbr)
{
    return make_transform_iterator1(
	       iter,
	       std::function<std::conditional_t<
				 std::is_same<iterator_pointer<ITER>,
					      iterator_value<ITER>*>::value,
				 T&, const T&>(iterator_reference<ITER>)>(
				     std::mem_fn(mbr)));
}

//! std::pairへの反復子からその第1要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_first_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter, &iterator_value<ITER>::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_second_iterator(const ITER& iter)
{
    return make_mbr_iterator(iter, &iterator_value<ITER>::second);
}
    
/************************************************************************
*  transform_iterator2<FUNC, ITER0, ITER1>				*
************************************************************************/
template <class FUNC, class ITER0, class ITER1>
class transform_iterator2
    : public boost::iterator_adaptor<
		 transform_iterator2<FUNC, ITER0, ITER1>,
		 ITER0,
		 std::result_of_t<FUNC(iterator_reference<ITER0>,
				       iterator_reference<ITER1>)>,
		 boost::use_default,
		 std::result_of_t<FUNC(iterator_reference<ITER0>,
				       iterator_reference<ITER1>)> >
{
  private:
    using ref	= std::result_of_t<FUNC(iterator_reference<ITER0>,
					iterator_reference<ITER1>)>;
    using super	= boost::iterator_adaptor<transform_iterator2,
					  ITER0,
					  ref,
					  boost::use_default,
					  ref>;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::reference;
	
  public:
		transform_iterator2(const ITER0& iter0,
				    const ITER1& iter1, FUNC func)
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
make_transform_iterator2(const ITER0& iter0, const ITER1& iter1, FUNC func)
{
    return {iter0, iter1, func};
}
    
/************************************************************************
*  class assignment_iterator<FUNC, ITER>				*
************************************************************************/
//! libTUTools++ のクラスや関数の実装の詳細を収める名前空間
namespace detail
{
  template <class FUNC, class ITER>
  class assignment_proxy
  {
    private:
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(func(*iter, val), std::true_type());
      template <class T_>
      static auto	check_func(ITER iter, const T_& val, FUNC func)
			    -> decltype(*iter = func(val), std::false_type());
      template <class T_>
      using is_binary_func	= decltype(check_func(std::declval<ITER>(),
						      std::declval<T_>(),
						      std::declval<FUNC>()));
      
    public:
      assignment_proxy(const ITER& iter, const FUNC& func)
	  :_iter(iter), _func(func)					{}

      template <class T_>
      std::enable_if_t<is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    _func(*_iter, std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      std::enable_if_t<!is_binary_func<T_>::value, assignment_proxy&>
			operator =(T_&& val)
			{
			    *_iter  = _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator +=(T_&& val)
			{
			    *_iter += _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator -=(T_&& val)
			{
			    *_iter -= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator *=(T_&& val)
			{
			    *_iter *= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator /=(T_&& val)
			{
			    *_iter /= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator &=(T_&& val)
			{
			    *_iter &= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator |=(T_&& val)
			{
			    *_iter |= _func(std::forward<T_>(val));
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator ^=(T_&& val)
			{
			    *_iter ^= _func(std::forward<T_>(val));
			    return *this;
			}

    private:
      const ITER&	_iter;
      const FUNC&	_func;
  };
}

//! operator *()を左辺値として使うときに，この左辺値と右辺値に指定された関数を適用するための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class ITER>
class assignment_iterator
    : public boost::iterator_adaptor<assignment_iterator<FUNC, ITER>,
				     ITER,
				     iterator_value<ITER>,
				     iterator_category<ITER>,
				     detail::assignment_proxy<FUNC, ITER> >
{
  private:
    using super	= boost::iterator_adaptor<
			assignment_iterator,
			ITER,
			iterator_value<ITER>,
			iterator_category<ITER>,
			detail::assignment_proxy<FUNC, ITER> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::reference;
    
  public:
    assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)			{}

    const auto&	functor()			const	{ return _func; }

  private:
    reference	dereference() const
		{
		    return {super::base(), _func};
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
{
    return {iter, func};
}

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
    using argument_type	= iterator_reference<ROW>;
    
  public:
			row2col(size_t col)	:_col(col)		{}
    
    decltype(auto)	operator ()(argument_type row) const
			{
			    return *(std::begin(row) + _col);
			}
    
  private:
    const size_t	_col;	//!< 列を指定するindex
};

/************************************************************************
*  alias vertical_iterator<ROW>						*
************************************************************************/
template <class ROW>
using vertical_iterator = boost::transform_iterator<row2col<ROW>, ROW>;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(const ROW& row, size_t col)
{
    return {row, {col}};
}

/************************************************************************
*  class ring_iterator<ITER>						*
************************************************************************/
//! 2つの反復子によって指定された範囲を循環バッファとしてアクセスする反復子
/*!
  \param ITER	データ列中の要素を指す反復子の型
*/
template <class ITER>
class ring_iterator : public boost::iterator_adaptor<ring_iterator<ITER>, ITER>
{
  private:
    using super	= boost::iterator_adaptor<ring_iterator, ITER>;
    friend	class boost::iterator_core_access;
    template <class>
    friend	class ring_iterator;
    
  public:
    using	typename super::difference_type;
    
  public:
    ring_iterator()
	:super(), _begin(super::base()), _end(super::base()), _d(0)	{}
    
    ring_iterator(ITER begin, ITER end)
	:super(begin),
	 _begin(begin), _end(end), _d(std::distance(_begin, _end))	{}
    template <class ITER_>
    ring_iterator(const ring_iterator<ITER_>& iter)
	:super(iter.base()),
	 _begin(iter._begin), _end(iter._end), _d(iter._d)		{}

    difference_type	position() const
			{
			    return std::distance(_begin, super::base());
			}
    
  private:
    void		advance(difference_type n)
			{
			    n %= _d;
			    difference_type	i = position() + n;
			    if (i >= _d)
				std::advance(super::base_reference(), n - _d);
			    else if (i < 0)
				std::advance(super::base_reference(), n + _d);
			    else
				std::advance(super::base_reference(), n);
			}
    
    void		increment()
			{
			    if (++super::base_reference() == _end)
				super::base_reference() = _begin;
			}

    void		decrement()
			{
			    if (super::base() == _begin)
				super::base_reference() = _end;
			    --super::base_reference();
			}

    difference_type	distance_to(const ring_iterator& iter) const
			{
			    difference_type	n = iter.base() - super::base();
			    return (n > 0 ? n - _d : n);
			}

    template <class ITER_>
    bool		equal(const ring_iterator<ITER_>& iter) const
			{
			    return super::base() == iter.base();
			}
    
  private:
    ITER		_begin;	// 代入を可能にするためconstは付けない
    ITER		_end;	// 同上
    difference_type	_d;	// 同上
};

template <class ITER> ring_iterator<ITER>
make_ring_iterator(ITER begin, ITER end)	{ return {begin, end}; }

}	// namespace TU
#endif	// !TU_ITERATOR_H
