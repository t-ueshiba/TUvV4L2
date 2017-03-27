/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TU_ITERATOR_H
#define __TU_ITERATOR_H

#include <cstddef>			// for size_t
#include <type_traits>			// for std::result_of<F(ARGS...)>
#include <utility>			// for std::declval<T>
#include <functional>			// for std::function
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
*  type aliases								*
************************************************************************/
template <class ITER>
using iterator_value	  = typename std::iterator_traits<ITER>::value_type;
template <class ITER>
using iterator_reference  = typename std::iterator_traits<ITER>::reference;
template <class ITER>
using iterator_pointer	  = typename std::iterator_traits<ITER>::pointer;
template <class ITER>
using iterator_difference = typename std::iterator_traits<ITER>
					::difference_type;
template <class ITER>
using iterator_category	  = typename std::iterator_traits<ITER>
					::iterator_category;

//! libTUTools++ のクラスや関数の実装の詳細を収める名前空間
namespace detail
{
  template <class E>
  auto	check_begin(E&& x) -> decltype(std::begin(x))			;
  void	check_begin(...)						;
}

//! 式が持つ反復子の型を返す
/*!
  \param E	式の型
  \return	E が反復子を持てばその型，持たなければ void
*/
template <class E>
using iterator_t = decltype(detail::check_begin(std::declval<E>()));

//! 式が反復子を持つか判定する
template <class E>
using has_begin	= std::integral_constant<bool,
					 !std::is_void<iterator_t<E> >::value>;

//! 式が持つ反復子が指す型を返す
/*!
  反復子を持たない式を与えるとコンパイルエラーとなる.
  \param E	反復子を持つ式の型
  \return	E の反復子が指す型
*/
template <class E>
using value_t	= iterator_value<iterator_t<E> >;

namespace detail
{
  template <class T>
  struct identity
  {
      using type = T;
  };

  template <class E, class=iterator_t<E> >
  struct element_t
  {
      using type = typename element_t<value_t<E> >::type;
  };
  template <class E>
  struct element_t<E, void> : identity<E>				{};
}	// namespace detail
    
//! 式が持つ反復子が指す型を再帰的に辿って到達する型を返す
/*!
  \param E	式の型
  \return	E が反復子を持てばそれが指す型を再帰的に辿って到達する型，
		持たなければ E 自身
*/
template <class E>
using element_t	= typename detail::element_t<E>::type;

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
    return boost::make_transform_iterator(
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
		 typename std::result_of<FUNC(iterator_reference<ITER0>,
					      iterator_reference<ITER1>)>::type,
		 boost::use_default,
		 typename std::result_of<FUNC(iterator_reference<ITER0>,
					      iterator_reference<ITER1>)>::type>
{
  private:
    using ref	= typename std::result_of<FUNC(iterator_reference<ITER0>,
					       iterator_reference<ITER1>)>::type;
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
*  class assignment_iterator<FUNC, ITER, VAL>				*
************************************************************************/
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
      using	is_binary_func = decltype(check_func(std::declval<ITER>(),
						     std::declval<T_>(),
						     std::declval<FUNC>()));
      
    public:
      assignment_proxy(const ITER& iter, const FUNC& func)
	  :_iter(iter), _func(func)					{}

      template <class T_>
      std::enable_if_t<is_binary_func<T_>::value, assignment_proxy&>
			operator =(const T_& val)
			{
			    _func(*_iter, val);
			    return *this;
			}
      template <class T_>
      std::enable_if_t<!is_binary_func<T_>::value, assignment_proxy&>
			operator =(const T_& val)
			{
			    *_iter  = _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator +=(const T_& val)
			{
			    *_iter += _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator -=(const T_& val)
			{
			    *_iter -= _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator *=(const T_& val)
			{
			    *_iter *= _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator /=(const T_& val)
			{
			    *_iter /= _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator &=(const T_& val)
			{
			    *_iter &= _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator |=(const T_& val)
			{
			    *_iter |= _func(val);
			    return *this;
			}
      template <class T_>
      assignment_proxy&	operator ^=(const T_& val)
			{
			    *_iter ^= _func(val);
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
template <class FUNC, class ITER, class VAL=boost::use_default>
class assignment_iterator
    : public boost::iterator_adaptor<assignment_iterator<FUNC, ITER, VAL>,
				     ITER,
				     VAL,
				     boost::use_default,
				     detail::assignment_proxy<FUNC, ITER> >
{
  private:
    using super	= boost::iterator_adaptor<
		      assignment_iterator,
		      ITER,
		      VAL,
		      boost::use_default,
		      detail::assignment_proxy<FUNC, ITER> >;

  public:
    using		typename super::reference;
    
    friend class	boost::iterator_core_access;

  public:
    assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)			{}

    const FUNC&	functor()			const	{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return {super::base(), _func};
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class VAL=boost::use_default, class FUNC, class ITER>
inline assignment_iterator<FUNC, ITER, VAL>
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
using vertical_iterator = boost::transform_iterator<
			      row2col<ROW>,
			      ROW,
			      boost::use_default,
			      value_t<iterator_value<ROW> > >;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(ROW row, size_t col)
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
    using super		= boost::iterator_adaptor<ring_iterator, ITER>;

    friend class	boost::iterator_core_access;
    
  public:
    using		typename super::difference_type;
    
  public:
    ring_iterator()
	:super(), _begin(super::base()), _end(super::base()), _d(0)	{}
    
    ring_iterator(ITER begin, ITER end)
	:super(begin),
	 _begin(begin), _end(end), _d(std::distance(_begin, _end))	{}

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
    
  private:
    ITER		_begin;	// 代入を可能にするためconstは付けない
    ITER		_end;	// 同上
    difference_type	_d;	// 同上
};

template <class ITER> ring_iterator<ITER>
make_ring_iterator(ITER begin, ITER end)	{ return {begin, end}; }

}	// namespace TU
#endif	// !__TU_ITERATOR_H
