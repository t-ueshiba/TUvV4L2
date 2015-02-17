/*
 *  平成14-24年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2012.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: iterator.h 1785 2015-02-14 05:43:15Z ueshiba $
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TU_ITERATOR_H
#define __TU_ITERATOR_H

#include <boost/iterator/transform_iterator.hpp>
#include "TU/tuple.h"

namespace TU
{
//! T型のメンバ変数を持つオブジェクトへの反復子からそのメンバに直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
  \param mbr	iterが指すオブジェクトのメンバへのポインタ
*/
template <class ITER, class T> inline auto
make_mbr_iterator(const ITER& iter,
		  T std::iterator_traits<ITER>::value_type::* mbr)
    -> decltype(boost::make_transform_iterator(
		    iter,
		    std::function<
			typename std::conditional<
			    std::is_same<
				typename std::iterator_traits<ITER>::pointer,
				typename std::iterator_traits<ITER>::value_type*>
				::value,
			    T&, const T&>::type
		        (typename std::iterator_traits<ITER>::reference)>(
			    std::mem_fn(mbr))))
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
    -> decltype(make_mbr_iterator(
		    iter, &std::iterator_traits<ITER>::value_type::first))
{
    return make_mbr_iterator(
		iter, &std::iterator_traits<ITER>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセスする反復子を作る．
/*!
  \param iter	ベースとなる反復子
*/
template <class ITER> inline auto
make_second_iterator(const ITER& iter)
    -> decltype(make_mbr_iterator(
		    iter, &std::iterator_traits<ITER>::value_type::second))
{
    return make_mbr_iterator(
		iter, &std::iterator_traits<ITER>::value_type::second);
}
    
/************************************************************************
*  class fast_zip_iterator<ITER_TUPLE>					*
************************************************************************/
namespace detail
{
  struct generic_dereference
  {
      template <class ITER> 
      typename std::enable_if<
	  !std::is_reference<
	      typename std::iterator_traits<ITER>::reference>::value,
	  typename std::iterator_traits<ITER>::reference>::type
      operator ()(const ITER& iter) const
      {
	  return *iter;
      }
      template <class ITER> auto
      operator ()(const ITER& iter) const
	  -> typename std::enable_if<
	      std::is_reference<
		  typename std::iterator_traits<ITER>::reference>::value,
	      decltype(std::ref(*iter))>::type
      {
	  return std::ref(*iter);
      }
  };
}
    
template <class ITER_TUPLE>
class fast_zip_iterator
    : public boost::iterator_facade<
	  fast_zip_iterator<ITER_TUPLE>,
	  decltype(std::detail::transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference())),
	  typename std::iterator_traits<
	      typename std::tuple_element<0, ITER_TUPLE>::type>
	      ::iterator_category,
	  decltype(std::detail::transform(std::declval<ITER_TUPLE>(),
					  detail::generic_dereference()))>
{
  private:
    typedef boost::iterator_facade<
	fast_zip_iterator<ITER_TUPLE>,
        decltype(
	    std::detail::transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference())),
        typename std::iterator_traits<
	    typename std::tuple_element<0, ITER_TUPLE>::type>
	    ::iterator_category,
        decltype(
	    std::detail::transform(std::declval<ITER_TUPLE>(),
				   detail::generic_dereference()))>	super;
    
  public:
    typedef typename super::reference			reference;
    typedef typename super::difference_type		difference_type;
    
    friend class	boost::iterator_core_access;

  private:
    struct Increment
    {
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ ++iter; }
    };

    struct Decrement
    {
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ --iter; }
    };

    struct Advance
    {
	Advance(difference_type n)	:_n(n)	{}
	
	template <class ITER_>
	void	operator ()(ITER_& iter) const	{ std::advance(iter, _n); }

      private:
	const difference_type	_n;
    };
    
  public:
    fast_zip_iterator(const ITER_TUPLE& iter_tuple)
	:_iter_tuple(iter_tuple)		{}

    const ITER_TUPLE&
		get_iterator_tuple()	const	{ return _iter_tuple; }
    
  private:
    reference	dereference() const
		{
		    return std::detail::transform(_iter_tuple,
						  detail::generic_dereference());
		}
    bool	equal(const fast_zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			== std::get<0>(_iter_tuple);
		}
    void	increment()
		{
		    std::detail::for_each(_iter_tuple, Increment());
		}
    void	decrement()
		{
		    std::detail::for_each(_iter_tuple, Decrement());
		}
    void	advance(difference_type n)
		{
		    std::detail::for_each(_iter_tuple, Advance(n));
		}
    difference_type
		distance_to(const fast_zip_iterator& iter) const
		{
		    return std::get<0>(iter.get_iterator_tuple())
			 - std::get<0>(_iter_tuple);
		}

  private:
    ITER_TUPLE	_iter_tuple;
};

template <class ITER_TUPLE> inline fast_zip_iterator<ITER_TUPLE>
make_fast_zip_iterator(const ITER_TUPLE& iter_tuple)
{
    return fast_zip_iterator<ITER_TUPLE>(iter_tuple);
}

}

namespace std
{
/************************************************************************
*  std::[begin|end](std::tuple<T...>)					*
************************************************************************/
namespace detail
{
  struct generic_begin
  {
      template <class T> auto
      operator ()(const T& x) const -> decltype(std::begin(x))
      {
	  return std::begin(x);
      }
      template <class T> auto
      operator ()(T& x) const -> decltype(std::begin(x))
      {
	  return std::begin(x);
      }
  };
    
  struct generic_end
  {
      template <class T> auto
      operator ()(const T& x) const -> decltype(std::end(x))
      {
	  return std::end(x);
      }
      template <class T> auto
      operator ()(T& x) const -> decltype(std::end(x))
      {
	  return std::end(x);
      }
  };
}

template <class ...T> inline auto
begin(const tuple<T...>& x)
    -> decltype(TU::make_fast_zip_iterator(detail::transform(
					       x, detail::generic_begin())))
{
    return TU::make_fast_zip_iterator(detail::transform(
					  x, detail::generic_begin()));
}
    
template <class ...T> inline auto
end(const tuple<T...>& x)
    -> decltype(TU::make_fast_zip_iterator(detail::transform(
					       x, detail::generic_end())))
{
    return TU::make_fast_zip_iterator(detail::transform(
					  x, detail::generic_end()));
}
    
}

namespace TU
{
/************************************************************************
*  size(T)								*
************************************************************************/
template <class T> inline size_t
size(const T& x)
{
    return x.size();
}
template <class ...T> inline size_t
size(const std::tuple<T...>& x)
{
    return size(std::get<0>(x));
}
      
/************************************************************************
*  ncol(T)								*
************************************************************************/
template <class T> inline size_t
ncol(const T& x)
{
    return (x.begin() != x.end() ? size(*x.begin()) : 0);
}
      
/************************************************************************
*  struct iterator_value<ITER>						*
************************************************************************/
namespace detail
{
  struct generic_value
  {
      template <class ITER> typename std::iterator_traits<ITER>::value_type
      operator ()(const ITER& iter) const
      {
	  return *iter;
      }
  };

  template <class ITER>
  struct iterator_value
  {
      typedef typename std::iterator_traits<ITER>::value_type	type;
  };
  template <class ITER_TUPLE>
  struct iterator_value<fast_zip_iterator<ITER_TUPLE> >
  {
      typedef decltype(
	  std::detail::transform(std::declval<ITER_TUPLE>(),
				 generic_value()))		type;
  };
}
    
//! 与えられた反復子が指す値の型を返す．
/*!
  fast_zip_iteratorのvalue_typeは参照のtupleとして定義されているが，
  本メタ関数は値のtupleを返す．
  \param ITER	反復子の型
*/
template <class ITER>
using iterator_value = typename detail::iterator_value<ITER>::type;
    
/************************************************************************
*  class assignment_iterator<FUNC, ITER>				*
************************************************************************/
namespace detail
{
    template <class FUNC, class ITER>
    class assignment_proxy
    {
      public:
	typedef assignment_proxy	self;

      public:
	assignment_proxy(const ITER& iter, const FUNC& func)
	    :_iter(iter), _func(func)					{}

	template <class T>
	self&	operator =(const T& val)
		{
		    *_iter  = _func(val);
		    return *this;
		}
	template <class T>
	self&	operator +=(const T& val)
		{
		    *_iter += _func(val);
		    return *this;
		}
	template <class T>
	self&	operator -=(const T& val)
		{
		    *_iter -= _func(val);
		    return *this;
		}
	template <class T>
	self&	operator *=(const T& val)
		{
		    *_iter *= _func(val);
		    return *this;
		}
	template <class T>
	self&	operator /=(const T& val)
		{
		    *_iter /= _func(val);
		    return *this;
		}
	template <class T>
	self&	operator &=(const T& val)
		{
		    *_iter &= _func(val);
		    return *this;
		}
	template <class T>
	self&	operator |=(const T& val)
		{
		    *_iter |= _func(val);
		    return *this;
		}
	template <class T>
	self&	operator ^=(const T& val)
		{
		    *_iter ^= _func(val);
		    return *this;
		}

      private:
	const ITER&	_iter;
	const FUNC&	_func;
    };
}

//! operator *()を左辺値として使うときに，右辺値を指定された関数によって変換してから代入を行うための反復子
/*!
  \param FUNC	変換を行う関数オブジェクトの型
  \param ITER	変換結果の代入先を指す反復子
*/
template <class FUNC, class ITER>
class assignment_iterator
    : public boost::iterator_adaptor<assignment_iterator<FUNC, ITER>,
				     ITER,
				     typename FUNC::argument_type,
				     boost::use_default,
				     detail::assignment_proxy<FUNC, ITER> >
{
  private:
    typedef boost::iterator_adaptor<
		assignment_iterator,
		ITER,
		typename FUNC::argument_type,
		boost::use_default,
		detail::assignment_proxy<FUNC, ITER> >	super;

  public:
    typedef typename super::reference	reference;
    friend class			boost::iterator_core_access;

  public:
    assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)			{}

    const FUNC&	functor()	const			{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(const ITER& iter, const FUNC& func=FUNC())
{
    return assignment_iterator<FUNC, ITER>(iter, func);
}

/************************************************************************
*  class assignment2_iterator<FUNC, ITER>				*
************************************************************************/
namespace detail
{
    template <class FUNC, class ITER>
    class assignment2_proxy
    {
      public:
	typedef assignment2_proxy	self;
	
      public:
	assignment2_proxy(ITER const& iter, FUNC const& func)
	    :_iter(iter), _func(func)					{}

	template <class T>
	self&	operator =(const T& val)
		{
		    _func(val, *_iter);
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
class assignment2_iterator
    : public boost::iterator_adaptor<assignment2_iterator<FUNC, ITER>,
				     ITER,
				     typename FUNC::first_argument_type,
				     boost::use_default,
				     detail::assignment2_proxy<FUNC, ITER> >
{
  private:
    typedef boost::iterator_adaptor<assignment2_iterator,
				    ITER,
				    typename FUNC::first_argument_type,
				    boost::use_default,
				    detail::assignment2_proxy<FUNC, ITER> >
						super;

  public:
    typedef typename super::reference	reference;
    
    friend class	boost::iterator_core_access;

  public:
    assignment2_iterator(const ITER& iter, const FUNC& func=FUNC())
	:super(iter), _func(func)			{}

    const FUNC&	functor()			const	{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment2_iterator<FUNC, ITER>
make_assignment2_iterator(const ITER& iter, const FUNC& func=FUNC())
{
    return assignment2_iterator<FUNC, ITER>(iter, func);
}

/************************************************************************
*  alias subiterator<ITER>						*
************************************************************************/
template <class ITER>
using subiterator = decltype(std::begin(*std::declval<ITER>()));

/************************************************************************
*  class range<ITER>							*
************************************************************************/
template <class ITER>
class range
{
  public:
    typedef ITER					iterator;
    typedef iterator					const_iterator;
    typedef std::reverse_iterator<iterator>		reverse_iterator;
    typedef std::reverse_iterator<const_iterator>	const_reverse_iterator;
    typedef typename std::iterator_traits<iterator>::value_type
							value_type;
    typedef typename std::iterator_traits<iterator>::reference
							reference;
      
  public:
    range(const iterator& begin, const iterator& end)
	:_begin(begin), _end(end)			{}

    size_t		size()	 const	{ return std::distance(_begin, _end); }
    iterator		begin()	 const	{ return _begin; }
    iterator		end()	 const	{ return _end; }
    reverse_iterator	rbegin() const	{ return reverse_iterator(_end); }
    reverse_iterator	rend()	 const	{ return reverse_iterator(_begin); }
    reference		operator [](size_t i) const
			{
			    return *(_begin + i);
			}
      
  private:
    const iterator	_begin;
    const iterator	_end;
};
    
/************************************************************************
*  class row_iterator<COL, ROW, ARG...>					*
************************************************************************/
//! コンテナを指す反復子に対して，取り出した値を変換したり値を変換してから格納する作業をサポートする反復子
/*!
  \param OUT	コンテナ中の個々の値に対して変換を行う反復子の型
  \param ROW	begin(), end()をサポートするコンテナを指す反復子の型
  \param ARG	OUTを生成するための引数の型
*/ 
template <class OUT, class ROW, class ...ARG>
class row_iterator
    : public boost::iterator_adaptor<row_iterator<OUT, ROW, ARG...>,
				     ROW,
				     range<typename std::conditional<
					       std::is_void<OUT>::value,
					       subiterator<ROW>, OUT>::type>,
				     boost::use_default,
				     range<typename std::conditional<
					       std::is_void<OUT>::value,
					       subiterator<ROW>, OUT>::type> >
{
  private:
    typedef typename std::conditional<std::is_void<OUT>::value,
				      subiterator<ROW>, OUT>::type
								out_iterator;
    typedef boost::iterator_adaptor<row_iterator,
				    ROW,
				    range<out_iterator>,
				    boost::use_default,
				    range<out_iterator> >
								super;

  public:
    typedef typename super::reference				reference;

    friend class	boost::iterator_core_access;

  private:
    reference	dereference() const
		{
		    return dereference(super::base());
		}
    reference	dereference(const ROW& row) const
		{
		    auto	col_begin = std::begin(*row);
		    std::advance(col_begin, _jb);

		    if (_je != 0)
		    {
			auto	col_end = std::begin(*row);
			std::advance(col_end, _je);
			return reference(_make_out_iterator(col_begin),
					 _make_out_iterator(col_end));
		    }
		    else
			return reference(_make_out_iterator(col_begin),
					 _make_out_iterator(std::end(*row)));
		}
    static out_iterator
		make_out_iterator(const subiterator<ROW>& col,
				  const ARG& ...arg)
		{
		    return make_out_iterator_impl(
			       typename std::is_void<OUT>::type(), col, arg...);
		}
    static out_iterator
		make_out_iterator_impl(std::true_type,
				       const subiterator<ROW>& col,
				       const ARG& ...arg)
		{
		    return col;
		}
    static out_iterator
		make_out_iterator_impl(std::false_type,
				       const subiterator<ROW>& col,
				       const ARG& ...arg)
		{
		    return out_iterator(col, arg...);
		}
    
  public:
    row_iterator(const ROW& row, size_t jb, size_t je, const ARG& ...arg)
	:super(row),
	 _make_out_iterator(std::bind(&row_iterator::make_out_iterator,
				      std::placeholders::_1, arg...)),
	 _jb(jb), _je(je)						{}
    row_iterator(const ROW& row, const ARG& ...arg)
	:row_iterator(row, 0, 0, arg...)				{}

    reference	operator [](size_t i) const
		{
		    return dereference(super::base() + i);
		}

  private:
    decltype(std::bind(&row_iterator::make_out_iterator,
		       std::placeholders::_1,
		       std::declval<const ARG&>()...))	_make_out_iterator;
    size_t						_jb;
    size_t						_je;
};

template <class OUT=void, class ROW, class ...ARG>
inline row_iterator<OUT, ROW, ARG...>
make_row_iterator(const ROW& row, const ARG& ...arg)
{
    return row_iterator<OUT, ROW, ARG...>(row, arg...);
}

template <class OUT=void, class ROW, class ...ARG>
inline row_iterator<OUT, ROW, ARG...>
make_row_iterator(size_t jb, size_t je, const ROW& row, const ARG& ...arg)
{
    return row_iterator<OUT, ROW, ARG...>(row, jb, je, arg...);
}

template <template <class, class> class OUT, class ROW, class ARG>
inline row_iterator<OUT<ARG, subiterator<ROW> >, ROW, ARG>
make_row_uniarg_iterator(const ROW& row, const ARG& arg)
{
    return row_iterator<OUT<ARG, subiterator<ROW> >, ROW, ARG>(row, arg);
}

template <template <class, class> class OUT, class ROW, class ARG>
inline row_iterator<OUT<ARG, subiterator<ROW> >, ROW, ARG>
make_row_uniarg_iterator(size_t jb, size_t je, const ROW& row, const ARG& arg)
{
    return row_iterator<OUT<ARG, subiterator<ROW> >, ROW, ARG>(row, jb, je, arg);
}

template <class FUNC, class ROW>
inline row_iterator<boost::transform_iterator<FUNC, subiterator<ROW> >,
		    ROW, FUNC>
make_row_transform_iterator(const ROW& row, const FUNC& func)
{
    return row_iterator<boost::transform_iterator<FUNC, subiterator<ROW> >,
			ROW, FUNC>(row, func);
}

template <class FUNC, class ROW>
inline row_iterator<boost::transform_iterator<FUNC, subiterator<ROW> >,
		    ROW, FUNC>
make_row_transform_iterator(size_t jb, size_t je,
			    const ROW& row, const FUNC& func)
{
    return row_iterator<boost::transform_iterator<FUNC, subiterator<ROW> >,
			ROW, FUNC>(row, jb, je, func);
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
    typedef typename std::iterator_traits<ROW>::reference	argument_type;
    typedef typename std::iterator_traits<subiterator<ROW> >::reference
								result_type;
    
  public:
    row2col(size_t col)	:_col(col)				{}
    
    result_type	operator ()(argument_type row) const
		{
		    return *(std::begin(row) + _col);
		}
    
  private:
    const size_t	_col;	//!< 列を指定するindex
};

template <class ROW>
using vertical_iterator = boost::transform_iterator<
			      row2col<ROW>, ROW,
			      boost::use_default,
			      typename std::iterator_traits<subiterator<ROW> >
					  ::value_type>;

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(const ROW& row, size_t idx)
{
    return vertical_iterator<ROW>(row, row2col<ROW>(idx));
}

/************************************************************************
*  class column_iterator<OUT, ROW, ARG...>				*
************************************************************************/
//! 2次元配列の列を指す反復子
/*!
  \param OUT	コンテナ中の個々の値に対して変換を行う反復子の型
  \param ROW	begin(), end()をサポートするコンテナを指す反復子の型
  \param ARG	OUTを生成するための引数の型
*/ 
template <class OUT, class ROW, class ...ARG>
class column_iterator
    : public boost::iterator_facade<column_iterator<OUT, ROW, ARG...>,
				    range<typename std::conditional<
					      std::is_void<OUT>::value,
					      vertical_iterator<ROW>,
					      OUT>::type>,
				    std::random_access_iterator_tag,
				    range<typename std::conditional<
					      std::is_void<OUT>::value,
					      vertical_iterator<ROW>,
					      OUT>::type> >
{
  private:
    typedef typename std::conditional<std::is_void<OUT>::value,
				      vertical_iterator<ROW>, OUT>::type
								out_iterator;
    typedef boost::iterator_facade<column_iterator,
				   range<out_iterator>,
				   std::random_access_iterator_tag,
				   range<out_iterator> >	super;

  public:
    typedef typename super::reference		reference;
    typedef typename super::difference_type	difference_type;
    
    friend class	boost::iterator_core_access;

  private:
    reference	dereference() const
		{
		    return (*this)[_col];
		}
    bool	equal(const column_iterator& iter) const
		{
		    return _col == iter._col;
		}
    void	increment()
		{
		    ++_col;
		}
    void	decrement()
		{
		    --_col;
		}
    void	advance(difference_type n)
		{
		    _col += n;
		}
    difference_type
		distance_to(const column_iterator& iter) const
		{
		    return iter._col - _col;
		}
    static out_iterator
		make_out_iterator(const vertical_iterator<ROW>& row,
				  const ARG& ...arg)
		{
		    return make_out_iterator_impl(
			       typename std::is_void<OUT>::type(), row, arg...);
		}
    static out_iterator
		make_out_iterator_impl(std::true_type,
				       const vertical_iterator<ROW>& row,
				       const ARG& ...arg)
		{
		    return row;
		}
    static out_iterator
		make_out_iterator_impl(std::false_type,
				       const vertical_iterator<ROW>& row,
				       const ARG& ...arg)
		{
		    return out_iterator(row, arg...);
		}
    
  public:
    column_iterator(const ROW& begin,
		    const ROW& end, size_t col, const ARG& ...arg)
	:_begin(begin), _end(end), _col(col),
	 _make_out_iterator(std::bind(&column_iterator::make_out_iterator,
				      std::placeholders::_1, arg...))	{}

    reference	operator [](size_t j) const
		{
		    return reference(_make_out_iterator(
					 make_vertical_iterator(_begin, j)),
				     _make_out_iterator(
					 make_vertical_iterator(_end, j)));
		}

  private:
    ROW							_begin;
    ROW							_end;
    difference_type					_col;
    decltype(std::bind(&column_iterator::make_out_iterator,
		       std::placeholders::_1,
		       std::declval<const ARG&>()...))	_make_out_iterator;
};

template <class OUT=void, class ROW, class ...ARG>
inline column_iterator<OUT, ROW, ARG...>
make_column_iterator(const ROW& begin,
		     const ROW& end, size_t col, const ARG& ...arg)
{
    return column_iterator<OUT, ROW, ARG...>(begin, end, col, arg...);
}

template <template <class, class> class OUT, class ROW, class ARG>
inline row_iterator<OUT<ARG, vertical_iterator<ROW> >, ROW, ARG>
make_column_uniarg_iterator(const ROW& begin,
			    const ROW& end, size_t col, const ARG& arg)
{
    return column_iterator<OUT<ARG, vertical_iterator<ROW> >, ROW, ARG>(
	       begin, end, col, arg);
}

template <class FUNC, class ROW>
inline column_iterator<boost::transform_iterator<FUNC, vertical_iterator<ROW> >,
		       ROW, FUNC>
make_column_transform_iterator(const ROW& begin,
			       const ROW& end, size_t col, const FUNC& func)
{
    return column_iterator<
	       boost::transform_iterator<FUNC, vertical_iterator<ROW> >,
	       ROW, FUNC>(begin, end, col, func);
}

template <class ROW> inline auto
column_begin(const ROW& begin, const ROW& end)
    -> decltype(make_column_iterator(begin, end, 0))
{
    return make_column_iterator(begin, end, 0);
}
    
template <class ROW> inline auto
column_end(const ROW& begin, const ROW& end)
    -> decltype(make_column_iterator(begin, end, 0))
{
    return make_column_iterator(begin, end, (begin != end ? size(*begin) : 0));
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
    typedef boost::iterator_adaptor<ring_iterator, ITER>	super;

  public:
    typedef typename super::difference_type			difference_type;
    
    friend class	boost::iterator_core_access;

  public:
    ring_iterator()
	:super(), _begin(super::base()), _end(super::base())	{}
    
    ring_iterator(const ITER& begin, const ITER& end)
	:super(begin), _begin(begin), _end(end)			{}

  private:
    void	advance(difference_type n)
		{
		    difference_type	d = std::distance(_begin, _end);
		    n %= d;
		    difference_type	i = std::distance(_begin,
							  super::base()) + n;
		    if (i >= d)
			std::advance(super::base_reference(), n - d);
		    else if (i < 0)
			std::advance(super::base_reference(), n + d);
		    else
			std::advance(super::base_reference(), n);
		}
    
    void	increment()
		{
		    if (++super::base_reference() == _end)
			super::base_reference() = _begin;
		}

    void	decrement()
		{
		    if (super::base() == _begin)
			super::base_reference() = _end;
		    --super::base_reference();
		}
    
  private:
    ITER	_begin;	// 代入を可能にするためconstは付けない
    ITER	_end;	// 同上
};

template <class ITER> ring_iterator<ITER>
make_ring_iterator(const ITER& begin, const ITER& end)
{
    return ring_iterator<ITER>(begin, end);
}

}	// namespace TU
#endif	// !__TU_ITERATOR_H
