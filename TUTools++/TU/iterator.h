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
 *  $Id$
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TUiterator_h
#define __TUiterator_h

#include <cstddef>				// for including size_t
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/type_traits.hpp>
#include <boost/mpl/if.hpp>
#include "TU/functional.h"

namespace TU
{
//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R/W)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<mem_var_ref_t<S, T>, Iterator>
make_mbr_iterator(Iterator i, S T::* m)
{
    return boost::make_transform_iterator(i, mem_var_ref(m));
}
    
//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<const_mem_var_ref_t<S, T>, Iterator>
make_const_mbr_iterator(Iterator i, S const T::* m)
{
    return boost::make_transform_iterator(i, const_mem_var_ref(m));
}

//! std::pairへの反復子からその第1要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_first_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第1要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_first_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::second_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_second_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::second_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_second_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
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
	assignment_proxy(ITER const& iter, FUNC const& func)
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
	ITER const&	_iter;
	FUNC const&	_func;
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
    typedef boost::iterator_adaptor<assignment_iterator,
				    ITER,
				    typename FUNC::argument_type,
				    boost::use_default,
				    detail::assignment_proxy<FUNC, ITER> >
						super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    assignment_iterator(ITER const& iter, FUNC const& func=FUNC())
	:super(iter), _func(func)			{}

    FUNC const&	functor()	const			{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(ITER iter, FUNC func)
{
    return assignment_iterator<FUNC, ITER>(iter, func);
}

template <class FUNC, class ITER> inline assignment_iterator<FUNC, ITER>
make_assignment_iterator(ITER iter)
{
    return assignment_iterator<FUNC, ITER>(iter);
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
	ITER const&	_iter;
	FUNC const&	_func;
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
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    assignment2_iterator(ITER const& iter, FUNC const& func=FUNC())
	:super(iter), _func(func)			{}

    FUNC const&	functor()			const	{ return _func; }
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    FUNC 	_func;	// 代入を可能にするためconstは付けない
};
    
template <class FUNC, class ITER> inline assignment2_iterator<FUNC, ITER>
make_assignment2_iterator(ITER iter, FUNC func)
{
    return assignment2_iterator<FUNC, ITER>(iter, func);
}

template <class FUNC, class ITER> inline assignment2_iterator<FUNC, ITER>
make_assignment2_iterator(ITER iter)
{
    return assignment2_iterator<FUNC, ITER>(iter);
}

/************************************************************************
*  class fast_zip_iterator<TUPLE>					*
************************************************************************/
//! iterator tupleの最初の成分のみを同一性判定に使うことにより boost::zip_iterator<TUPLE> を高速化した反復子
template <class TUPLE>
class fast_zip_iterator
    : public boost::iterator_adaptor<fast_zip_iterator<TUPLE>,
				     boost::zip_iterator<TUPLE> >
{
  private:
    typedef boost::iterator_adaptor<fast_zip_iterator<TUPLE>,
				    boost::zip_iterator<TUPLE> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    fast_zip_iterator(TUPLE const& t)	:super(t)	{}

    TUPLE const&	get_iterator_tuple() const
			{
			    return super::base().get_iterator_tuple();
			}
    
  private:
    bool		equal(fast_zip_iterator const& iter) const
			{
			    return boost::get<0>(get_iterator_tuple()) ==
				   boost::get<0>(iter.get_iterator_tuple());
			}
};

template <class TUPLE> fast_zip_iterator<TUPLE>
make_fast_zip_iterator(TUPLE t)
{
    return fast_zip_iterator<TUPLE>(t);
}

/************************************************************************
*  struct subiterator<ITER>						*
************************************************************************/
//! 反復子 ITER が指す値が持つ iterator もしくは const_iterator
template <class ITER>
struct subiterator
{
    typedef typename boost::mpl::if_<
	boost::is_same<typename std::iterator_traits<ITER>::pointer,
		       typename std::iterator_traits<ITER>::value_type*>,
	typename std::iterator_traits<ITER>::value_type::iterator,
	typename std::iterator_traits<ITER>::value_type::const_iterator>
	::type						type;
    
    typedef typename std::iterator_traits<type>::difference_type
							difference_type;
    typedef typename std::iterator_traits<type>::value_type
							value_type;
    typedef typename std::iterator_traits<type>::pointer
							pointer;
    typedef typename std::iterator_traits<type>::reference
							reference;
    typedef typename std::iterator_traits<type>::iterator_category
							iterator_category;
};

template <class TUPLE>
struct subiterator<boost::zip_iterator<TUPLE> >
{
    typedef boost::zip_iterator<
		typename boost::detail::tuple_impl_specific
			      ::tuple_meta_transform<
		    TUPLE, subiterator<boost::mpl::_1> >::type>	type;

    typedef typename std::iterator_traits<type>::difference_type
							difference_type;
    typedef typename std::iterator_traits<type>::value_type
							value_type;
    typedef typename std::iterator_traits<type>::pointer
							pointer;
    typedef typename std::iterator_traits<type>::reference
							reference;
    typedef typename std::iterator_traits<type>::iterator_category
							iterator_category;
};

template <class TUPLE>
struct subiterator<fast_zip_iterator<TUPLE> >
{
    typedef fast_zip_iterator<
		typename boost::detail::tuple_impl_specific
			      ::tuple_meta_transform<
		    TUPLE, subiterator<boost::mpl::_1> >::type>	type;

    typedef typename std::iterator_traits<type>::difference_type
							difference_type;
    typedef typename std::iterator_traits<type>::value_type
							value_type;
    typedef typename std::iterator_traits<type>::pointer
							pointer;
    typedef typename std::iterator_traits<type>::reference
							reference;
    typedef typename std::iterator_traits<type>::iterator_category
							iterator_category;
};

/************************************************************************
*  class row_iterator<ROW, COL, ARG>					*
************************************************************************/
namespace detail
{
    template <class ROW, class COL, class ARG>
    class row_proxy
    {
      private:
	typedef typename subiterator<ROW>::type		subiter_t;
	
	struct invoke_begin
	{
	    template <class _ROW>
	    struct apply { typedef typename subiterator<_ROW>::type type; };

	    invoke_begin(size_t j)	:_j(j)		{}
	    template <class _ROW> typename apply<_ROW>::type
	    operator ()(_ROW const& row) const
	    {
		typename apply<_ROW>::type	col = row->begin();
		std::advance(col, _j);
		return col;
	    }

	    size_t const	_j;
	};
	
	struct invoke_end
	{
	    template <class _ROW>
	    struct apply { typedef typename subiterator<_ROW>::type type; };

	    invoke_end(size_t)				{}
	    template <class _ROW> typename apply<_ROW>::type
	    operator ()(_ROW const& row)	const	{return row->end();}
	};

	template <class _COL, class _ARG, class=void>
	struct col_iterator
	{
	    static _COL
	    make(subiter_t col, _ARG const& arg)
	    {
		return _COL(col, arg);
	    }
	};
	template <class _COL, class DUMMY>
	struct col_iterator<_COL, boost::tuples::null_type, DUMMY>
	{
	    static _COL
	    make(subiter_t col, boost::tuples::null_type)
	    {
		return _COL(col);
	    }
	};
	template <class DUMMY>
	struct col_iterator<boost::use_default,
			    boost::tuples::null_type, DUMMY>
	{
	    static subiter_t
	    make(subiter_t col, boost::tuples::null_type)
	    {
		return col;
	    }
	};

	template <class INVOKE, class _ROW>
	static typename subiterator<_ROW>::type
	make_subiterator(_ROW const& row, size_t j)
	{
	    return INVOKE(j)(row);
	}
	template <class INVOKE, class TUPLE>
	static typename subiterator<fast_zip_iterator<TUPLE> >::type
	make_subiterator(fast_zip_iterator<TUPLE> const& row, size_t j)
	{
	    return make_fast_zip_iterator(
		boost::detail::tuple_impl_specific::
		tuple_transform(row.get_iterator_tuple(), INVOKE(j)));
	}

      public:
	typedef typename boost::mpl::if_<
	    boost::is_same<COL, boost::use_default>,
	    subiter_t, COL>::type			iterator;
	typedef iterator				const_iterator;

      public:
	row_proxy(ROW const& row, ARG const& arg, size_t jb, size_t je)
	    :_row(row), _arg(arg), _jb(jb), _je(je)			{}

	iterator
	begin() const
	{
	    return col_iterator<COL, ARG>::make(
		make_subiterator<invoke_begin>(_row, _jb), _arg);
	}

	iterator
	end() const
	{
	    if (_je != 0)
		return col_iterator<COL, ARG>::make(
		    make_subiterator<invoke_begin>(_row, _je), _arg);
	    else
		return col_iterator<COL, ARG>::make(
		    make_subiterator<invoke_end>(_row, 0), _arg);
	}

      private:
	ROW const&	_row;
	ARG const&	_arg;
	size_t const	_jb;
	size_t const	_je;
    };
}
    
//! コンテナを指す反復子に対して，取り出した値を変換したり値を変換してから格納する作業をサポートする反復子
/*!
  \param ROW	begin(), end()をサポートするコンテナを指す反復子
  \param COL	コンテナ中の個々の値に対して変換を行う反復子
  \param ARG	変換関数
*/ 
template <class ROW,
	  class COL=boost::use_default, class ARG=boost::tuples::null_type>
class row_iterator
    : public boost::iterator_adaptor<row_iterator<ROW, COL, ARG>,
				     ROW,
				     detail::row_proxy<ROW, COL, ARG>,
				     boost::use_default,
				     detail::row_proxy<ROW, COL, ARG> >
{
  private:
    typedef boost::iterator_adaptor<row_iterator,
				    ROW,
				    detail::row_proxy<ROW, COL, ARG>,
				    boost::use_default,
				    detail::row_proxy<ROW, COL, ARG> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    row_iterator(ROW const& row,
		 ARG const& arg=ARG(), size_t jb=0, size_t je=0)
	:super(row), _arg(arg), _jb(jb), _je(je)		{}

    reference	operator [](size_t i) const
		{
		    return reference(super::base() + i, _arg, _jb, _je);
		}
    
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _arg, _jb, _je);
		}

  private:
    ARG 	_arg;	// 代入を可能にするためconstは付けない
    size_t	_jb;	// 同上
    size_t	_je;	// 同上
};

template <template <class, class> class COL, class ARG, class ROW>
inline row_iterator<ROW, COL<ARG, typename subiterator<ROW>::type>, ARG>
make_row_iterator(ROW row, ARG arg=ARG(), size_t jb=0, size_t je=0)
{
    typedef typename subiterator<ROW>::type	col_iterator;

    return row_iterator<ROW, COL<ARG, col_iterator>, ARG>(row, arg, jb, je);
}

template <class COL, class ROW>
inline row_iterator<ROW, COL>
make_row_iterator(ROW row, size_t jb=0, size_t je=0)
{
    return row_iterator<ROW, COL>(row, boost::tuples::null_type(), jb, je);
}

template <class FUNC, class ROW>
inline row_iterator<ROW, boost::transform_iterator<
			     FUNC, typename subiterator<ROW>::type>, FUNC>
make_row_transform_iterator(ROW row, FUNC func=FUNC(),
			    size_t jb=0, size_t je=0)
{
    typedef typename subiterator<ROW>::type	subiter_t;

    return row_iterator<ROW, boost::transform_iterator<FUNC, subiter_t>,
			FUNC>(row, func, jb, je);
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
  private:
    typedef typename subiterator<ROW>::type			COL;

  public:
    typedef typename std::iterator_traits<ROW>::reference	argument_type;
    typedef typename std::iterator_traits<COL>::reference	result_type;
    
  public:
    row2col(size_t idx)	:_idx(idx)				{}
    
    result_type	operator ()(argument_type row) const
		{
		    return *(row.begin() + _idx);
		}
    
  private:
    size_t const	_idx;	//!< 列を指定するindex
};

template <class ROW>
struct vertical_iterator
{
    typedef boost::transform_iterator<
		row2col<ROW>, ROW, boost::use_default,
		typename subiterator<ROW>::value_type>	type;

    typedef typename type::difference_type		difference_type;
    typedef typename type::value_type			value_type;
    typedef typename type::pointer			pointer;
    typedef typename type::reference			reference;
    typedef typename type::iterator_category		iterator_category;
};

template <class ROW>
inline typename vertical_iterator<ROW>::type
make_vertical_iterator(ROW row, size_t idx)
{
    return typename vertical_iterator<ROW>::type(row, row2col<ROW>(idx));
}
    
/************************************************************************
*  class ring_iterator<ITER>						*
************************************************************************/
//! 2つの反復子によって指定された範囲を循環バッファとしてアクセスする反復子
/*!
  \param ITER	データ列中の要素を指す反復子の型
*/
template <class ITER>
class ring_iterator
    : public boost::iterator_adaptor<ring_iterator<ITER>,	// self
				     ITER,			// base
				     boost::use_default,	// value_type
				     boost::single_pass_traversal_tag>
{
  private:
    typedef boost::iterator_adaptor<ring_iterator,
				    ITER,
				    boost::use_default,
				    boost::single_pass_traversal_tag>	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    ring_iterator()
	:super(), _begin(super::base()), _end(super::base())	{}
    
    ring_iterator(ITER const& begin, ITER const& end)
	:super(begin), _begin(begin), _end(end)			{}

  private:
    void	advance(difference_type n)
		{
		    difference_type	d = std::distance(_begin, _end);
		    n %= d;
		    if (n >= std::distance(super::base(), _end))
			std::advance(super::base_reference(), n - d);
		    else if (n < std::distance(super::base(), _begin))
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
make_ring_iterator(ITER begin, ITER end)
{
    return ring_iterator<ITER>(begin, end);
}

/************************************************************************
*  struct iterator_tuple<ITER, N>					*
************************************************************************/
template <class ITER, size_t N>
struct iterator_tuple
{
    typedef iterator_tuple<ITER, N-1>				tail;
    typedef boost::tuples::cons<ITER, typename tail::type>	type;

    static type	make(ITER iter)
		{
		    ITER	iter0 = iter;
		    return type(iter0, tail::make(++iter));
		}
};
template <class ITER>
struct iterator_tuple<ITER, 0>
{
    typedef boost::tuples::null_type	type;
    
    static type	make(ITER)		{ return type(); }
};

//! 連続するN個の反復子をまとめて1つのboost::tupleを作る
/*
 \param iter	前進反復子
 \return	iter, iter+1,..., iter+N-1 から成るboost::tuple
*/
template <size_t N, class ITER> typename iterator_tuple<ITER, N>::type
make_iterator_tuple(ITER iter)
{
    return iterator_tuple<ITER, N>::make(iter);
}

}	// namespace TU
#endif
