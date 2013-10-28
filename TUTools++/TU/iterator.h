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
#include "TU/Array++.h"

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
	typedef typename std::iterator_traits<ITER>::reference	reference;

      public:
	assignment_proxy(ITER const& iter, FUNC const& func)
	    :_iter(iter), _func(func)					{}

	template <class T>
	reference	operator =(const T& val) const
			{
			    return *_iter = _func(val);
			}
	template <class T>
	reference	operator +=(const T& val) const
			{
			    return *_iter += _func(val);
			}
	template <class T>
	reference	operator -=(const T& val) const
			{
			    return *_iter -= _func(val);
			}
	template <class T>
	reference	operator *=(const T& val) const
			{
			    return *_iter *= _func(val);
			}
	template <class T>
	reference	operator /=(const T& val) const
			{
			    return *_iter /= _func(val);
			}
	template <class T>
	reference	operator &=(const T& val) const
			{
			    return *_iter &= _func(val);
			}
	template <class T>
	reference	operator |=(const T& val) const
			{
			    return *_iter |= _func(val);
			}
	template <class T>
	reference	operator ^=(const T& val) const
			{
			    return *_iter ^= _func(val);
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
	assignment2_proxy(ITER const& iter, FUNC const& func)
	    :_iter(iter), _func(func)			{}

	template <class T>
	void	operator =(const T& val)	const	{ _func(val, *_iter); }

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
	make_subiterator(fast_zip_iterator<TUPLE> row, size_t j)
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

template <class COL=boost::use_default, class ROW>
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

#if defined(MMX)
namespace mm
{
/************************************************************************
*  class row_vec_iterator<T, ROW>					*
************************************************************************/
template <class T, class ROW>
class row_vec_iterator
    : public boost::iterator_adaptor<row_vec_iterator<T, ROW>,
				     row_iterator<
					 fast_zip_iterator<
					     typename iterator_tuple<
						 ROW, vec<T>::size>::type>,
					 boost::transform_iterator<
					     tuple2vec<T>,
					     typename subiterator<
						 fast_zip_iterator<
						     typename iterator_tuple<
							 ROW,
							 vec<T>::size>::type>
						 >::type>,
					 tuple2vec<T> > >
{
  private:
    typedef fast_zip_iterator<
      typename iterator_tuple<ROW, vec<T>::size>::type>	row_zip_iterator;
    typedef boost::iterator_adaptor<
		row_vec_iterator,
		row_iterator<
		    row_zip_iterator,
		    boost::transform_iterator<
			tuple2vec<T>,
			typename subiterator<
			    row_zip_iterator>::type>,
		    tuple2vec<T> > >			super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;
    
  public:
    row_vec_iterator(ROW const& row)
	:super(make_row_transform_iterator(
		   make_fast_zip_iterator(
		       make_iterator_tuple<vec<T>::size>(row)),
		   tuple2vec<T>()))					{}

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
make_row_vec_iterator(ROW const& row)
{
    return row_vec_iterator<T, ROW>(row);
}

}	// namespace mm
#endif	// MMX
/************************************************************************
*  class box_filter_iterator<ITER>					*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param ITER	コンテナ中の要素を指す定数反復子の型
*/
template <class ITER>
class box_filter_iterator
    : public boost::iterator_adaptor<box_filter_iterator<ITER>,	// self
				     ITER,			// base
				     boost::use_default,	// value_type
				     boost::single_pass_traversal_tag>
{
  private:
    typedef boost::iterator_adaptor<box_filter_iterator,
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
		box_filter_iterator()
		    :super(), _head(super::base()), _valid(true), _val()
		{
		}
    
		box_filter_iterator(ITER const& iter, size_t w=0)
		    :super(iter), _head(iter), _valid(true), _val()
		{
		    if (w > 0)
		    {
			_val = *super::base();
				
			while (--w > 0)
			    _val += *++super::base_reference();
		    }
		}

    void	initialize(ITER const& iter, size_t w=0)
		{
		    super::base_reference() = iter;
		    _head = iter;
		    _valid = true;

		    if (w > 0)
		    {
			_val = *super::base();
				
			while (--w > 0)
			    _val += *++super::base_reference();
		    }
		}
    
  private:
    reference	dereference() const
		{
		    if (!_valid)
		    {
			(_val -= *_head) += *super::base();
			_valid = true;
		    }
		    return _val;
		}
    
    void	increment()
		{
		    if (!_valid)
			(_val -= *_head) += *super::base();
		    else
			_valid = false;
		    ++_head;
		    ++super::base_reference();
		}

  private:
    ITER		_head;
    mutable bool	_valid;	// _val が [_head, base()] の総和ならtrue
    mutable value_type	_val;	// [_head, base()) or [_head, base()] の総和
};

//! box filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \return	box filter反復子
*/
template <class ITER> box_filter_iterator<ITER>
make_box_filter_iterator(ITER iter, size_t w=0)
{
    return box_filter_iterator<ITER>(iter, w);
}

/************************************************************************
*  class fir_filter_iterator<D, COEFF, ITER, T>				*
************************************************************************/
//! データ列中の指定された要素に対してfinite impulse response filterを適用した結果を返す反復子
/*!
  \param D	フィルタの階数
  \param COEFF	フィルタのz変換係数
  \param ITER	データ列中の要素を指す定数反復子の型
  \param T	フィルタ出力の型
*/
template <size_t D, class COEFF, class ITER,
	  class T=typename std::iterator_traits<COEFF>::value_type>
class fir_filter_iterator
    : public boost::iterator_adaptor<
		fir_filter_iterator<D, COEFF, ITER, T>,		// self
		ITER,						// base
		T,						// value_type
		boost::forward_traversal_tag,			// traversal
		T>						// reference
{
  private:
    typedef boost::iterator_adaptor<
		fir_filter_iterator, ITER, T,
		boost::forward_traversal_tag, T>	super;
    typedef Array<T, FixedSizedBuf<T, D, true> >	buf_type;
    typedef typename buf_type::const_iterator		buf_iterator;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		fir_filter_iterator(ITER const& iter, COEFF c)
		    :super(iter), _c(c), _ibuf(), _i(0)
		{
		    for (; _i != D - 1; ++_i, ++super::base_reference())
			_ibuf[_i] = *super::base();
		}
		fir_filter_iterator(ITER const& iter)
		    :super(iter), _c(), _ibuf(), _i(0)
		{
		}

  private:
    reference	dereference() const
		{
		    value_type		val = _c[D-1]
					    * (_ibuf[_i] = *super::base());
		    COEFF		c   = _c;
		    buf_iterator	bi  = _ibuf.cbegin() + _i;
		    for (buf_iterator p = bi; ++p != _ibuf.cend(); ++c)
			val += *c * *p;
		    for (buf_iterator p = _ibuf.cbegin(); p != bi; ++p, ++c)
			val += *c * *p;

		    return val;
		}
    void	increment()
		{
		    ++super::base_reference();
		    if (++_i == D)
			_i = 0;
		}

  private:
    const COEFF		_c;	//!< 先頭のフィルタ係数を指す反復子
    mutable buf_type	_ibuf;	//!< 過去D時点の入力データ
    size_t		_i;	//!< 最新の入力データへのindex
};

//! finite impulse response filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \param c	先頭の入力フィルタ係数を指す反復子
  \return	finite impulse response filter反復子
*/
template <size_t D, class T, class COEFF, class ITER>
fir_filter_iterator<D, COEFF, ITER, T>
make_fir_filter_iterator(ITER iter, COEFF c)
{
    return fir_filter_iterator<D, COEFF, ITER, T>(iter, c);
}

//! finite impulse response filter反復子(終端)を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \return	finite impulse response filter反復子(終端)
*/
template <size_t D, class T, class COEFF, class ITER>
fir_filter_iterator<D, COEFF, ITER, T>
make_fir_filter_iterator(ITER iter)
{
    return fir_filter_iterator<D, COEFF, ITER, T>(iter);
}

/************************************************************************
*  class iir_filter_iterator<D, FWD, COEFF, ITER, T>			*
************************************************************************/
//! データ列中の指定された要素に対してinfinite impulse response filterを適用した結果を返す反復子
/*!
  \param D	フィルタの階数
  \param FWD	前進フィルタならtrue, 後退フィルタならfalse
  \param COEFF	フィルタのz変換係数
  \param ITER	データ列中の要素を指す定数反復子の型
  \param T	フィルタ出力の型
*/
template <size_t D, bool FWD, class COEFF, class ITER,
	  class T=typename std::iterator_traits<COEFF>::value_type>
class iir_filter_iterator
    : public boost::iterator_adaptor<
		iir_filter_iterator<D, FWD, COEFF, ITER, T>,	// self
		ITER,						// base
		T,						// value_type
		boost::single_pass_traversal_tag,		// traversal
		T>						// reference
{
  private:
    typedef boost::iterator_adaptor<
		iir_filter_iterator, ITER, T,
		boost::single_pass_traversal_tag, T>	super;
    typedef Array<T, FixedSizedBuf<T, D, true> >	buf_type;
    typedef typename buf_type::const_iterator		buf_iterator;

    template <size_t DD, bool FF>
    struct selector				{enum {dim = DD, fwd = FF};};
    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
		iir_filter_iterator(ITER const& iter, COEFF ci, COEFF co)
		    :super(iter), _ci(ci), _co(co), _ibuf(), _obuf(), _i(0)
		{
		    std::fill(_ibuf.begin(), _ibuf.end(), value_type(0));
		    std::fill(_obuf.begin(), _obuf.end(), value_type(0));
		}

  private:
    static value_type	inpro(COEFF c, buf_iterator b, size_t i)
			{
			    buf_iterator	bi  = b + i;
			    value_type		val = *c * *bi;
			    for (buf_iterator p = bi; ++p != b + D; )
				val += *++c * *p;
			    for (buf_iterator p = b; p != bi; ++p)
				val += *++c * *p;

			    return val;
			}
    static value_type	inpro_2_0(COEFF c, buf_iterator b)
			{
			    return *c * *b + *(c+1) * *(b+1);
			}
    static value_type	inpro_2_1(COEFF c, buf_iterator b)
			{
			    return *c * *(b+1) + *(c+1) * *b;
			}
    static value_type	inpro_4_0(COEFF c, buf_iterator b)
			{
			    return *c	  * *b	   + *(c+1) * *(b+1)
				 + *(c+2) * *(b+2) + *(c+3) * *(b+3);
			}
    static value_type	inpro_4_1(COEFF c, buf_iterator b)
			{
			    return *c	  * *(b+1) + *(c+1) * *(b+2)
				 + *(c+2) * *(b+3) + *(c+3) * *b;
			}
    static value_type	inpro_4_2(COEFF c, buf_iterator b)
			{
			    return *c	  * *(b+2) + *(c+1) * *(b+3)
				 + *(c+2) * *b	   + *(c+3) * *(b+1);
			}
    static value_type 	inpro_4_3(COEFF c, buf_iterator b)
			{
			    return *c	  * *(b+3) + *(c+1) * *b
				 + *(c+2) * *(b+1) + *(c+3) * *(b+2);
			}

    template <size_t DD>
    value_type	update(selector<DD, true>) const
		{
		    size_t	i = _i;
		    if (++_i == D)
			_i = 0;
		    _ibuf[i] = *super::base();
		    return (_obuf[i] = inpro(_co, _obuf.cbegin(),  i)
				     + inpro(_ci, _ibuf.cbegin(), _i));
		}
    
    template <size_t DD>
    value_type	update(selector<DD, false>) const
		{
		    value_type	val = inpro(_co, _obuf.cbegin(), _i)
				    + inpro(_ci, _ibuf.cbegin(), _i);
		    _obuf[_i] = val;
		    _ibuf[_i] = *super::base();
		    if (++_i == D)
			_i = 0;
		    return val;
		}

    value_type	update(selector<2, true>) const
		{
		    value_type	val;
		    
		    if (_i == 0)
		    {
			_i = 1;
			_ibuf[0] = *super::base();
			_obuf[0] = val
				 = inpro_2_0(_co, _obuf.cbegin())
				 + inpro_2_1(_ci, _ibuf.cbegin());
		    }
		    else
		    {
			_i = 0;
			_ibuf[1] = *super::base();
			_obuf[1] = val
				 = inpro_2_1(_co, _obuf.cbegin())
				 + inpro_2_0(_ci, _ibuf.cbegin());
		    }

		    return val;
		}

    value_type	update(selector<2, false>) const
		{
		    value_type	val;
		    
		    if (_i == 0)
		    {
			_i = 1;
			_obuf[0] = val
				 = inpro_2_0(_co, _obuf.cbegin())
				 + inpro_2_0(_ci, _ibuf.cbegin());
			_ibuf[0] = *super::base();
		    }
		    else
		    {
			_i = 0;
			_obuf[1] = val
				 = inpro_2_1(_co, _obuf.cbegin())
				 + inpro_2_1(_ci, _ibuf.cbegin());
			_ibuf[1] = *super::base();
		    }

		    return val;
		}

    value_type	update(selector<4, true>) const
		{
		    value_type	val;
		    
		    switch (_i)
		    {
		      case 0:
			_i = 1;
			_ibuf[0] = *super::base();
			_obuf[0] = val
				 = inpro_4_0(_co, _obuf.cbegin())
				 + inpro_4_1(_ci, _ibuf.cbegin());
			break;
		      case 1:
			_i = 2;
			_ibuf[1] = *super::base();
			_obuf[1] = val
				 = inpro_4_1(_co, _obuf.cbegin())
				 + inpro_4_2(_ci, _ibuf.cbegin());
			break;
		      case 2:
			_i = 3;
			_ibuf[2] = *super::base();
			_obuf[2] = val
				 = inpro_4_2(_co, _obuf.cbegin())
				 + inpro_4_3(_ci, _ibuf.cbegin());
			break;
		      default:
			_i = 0;
			_ibuf[3] = *super::base();
			_obuf[3] = val
				 = inpro_4_3(_co, _obuf.cbegin())
				 + inpro_4_0(_ci, _ibuf.cbegin());
			break;
		    }

		    return val;
		}

    value_type	update(selector<4, false>) const
		{
		    value_type	val;
		    
		    switch (_i)
		    {
		      case 0:
			_i = 1;
			_obuf[0] = val
				 = inpro_4_0(_co, _obuf.cbegin())
				 + inpro_4_0(_ci, _ibuf.cbegin());
			_ibuf[0] = *super::base();
			break;
		      case 1:
			_i = 2;
			_obuf[1] = val
				 = inpro_4_1(_co, _obuf.cbegin())
				 + inpro_4_1(_ci, _ibuf.cbegin());
			_ibuf[1] = *super::base();
			break;
		      case 2:
			_i = 3;
			_obuf[2] = val
				 = inpro_4_2(_co, _obuf.cbegin())
				 + inpro_4_2(_ci, _ibuf.cbegin());
			_ibuf[2] = *super::base();
			break;
		      default:
			_i = 0;
			_obuf[3] = val
				 = inpro_4_3(_co, _obuf.cbegin())
				 + inpro_4_3(_ci, _ibuf.cbegin());
			_ibuf[3] = *super::base();
			break;
		    }

		    return val;
		}

    reference	dereference() const
		{
		    return update(selector<D, FWD>());
		}
    
  private:
    const COEFF		_ci;	//!< 先頭の入力フィルタ係数を指す反復子
    const COEFF		_co;	//!< 先頭の出力フィルタ係数を指す反復子
    mutable buf_type	_ibuf;	//!< 過去D時点の入力データ
    mutable buf_type	_obuf;	//!< 過去D時点の出力データ
    mutable size_t	_i;
};

//! infinite impulse response filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \param ci	先頭の入力フィルタ係数を指す反復子
  \param ci	先頭の出力フィルタ係数を指す反復子
  \return	infinite impulse response filter反復子
*/
template <size_t D, bool FWD, class T, class COEFF, class ITER>
iir_filter_iterator<D, FWD, COEFF, ITER, T>
make_iir_filter_iterator(ITER iter, COEFF ci, COEFF co)
{
    return iir_filter_iterator<D, FWD, COEFF, ITER, T>(iter, ci, co);
}

}	// namespace TU
#endif
