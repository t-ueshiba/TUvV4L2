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
    typedef TUPLE				iterator_tuple;
    
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

namespace detail
{
/************************************************************************
*  tuple_transform(TUPLE, TUPLE, FUNC)					*
************************************************************************/
template <class FUNC> inline boost::tuples::null_type
tuple_transform(boost::tuples::null_type, boost::tuples::null_type, FUNC)
{
    return boost::tuples::null_type();
}
template <class TUPLE, class FUNC>
inline typename boost::detail::tuple_impl_specific::
tuple_meta_transform<TUPLE, FUNC>::type
tuple_transform(TUPLE const& t1, TUPLE const& t2, FUNC func)
{ 
    typedef typename boost::detail::tuple_impl_specific::
	tuple_meta_transform<typename TUPLE::tail_type, FUNC>::type
						transformed_tail_type;

    return boost::tuples::cons<
	typename boost::mpl::apply<FUNC, typename TUPLE::head_type>::type,
	transformed_tail_type>(func(t1.get_head(), t2.get_head()),
			       tuple_transform(t1.get_tail(),
					       t2.get_tail(), func));
}

/************************************************************************
*  struct tuple_meta_transform2<TUPLE1, TUPLE2, BINARY_META_FUN>	*
************************************************************************/
template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
struct tuple_meta_transform2;
      
template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
struct tuple_meta_transform2_impl
{
    typedef boost::tuples::cons<
	typename boost::mpl::apply2<
	    typename boost::mpl::lambda<BINARY_META_FUN>::type,
	    typename TUPLE1::head_type,
	    typename TUPLE2::head_type>::type,
	typename tuple_meta_transform2<
	    typename TUPLE1::tail_type,
	    typename TUPLE2::tail_type,
	    BINARY_META_FUN>::type>				type;
};

template<class TUPLE1, class TUPLE2, class BINARY_META_FUN>
struct tuple_meta_transform2
    : boost::mpl::eval_if<
	boost::is_same<TUPLE1, boost::tuples::null_type>,
	boost::mpl::identity<boost::tuples::null_type>,
	tuple_meta_transform2_impl<TUPLE1, TUPLE2, BINARY_META_FUN> >
{
};

/************************************************************************
*  tuple_transform2<TUPLE1>(TUPLE2, FUNC)				*
************************************************************************/
template <class, class FUNC> inline boost::tuples::null_type
tuple_transform2(boost::tuples::null_type const&, FUNC)
{
    return boost::tuples::null_type();
}
template <class TUPLE1, class TUPLE2, class FUNC>
inline typename tuple_meta_transform2<TUPLE1, TUPLE2, FUNC>::type
tuple_transform2(TUPLE2 const& t, FUNC func)
{ 
    typedef typename tuple_meta_transform2<
	typename TUPLE1::tail_type,
	typename TUPLE2::tail_type, FUNC>::type	transformed_tail_type;

    return boost::tuples::cons<
	typename boost::mpl::apply2<
	    FUNC,
	    typename TUPLE1::head_type,
	    typename TUPLE2::head_type>::type,
	transformed_tail_type>(
	    func.template operator ()<typename TUPLE1::head_type>(
		boost::tuples::get<0>(t)),
	    tuple_transform2<typename TUPLE1::tail_type>(
		t.get_tail(), func));
}

}	// namespace detail

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
    template <class ROW,
	      class COL=boost::use_default, class ARG=boost::tuples::null_type>
    class row_proxy : public container<row_proxy<ROW, COL, ARG> >
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

	size_t
	size() const
	{
	    return std::distance(begin(), end());
	}

	size_t
	ncol() const
	{
	    return (size() ? begin()->size() : 0);
	}
	
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
    : public boost::transform_iterator<row2col<ROW>, ROW, boost::use_default,
				       typename subiterator<ROW>::value_type>
{
    typedef boost::transform_iterator<
		row2col<ROW>, ROW, boost::use_default,
		typename subiterator<ROW>::value_type>		super;

    vertical_iterator(ROW row, size_t idx)
	:super(row, row2col<ROW>(idx))				{}
    vertical_iterator(super const& iter)	:super(iter)	{}
    vertical_iterator&	operator =(super const& iter)
			{
			    super::operator =(iter);
			    return *this;
			}
};

template <class ROW> inline vertical_iterator<ROW>
make_vertical_iterator(ROW row, size_t idx)
{
    return vertical_iterator<ROW>(row, idx);
}

/************************************************************************
*  class column_iterator<A>						*
************************************************************************/
namespace detail
{
  //! 2次元配列の列を表す代理オブジェクト
  /*!
    \param A	2次元配列の型
  */
    template <class A>
    class column_proxy : public container<column_proxy<A> >
    {
      public:
      //! 定数反復子
	typedef vertical_iterator<typename A::const_iterator>
							const_iterator;
      //! 反復子
	typedef vertical_iterator<typename A::iterator>	iterator;
      //! 定数逆反復子
	typedef std::reverse_iterator<const_iterator>	const_reverse_iterator;
      //! 逆反復子
	typedef std::reverse_iterator<iterator>		reverse_iterator;
      //! 要素の型
	typedef typename iterator::value_type		value_type;
      //! 定数要素への参照
	typedef typename const_iterator::reference	const_reference;
      //! 要素への参照
	typedef typename iterator::reference		reference;
      //! 評価結果の型
	typedef column_proxy<typename A::result_type>	result_type;
      //! 成分の型
	typedef typename A::element_type		element_type;
    
      public:
      //! 2次元配列の列を表す代理オブジェクトを生成する.
      /*!
	\param a	2次元配列
	\param col	列を指定するindex
      */
	column_proxy(A& a, size_t col)	:_a(a), _col(col)		{}

      //! この列に他の配列を代入する.
      /*!
	\param expr	代入元の配列を表す式
	\return		この列
      */
	template <class E>
	column_proxy&		operator =(const container<E>& expr)
				{
#if !defined(NO_CHECK_SIZE)
				    if (expr().size() != size())
					throw std::logic_error("column_proxy<A>::operator =: mismatched size!");
#endif
				    std::copy(expr().begin(), expr().end(),
					      begin());
				    return *this;
				}
	
      //! 列の要素数すなわち行数を返す.
	size_t			size() const
				{
				    return _a.size();
				}
      //! 列の先頭要素を指す定数反復子を返す.
	const_iterator		begin() const
				{
				    return const_iterator(_a.begin(), _col);
				}
      //! 列の先頭要素を指す定数反復子を返す.
	const_iterator		cbegin() const
				{
				    return begin();
				}
      //! 列の先頭要素を指す反復子を返す.
	iterator		begin()
				{
				    return iterator(_a.begin(), _col);
				}
      //! 列の末尾を指す定数反復子を返す.
	const_iterator		end() const
				{
				    return const_iterator(_a.end(), _col);
				}
      //! 列の末尾を指す定数反復子を返す.
	const_iterator		cend() const
				{
				    return end();
				}
      //! 列の末尾を指す反復子を返す.
	iterator		end()
				{
				    return iterator(_a.end(), _col);
				}
      //! 列の末尾要素を指す定数逆反復子を返す.
	const_reverse_iterator	rbegin() const
				{
				    return const_reverse_iterator(end());
				}
      //! 列の末尾要素を指す定数逆反復子を返す.
	const_reverse_iterator	crbegin() const
				{
				    return rbegin();
				}
      //! 列の末尾要素を指す逆反復子を返す.
	reverse_iterator	rbegin()
				{
				    return reverse_iterator(end());
				}
      //! 列の先頭を指す定数逆反復子を返す.
	const_reverse_iterator	rend() const
				{
				    return const_reverse_iterator(begin());
				}
      //! 列の先頭を指す定数逆反復子を返す.
	const_reverse_iterator	crend() const
				{
				    return rend();
				}
      //! 列の先頭を指す逆反復子を返す.
	reverse_iterator	rend()
				{
				    return reverse_iterator(begin());
				}
      //! 列の定数要素にアクセスする.
      /*!
	\param i	要素を指定するindex
	\return		indexによって指定された定数要素
      */
	const_reference		operator [](size_t i) const
				{
				    return *(cbegin() + i);
				}
      //! 列の要素にアクセスする.
      /*!
	\param i	要素を指定するindex
	\return		indexによって指定された要素
      */
	reference		operator [](size_t i)
				{
				    return *(begin() + i);
				}

      private:
	A&		_a;	//!< 2次元配列への参照
	size_t const	_col;	//!< 列を指定するindex
    };
}

//! 2次元配列の列を指す反復子
/*!
  \param A	2次元配列の型
*/
template <class A>
class column_iterator
    : public boost::iterator_facade<column_iterator<A>,
				    detail::column_proxy<A>,
				    boost::random_access_traversal_tag,
				    detail::column_proxy<A> >
{
  private:
    typedef boost::iterator_facade<column_iterator,
				   detail::column_proxy<A>,
				   boost::random_access_traversal_tag,
				   detail::column_proxy<A> >	super;

  public:
    typedef typename super::value_type			value_type;
    typedef typename super::reference			reference;
    typedef typename super::pointer			pointer;
    typedef typename super::difference_type		difference_type;
    typedef typename super::iterator_category		iterator_category;
    
    friend class	boost::iterator_core_access;
    
  public:
    column_iterator(A& a, size_t col)	:_a(a), _col(col)		{}

    reference		dereference() const
			{
			    return reference(_a, _col);
			}
    bool		equal(const column_iterator& iter) const
			{
			    return _col == iter._col;
			}
    void		increment()
			{
			    ++_col;
			}
    void		decrement()
			{
			    --_col;
			}
    void		advance(difference_type n)
			{
			    _col += n;
			}
    difference_type	distance_to(const column_iterator& iter) const
			{
			    return iter._col - _col;
			}
    
  private:
    A&			_a;
    difference_type	_col;
};

//! 2次元配列の先頭の列を指す定数反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す定数反復子
*/
template <class A> column_iterator<const A>
column_cbegin(const A& a)
{
    return column_iterator<const A>(a, 0);
}
    
//! 2次元配列の先頭の列を指す反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す反復子
*/
template <class A> column_iterator<A>
column_begin(A& a)
{
    return column_iterator<A>(a, 0);
}
    
//! 2次元配列の末尾の列を指す定数反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す定数反復子
*/
template <class A> column_iterator<const A>
column_cend(const A& a)
{
    return column_iterator<const A>(a, a.ncol());
}
    
//! 2次元配列の末尾の列を指す反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す反復子
*/
template <class A> column_iterator<A>
column_end(A& a)
{
    return column_iterator<A>(a, a.ncol());
}
    
//! 2次元配列の末尾の列を指す定数逆反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す定数逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<const A> >
column_crbegin(const A& a)
{
    return std::reverse_iterator<column_iterator<const A> >(column_cend(a));
}
    
//! 2次元配列の末尾の列を指す逆反復子を返す.
/*!
  \param a	2次元配列
  \return	末尾の列を指す逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<A> >
column_rbegin(A& a)
{
    return std::reverse_iterator<column_iterator<A> >(column_end(a));
}
    
//! 2次元配列の先頭の列を指す定数逆反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す定数逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<const A> >
column_crend(const A& a)
{
    return std::reverse_iterator<column_iterator<const A> >(column_cbegin(a));
}
    
//! 2次元配列の先頭の列を指す逆反復子を返す.
/*!
  \param a	2次元配列
  \return	先頭の列を指す逆反復子
*/
template <class A> std::reverse_iterator<column_iterator<A> >
column_rend(A& a)
{
    return std::reverse_iterator<column_iterator<A> >(column_begin(a));
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
