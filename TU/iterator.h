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
#include <boost/iterator_adaptors.hpp>
#include <boost/array.hpp>
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
	assignment_proxy(ITER const& iter, FUNC const& func)
	    :_iter(iter), _func(func)					{}

	template <class T>
	assignment_proxy&	operator =(const T& val)
				{
				    *_iter = _func(val);
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
	:super(iter), _func(func)					{}

    FUNC const&	functor() const
		{
		    return _func;
		}
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    const FUNC	_func;
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
	    :_iter(iter), _func(func)					{}

	template <class T>
	assignment2_proxy&	operator =(const T& val)
				{
				    _func(*_iter, val);
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
				     typename FUNC::second_argument_type,
				     boost::use_default,
				     detail::assignment2_proxy<FUNC, ITER> >
{
  private:
    typedef boost::iterator_adaptor<assignment2_iterator,
				    ITER,
				    typename FUNC::second_argument_type,
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
	:super(iter), _func(func)					{}

    FUNC const&	functor() const
		{
		    return _func;
		}
	
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}
    
  private:
    const FUNC	_func;
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
*  struct subiterator<ITER>						*
************************************************************************/
template <class ITER>
struct subiterator
    : boost::mpl::if_<
	boost::is_same<typename std::iterator_traits<ITER>::pointer,
		       typename std::iterator_traits<ITER>::value_type*>,
	typename std::iterator_traits<ITER>::value_type::iterator,
	typename std::iterator_traits<ITER>::value_type::const_iterator>
{
};

template <class TUPLE>
struct subiterator<boost::zip_iterator<TUPLE> >
{
    typedef boost::zip_iterator<
		typename boost::detail::tuple_impl_specific
			      ::tuple_meta_transform<
		    TUPLE, subiterator<boost::mpl::_1> >::type>		type;
};

/************************************************************************
*  class row_iterator<COL, FUNC, ROWBASE>				*
************************************************************************/
namespace detail
{
    template <class COL, class FUNC, class ROWBASE>
    class row_proxy
    {
      public:
	typedef COL	iterator;
	typedef COL	const_iterator;
	
      public:
	row_proxy(ROWBASE const& row, FUNC const& func)
	    :_row(row), _func(func)				{}
    
	iterator	begin()	const
			{
			    return iterator(_row->begin(), _func);
			}
	iterator	end() const
			{
			    return iterator(_row->end(), _func);
			}

      private:
	ROWBASE const&	_row;
	FUNC const&	_func;
    };

    template <class COL, class FUNC, class TUPLE>
    class row_proxy<COL, FUNC, boost::zip_iterator<TUPLE> >
    {
      public:
	typedef COL	iterator;
	typedef COL	const_iterator;
	
      private:
	struct begin_iterator
	{
	    template <class ROW>
	    struct apply
	    {
		typedef typename subiterator<ROW>::type	type;
	    };

	    template <class ROW> typename apply<ROW>::type
	    operator ()(ROW const& row)	const	{return row->begin();}
	};
	
	struct end_iterator
	{
	    template <class ROW>
	    struct apply
	    {
		typedef typename subiterator<ROW>::type	type;
	    };

	    template <class ROW> typename apply<ROW>::type
	    operator ()(ROW const& row)	const	{return row->end();}
	};
	
      public:
	row_proxy(boost::zip_iterator<TUPLE> const& row, FUNC const& func)
	    :_row(row), _func(func)				{}
    
	iterator	begin()	const
			{
			    return iterator(
				boost::make_zip_iterator(
				    boost::detail::tuple_impl_specific
				    ::tuple_transform(
					_row.get_iterator_tuple(),
					begin_iterator())),
				_func);
			}
	iterator	end() const
			{
			    return iterator(
				boost::make_zip_iterator(
				    boost::detail::tuple_impl_specific
				    ::tuple_transform(
					_row.get_iterator_tuple(),
					end_iterator())),
				_func);
			}
    
      private:
	boost::zip_iterator<TUPLE> const&	_row;
	FUNC const&				_func;
    };
}
    
//! コンテナを指す反復子に対して，取り出した値を変換したり値を変換してから格納する作業をサポートする反復子
/*
  \param COL		コンテナ中の個々の値に対して変換を行う反復子
  \param FUNC		変換関数
  \param ROWBASE	begin(), end()をサポートするコンテナを指す反復子
*/ 
template <class COL, class FUNC, class ROWBASE>
class row_iterator
    : public boost::iterator_adaptor<row_iterator<COL, FUNC, ROWBASE>,
				     ROWBASE,
				     detail::row_proxy<COL, FUNC, ROWBASE>,
				     boost::use_default,
				     detail::row_proxy<COL, FUNC, ROWBASE> >
{
  private:
    typedef boost::iterator_adaptor<
		row_iterator,
		ROWBASE,
		detail::row_proxy<COL, FUNC, ROWBASE>,
		boost::use_default,
		detail::row_proxy<COL, FUNC, ROWBASE> >	super;

  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
    row_iterator(ROWBASE const& row, FUNC const& func)
	:super(row), _func(func)				{}

    reference	operator [](size_t i) const
		{
		    return reference(super::base() + i, _func);
		}
    
  private:
    reference	dereference() const
		{
		    return reference(super::base(), _func);
		}

  private:
    FUNC const	_func;
};

template <template <class, class> class COL, class FUNC, class ROWBASE>
inline row_iterator<COL<FUNC, typename subiterator<ROWBASE>::type>,
		    FUNC, ROWBASE>
make_row_iterator(ROWBASE row, FUNC func)
{
    typedef typename subiterator<ROWBASE>::type	col_base_iterator;

    return row_iterator<COL<FUNC, col_base_iterator>,
			FUNC, ROWBASE>(row, func);
}

template <class FUNC, class ROWBASE>
inline row_iterator<boost::transform_iterator<
			FUNC, typename subiterator<ROWBASE>::type>,
		    FUNC, ROWBASE>
make_row_transform_iterator(ROWBASE row, FUNC func)
{
    typedef typename subiterator<ROWBASE>::type	col_base_iterator;

    return row_iterator<boost::transform_iterator<FUNC, col_base_iterator>,
			FUNC, ROWBASE>(row, func);
}

/************************************************************************
*  class vertical_iterator<ITER>					*
************************************************************************/
//! ランダムアクセス可能なコンテナの配列に対して，各コンテナ中の特定のindexに対応する要素にアクセスする反復子
/*
  \param ITER	ランダムアクセス可能なコンテナを指す反復子
*/
template <class ITER>
class vertical_iterator
    : public boost::iterator_adaptor<
		vertical_iterator<ITER>,			   // self
		ITER,						   // base
		typename std::iterator_traits<
		    typename subiterator<ITER>::type>::value_type, // value
		boost::use_default,				   // traversal
		typename std::iterator_traits<
		    typename subiterator<ITER>::type>::reference>  // reference	
{	
  private:
    typedef typename subiterator<ITER>::type		col_iterator;
    typedef typename boost::iterator_adaptor<
			vertical_iterator,
			ITER,
			typename std::iterator_traits<col_iterator>
				    ::value_type,
			boost::use_default,
			typename std::iterator_traits<col_iterator>
				    ::reference>	super;
				    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;
    
  public:
    vertical_iterator(ITER const& iter, size_t idx)
	:super(iter), _idx(idx)					{}

  private:
    bool	equal(const vertical_iterator& iter) const
		{
		    return super::equal() && (_idx == iter._idx);
		}
	
    reference	dereference() const
		{
		    col_iterator	col = super::base()->begin();
		    std::advance(col, _idx);
		    return *col;
		}
    
  private:
    const size_t	_idx;
};

//! vertical反復子を生成する
/*!
  \param iter	ランダムアクセス可能なコンテナを指す定数反復子
  \return	vertical反復子
*/
template <class ITER> vertical_iterator<ITER>
make_vertical_iterator(ITER iter, size_t idx)
{
    return vertical_iterator<ITER>(iter, idx);
}

/************************************************************************
*  class box_filter_iterator<ITER, T>					*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param ITER	コンテナ中の要素を指す定数反復子の型
*/
template <class ITER, class T=typename std::iterator_traits<ITER>::value_type>
class box_filter_iterator
    : public boost::iterator_adaptor<box_filter_iterator<ITER, T>,	// self
				     ITER,			// base
				     T,				// value_type
				     boost::single_pass_traversal_tag>
{
  private:
    typedef boost::iterator_adaptor<box_filter_iterator,
				    ITER,
				    T,
				    boost::single_pass_traversal_tag>	super;
    
  public:
    typedef typename super::difference_type	difference_type;
    typedef typename super::value_type		value_type;
    typedef typename super::pointer		pointer;
    typedef typename super::reference		reference;
    typedef typename super::iterator_category	iterator_category;

    friend class				boost::iterator_core_access;

  public:
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

  private:
    reference	dereference() const
		{
		    if (!_valid)
		    {
			_val += *super::base();
			_valid = true;
		    }
		    return _val;
		}
    
    void	increment()
		{
		    _val -= *_head;
		    if (!_valid)
			_val += *super::base();
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
template <class T, class ITER> box_filter_iterator<ITER, T>
make_box_filter_iterator(ITER iter)
{
    return box_filter_iterator<ITER, T>(iter);
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
template <unsigned int D, bool FWD, class COEFF, class ITER,
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
    typedef boost::array<T, D>				buf_type;

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
		    _ibuf.fill(value_type(0));
		    _obuf.fill(value_type(0));
		}

  private:
    static inline value_type
		inner_product(COEFF c, const buf_type& buf,
			      value_type init, size_t i)
		{
		    typedef typename buf_type::const_iterator	const_iterator;

		    const_iterator	p = buf.begin() + i;
		    for (const_iterator q = p; q != buf.end(); ++q)
			init += *c++ * *q;
		    for (const_iterator q = buf.begin(); q != p; ++q)
			init += *c++ * *q;
		    
		    return init;
		}

#define TU_INPRO_2_0(C, BUF, OUT)	OUT += *C   * BUF[0];	\
					OUT += *++C * BUF[1]
#define TU_INPRO_2_1(C, BUF, OUT)	OUT += *C   * BUF[1];	\
					OUT += *++C * BUF[0]
#define TU_INPRO_4_0(C, BUF, OUT)	OUT += *C   * BUF[0];	\
					OUT += *++C * BUF[1];	\
					OUT += *++C * BUF[2];	\
					OUT += *++C * BUF[3]
#define TU_INPRO_4_1(C, BUF, OUT)	OUT += *C   * BUF[1];	\
					OUT += *++C * BUF[2];	\
					OUT += *++C * BUF[3];	\
					OUT += *++C * BUF[0]
#define TU_INPRO_4_2(C, BUF, OUT)	OUT += *C   * BUF[2];	\
					OUT += *++C * BUF[3];	\
					OUT += *++C * BUF[0];	\
					OUT += *++C * BUF[1]
#define TU_INPRO_4_3(C, BUF, OUT)	OUT += *C   * BUF[3];	\
					OUT += *++C * BUF[0];	\
					OUT += *++C * BUF[1];	\
					OUT += *++C * BUF[2]

    value_type	dereference() const
		{
		    value_type	out = value_type(0);
		    
		    switch (D)
		    {
		      case 2:
		      {
			COEFF	c = _co;
			
			if (_i == 0)
			{
			    TU_INPRO_2_0(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[0] = *super::base();
			        TU_INPRO_2_1(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_2_0(c, _ibuf, out);
				_ibuf[0] = *super::base();
			    }
			    _obuf[0] = out;
			    _i = 1;
			}
			else
			{
			    TU_INPRO_2_1(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[1] = *super::base();
				TU_INPRO_2_0(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_2_1(c, _ibuf, out);
				_ibuf[1] = *super::base();
			    }
			    _obuf[1] = out;
			    _i = 0;
			}
		      }
			break;

		      case 4:
		      {
		        COEFF	c = _co;
			
			switch(_i)
			{
			  case 0:
			    TU_INPRO_4_0(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[0] = *super::base();
				TU_INPRO_4_1(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_0(c, _ibuf, out);
				_ibuf[0] = *super::base();
			    }
			    _obuf[0] = out;
			    _i = 1;
			    break;
			
			  case 1:
			    TU_INPRO_4_1(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[1] = *super::base();
				TU_INPRO_4_2(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_1(c, _ibuf, out);
				_ibuf[1] = *super::base();
			    }
			    _obuf[1] = out;
			    _i = 2;
			    break;

			  case 2:
			    TU_INPRO_4_2(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[2] = *super::base();
				TU_INPRO_4_3(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_2(c, _ibuf, out);
				_ibuf[2] = *super::base();
			    }
			    _obuf[2] = out;
			    _i = 3;
			    break;

			  default:
			    TU_INPRO_4_3(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[3] = *super::base();
				TU_INPRO_4_0(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_3(c, _ibuf, out);
				_ibuf[3] = *super::base();
			    }
			    _obuf[3] = out;
			    _i = 0;
			    break;
			}
		      }
			break;

		      default:
		      {
			  out = inner_product(_co, _obuf, out, _i);
			  if (FWD)
			  {
			      value_type&	obuf = _obuf[_i];
			
			      _ibuf[_i] = *super::base();
			      _i = (_i + 1) % D;

			      out = inner_product(_ci, _ibuf, out, _i);
			      obuf = out;
			  }
			  else
			  {
			      out = inner_product(_ci, _ibuf, out, _i);
			      _obuf[_i] = out;
			      
			      _ibuf[_i] = *super::base();
			      _i = (_i + 1) % D;
			  }
		      }
		        break;
		    }
		    
		    return out;
		}

#undef TU_INPRO_2_0
#undef TU_INPRO_2_1
#undef TU_INPRO_4_0
#undef TU_INPRO_4_1
#undef TU_INPRO_4_2
#undef TU_INPRO_4_3

  private:
    const COEFF		_ci;	//!< 先頭の入力フィルタ係数を指す反復子
    const COEFF		_co;	//!< 先頭の出力フィルタ係数を指す反復子
    mutable buf_type	_ibuf;	//!< 過去D時点の入力データ
    mutable buf_type	_obuf;	//!< 過去D時点の出力データ
    mutable size_t	_i;	//!< 最も古い(D時点前)入/出力データへのindex
};

//! infinite impulse response filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子
  \param ci	先頭の入力フィルタ係数を指す反復子
  \param ci	先頭の出力フィルタ係数を指す反復子
  \return	infinite impulse response filter反復子
*/
template <unsigned int D, bool FWD, class T, class COEFF, class ITER>
iir_filter_iterator<D, FWD, COEFF, ITER, T>
make_iir_filter_iterator(ITER iter, COEFF ci, COEFF co)
{
    return iir_filter_iterator<D, FWD, COEFF, ITER, T>(iter, ci, co);
}

}
#endif
