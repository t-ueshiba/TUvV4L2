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
 *  $Id: iterator.h,v 1.1 2012-09-15 03:59:19 ueshiba Exp $
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TUiterator_h
#define __TUiterator_h

#include <iterator>
#include <boost/iterator/transform_iterator.hpp>
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
*  class box_filter_iterator<Iterator>					*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param Iterator	コンテナ中の要素を指す定数反復子の型
*/
template <class Iterator>
class box_filter_iterator
    : public std::iterator<
		std::input_iterator_tag,
		typename std::iterator_traits<Iterator>::value_type>
{
  private:
    typedef std::iterator<
		std::input_iterator_tag,
		typename std::iterator_traits<Iterator>::value_type> super;
    
  public:
    typedef typename super::difference_type		difference_type;
    typedef typename super::value_type			value_type;
    typedef typename super::reference			reference;
    typedef typename super::pointer			pointer;

  public:
			box_filter_iterator(Iterator i, unsigned int w=0)
			    :_head(i), _tail(_head), _valid(true), _val()
			{
			    if (w > 0)
			    {
				_val = *_tail;
				
				while (--w > 0)
				    _val += *++_tail;
			    }
			}

    reference		operator *() const
			{
			    if (!_valid)
			    {
				_val += *_tail;
				_valid = true;
			    }
			    return _val;
			}
    
    const pointer	operator ->() const
			{
			    return &operator *();
			}
    
    box_filter_iterator&
			operator ++()
			{
			    _val -= *_head;
			    if (!_valid)
				_val += *_tail;
			    else
				_valid = false;
			    ++_head;
			    ++_tail;
			    return *this;
			}
    
    box_filter_iterator	operator ++(int)
			{
			    box_filter_iterator	tmp = *this;
			    operator ++();
			    return tmp;
			}
    
    bool		operator ==(const box_filter_iterator& a) const
			{
			    return _head == a._head;
			}
    
    bool		operator !=(const box_filter_iterator& a) const
			{
			    return !operator ==(a);
			}

  private:
    Iterator		_head;
    Iterator		_tail;
    mutable bool	_valid;	//!< _val が [_head, _tail] の総和ならtrue
    mutable value_type	_val;	//!< [_head, _tail) または [_head, _tail] の総和
};

//! box filter反復子を生成する
/*!
  \param iter	コンテナ中の要素を指す定数反復子の型
  \return	box filter反復子
*/
template <class Iterator> box_filter_iterator<Iterator>
make_box_filter_iterator(Iterator iter)
{
    return box_filter_iterator<Iterator>(iter);
}

/************************************************************************
*  class iir_filter_iterator<D, FWD, IN, COEFF, OUT>			*
************************************************************************/
//! データ列中の指定された要素に対してinfinite impulse response filterを適用した結果を返す反復子
/*!
  \param D	フィルタの階数
  \param FWD	前進フィルタならtrue, 後退フィルタならfalse
  \param IN	データ列中の要素を指す定数反復子の型
  \param COEFF	フィルタのz変換係数
*/
template <unsigned int D, bool FWD, class COEFF, class IN,
	  class OUT=typename std::iterator_traits<COEFF>::value_type>
class iir_filter_iterator : public std::iterator<std::input_iterator_tag, OUT>
{
  public:
    typedef OUT				value_type;
    typedef iir_filter_iterator		self;

  private:
    typedef boost::array<value_type, D>	buf_type;
    
  private:
    static inline value_type
		inner_product(COEFF c, const buf_type& buf,
			      value_type init, std::size_t i)
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

  public:
		iir_filter_iterator(IN in, COEFF ci, COEFF co)
		    :_in(in), _ci(ci), _co(co), _ibuf(), _obuf(), _i(0)
		{
		    _ibuf.fill(value_type(0));
		    _obuf.fill(value_type(0));
		}

    value_type	operator *()
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
				_ibuf[0] = *_in++;
			        TU_INPRO_2_1(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_2_0(c, _ibuf, out);
				_ibuf[0] = *_in++;
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
				_ibuf[1] = *_in++;
				TU_INPRO_2_0(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_2_1(c, _ibuf, out);
				_ibuf[1] = *_in++;
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
				_ibuf[0] = *_in++;
				TU_INPRO_4_1(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_0(c, _ibuf, out);
				_ibuf[0] = *_in++;
			    }
			    _obuf[0] = out;
			    _i = 1;
			    break;
			
			  case 1:
			    TU_INPRO_4_1(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[1] = *_in++;
				TU_INPRO_4_2(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_1(c, _ibuf, out);
				_ibuf[1] = *_in++;
			    }
			    _obuf[1] = out;
			    _i = 2;
			    break;

			  case 2:
			    TU_INPRO_4_2(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[2] = *_in++;
				TU_INPRO_4_3(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_2(c, _ibuf, out);
				_ibuf[2] = *_in++;
			    }
			    _obuf[2] = out;
			    _i = 3;
			    break;

			  default:
			    TU_INPRO_4_3(c, _obuf, out);
			    c = _ci;
			    if (FWD)
			    {
				_ibuf[3] = *_in++;
				TU_INPRO_4_0(c, _ibuf, out);
			    }
			    else
			    {
				TU_INPRO_4_3(c, _ibuf, out);
				_ibuf[3] = *_in++;
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
			
			      _ibuf[_i] = *_in++;	// 最新のデータを入力
			      _i = (_i + 1) % D;

			      out = inner_product(_ci, _ibuf, out, _i);
			      obuf = out;
			  }
			  else
			  {
			      out = inner_product(_ci, _ibuf, out, _i);

			      _obuf[_i] = out;
			      _ibuf[_i] = *_in++;	// 最新のデータを入力
			      _i = (_i + 1) % D;
			  }
		      }
		        break;
		    }
		    
		    return out;
		}
    
    self&	operator ++()			{return *this;}
    self&	operator ++(int)		{return *this;}
    bool	operator ==(const self& a)const	{return _in == a._in;}
    bool	operator !=(const self& a)const	{return !operator ==(a);}

  private:
    IN			_in;		//!< 入力データ列の現在位置を指す反復子
    const COEFF		_ci;		//!< 先頭の入力フィルタ係数を指す反復子
    const COEFF		_co;		//!< 先頭の出力フィルタ係数を指す反復子
    buf_type		_ibuf;		//!< 過去D時点の入力データ
    buf_type		_obuf;		//!< 過去D時点の出力データ
    std::size_t		_i;		//!< D時点前の出力データを指す反復子

#undef TU_INPRO_2_0
#undef TU_INPRO_2_1
#undef TU_INPRO_4_0
#undef TU_INPRO_4_1
#undef TU_INPRO_4_2
#undef TU_INPRO_4_3
};

template <unsigned int D, bool FWD, class OUT, class COEFF, class IN>
iir_filter_iterator<D, FWD, COEFF, IN, OUT>
make_iir_filter_iterator(IN in, COEFF ci, COEFF co)
{
    return iir_filter_iterator<D, FWD, COEFF, IN, OUT>(in, ci, co);
}

}
#endif
