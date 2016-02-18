/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2007.
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
  \file		IIRFilter.h
  \brief	各種infinite impulse response filterに関するクラスの定義と実装
*/
#ifndef	__TU_IIRFILTER_H
#define	__TU_IIRFILTER_H

#include <algorithm>
#include <array>
#include "TU/SeparableFilter2.h"

namespace TU
{
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
    typedef FixedSizedArray<T, D>			buf_type;
    typedef typename buf_type::const_iterator		buf_iterator;

    template <size_t D_, bool FWD_>
    struct selector				{enum {dim = D_, fwd = FWD_};};
    
  public:
    typedef typename super::value_type	value_type;
    typedef typename super::reference	reference;

    friend class	boost::iterator_core_access;

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
			    const auto	bi  = b + i;
			    value_type	val = *c * *bi;
			    for (auto p = bi; ++p != b + D; )
				val += *++c * *p;
			    for (auto p = b; p != bi; ++p)
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

    template <size_t D_>
    value_type	update(selector<D_, true>) const
		{
		    const auto	i = _i;
		    if (++_i == D)
			_i = 0;
		    _ibuf[i] = *super::base();
		    return (_obuf[i] = inpro(_co, _obuf.cbegin(),  i)
				     + inpro(_ci, _ibuf.cbegin(), _i));
		}
    
    template <size_t D_>
    value_type	update(selector<D_, false>) const
		{
		    const auto	val = inpro(_co, _obuf.cbegin(), _i)
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

/************************************************************************
*  class IIRFilter<D, T>						*
************************************************************************/
//! 片側Infinite Inpulse Response Filterを表すクラス
template <size_t D, class T=float> class IIRFilter
{
  public:
    typedef T				coeff_type;
    typedef std::array<T, D>		coeffs_type;
    
    IIRFilter&	initialize(const T c[D+D])				;
    void	limitsF(T& limit0F, T& limit1F, T& limit2F)	const	;
    void	limitsB(T& limit0B, T& limit1B, T& limit2B)	const	;
    template <class IN, class OUT>
    OUT		forward(IN ib, IN ie, OUT out)			const	;
    template <class IN, class OUT>
    OUT		backward(IN ib, IN ie, OUT out)			const	;
    
    const coeffs_type&	ci()			const	{return _ci;}
    const coeffs_type&	co()			const	{return _co;}
	
    static size_t	outLength(size_t inLength)	;

  private:
    coeffs_type	_ci;	//!< 入力フィルタ係数
    coeffs_type	_co;	//!< 出力フィルタ係数
};

//! フィルタのz変換係数をセットする
/*!
  \param c	z変換係数. z変換関数は，前進フィルタの場合は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
		後退フィルタの場合は
		\f[
		  H(z) = \frac{c_{0}z + c_{1}z^2 + \cdots + c_{D-1}z^D}
		       {1 - c_{D}z - c_{D+1}z^2 - \cdots - c_{2D-1}z^D}
		\f]
  \return	このフィルタ自身
*/
template <size_t D, class T> IIRFilter<D, T>&
IIRFilter<D, T>::initialize(const T c[D+D])
{
    std::copy(c,     c + D,	_ci.begin());
    std::copy(c + D, c + D + D,	_co.begin());

    return *this;
}

//! 特定の入力データ列に対して前進方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0F	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1F	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2F	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <size_t D, class T> void
IIRFilter<D, T>::limitsF(T& limit0F, T& limit1F, T& limit2F) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (size_t i = 0; i < D; ++i)
    {
	n0 +=	      _ci[i];
	d0 -=	      _co[i];
	n1 +=	    i*_ci[D-1-i];
	d1 -=	(i+1)*_co[D-1-i];
	n2 += (i-1)*i*_ci[D-1-i];
	d2 -= i*(i+1)*_co[D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0F =  x0;
    limit1F = -x1;
    limit2F =  x1 + x2;
}

//! 特定の入力データ列に対して後退方向にフィルタを適用した場合の極限値を求める
/*!
  \param limit0B	一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1B	傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2B	2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <size_t D, class T> void
IIRFilter<D, T>::limitsB(T& limit0B, T& limit1B, T& limit2B) const
{
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (size_t i = 0; i < D; ++i)
    {
	n0 +=	      _ci[i];
	d0 -=	      _co[i];
	n1 +=	(i+1)*_ci[i];
	d1 -=	(i+1)*_co[i];
	n2 += i*(i+1)*_ci[i];
	d2 -= i*(i+1)*_co[i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    limit0B = x0;
    limit1B = x1;
    limit2B = x1 + x2;
}

//! 前進方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
  \return	出力データ列の末尾の次を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::forward(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    
    return std::copy(make_iir_filter_iterator<D, true, value_type>(
			 ib, _ci.begin(), _co.begin()),
		     make_iir_filter_iterator<D, true, value_type>(
			 ie, _ci.begin(), _co.begin()),
		     out);
}
    
//! 後退方向にフィルタを適用する
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param oe	出力データ列の末尾の次を指す反復子
  \return	出力データ列の先頭を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> OUT
IIRFilter<D, T>::backward(IN ib, IN ie, OUT oe) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    
    return std::copy(make_iir_filter_iterator<D, false, value_type>(
			 ib, _ci.rbegin(), _co.rbegin()),
		     make_iir_filter_iterator<D, false, value_type>(
			 ie, _ci.rbegin(), _co.rbegin()),
		     oe);
}

//! 与えられた長さの入力データ列に対する出力データ列の長さを返す
/*!
  \param inLength	入力データ列の長さ
  \return		出力データ列の長さ
*/
template <size_t D, class T> inline size_t
IIRFilter<D, T>::outLength(size_t inLength)
{
    return inLength;
}
    
/************************************************************************
*  class BidirectionalIIRFilter<D, T>					*
************************************************************************/
//! 両側Infinite Inpulse Response Filterを表すクラス
template <size_t D, class T=float> class BidirectionalIIRFilter
{
  private:
    typedef IIRFilter<D, T>			iirf_type;
    
  public:
    typedef typename iirf_type::coeff_type	coeff_type;
    typedef typename iirf_type::coeffs_type	coeffs_type;
    
  //! 微分の階数
    enum Order
    {
	Zeroth,		//!< 0階微分
	First,		//!< 1階微分
	Second		//!< 2階微分
    };

    BidirectionalIIRFilter&
		initialize(const T cF[D+D], const T cB[D+D])		;
    BidirectionalIIRFilter&
		initialize(const T c[D+D], Order order)			;
    void	limits(T& limit0, T& limit1, T& limit2)		const	;
    template <class IN, class OUT>
    OUT		convolve(IN ib, IN ie, OUT out)			const	;

    const coeffs_type&	ciF()			const	{return _iirF.ci();}
    const coeffs_type&	coF()			const	{return _iirF.co();}
    const coeffs_type&	ciB()			const	{return _iirB.ci();}
    const coeffs_type&	coB()			const	{return _iirB.co();}

    static size_t	outLength(size_t inLength)	;
	
  private:
    IIRFilter<D, T>	_iirF;
    IIRFilter<D, T>	_iirB;
};

//! フィルタのz変換係数をセットする
/*!
  \param cF	前進z変換係数. z変換は 
		\f[
		  H^F(z^{-1}) = \frac{c^F_{D-1} + c^F_{D-2}z^{-1}
		  + c^F_{D-3}z^{-2} + \cdots
		  + c^F_{0}z^{-(D-1)}}{1 - c^F_{2D-1}z^{-1}
		  - c^F_{2D-2}z^{-2} - \cdots - c^F_{D}z^{-D}}
		\f]
		となる. 
  \param cB	後退z変換係数. z変換は
		\f[
		  H^B(z) = \frac{c^B_{0}z + c^B_{1}z^2 + \cdots + c^B_{D-1}z^D}
		       {1 - c^B_{D}z - c^B_{D+1}z^2 - \cdots - c^B_{2D-1}z^D}
		\f]
		となる.
*/
template <size_t D, class T> inline BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T cF[D+D], const T cB[D+D])
{
    _iirF.initialize(cF);
    _iirB.initialize(cB);
#ifdef TU_DEBUG
  /*T	limit0, limit1, limit2;
    limits(limit0, limit1, limit2);
    std::cerr << "limit0 = " << limit0 << ", limit1 = " << limit1
    << ", limit2 = " << limit2 << std::endl;*/
#endif
    return *this;
}

//! 両側フィルタのz変換係数をセットする
/*!
  \param c	前進方向z変換係数. z変換関数は
		\f[
		  H(z^{-1}) = \frac{c_{D-1} + c_{D-2}z^{-1} + c_{D-3}z^{-2} +
		  \cdots
		  + c_{0}z^{-(D-1)}}{1 - c_{2D-1}z^{-1} - c_{2D-2}z^{-2} -
		  \cdots - c_{D}z^{-D}}
		\f]
  \param order	フィルタの微分階数． #Zeroth または #Second ならば対称フィルタ
		として， #First ならば反対称フィルタとして自動的に後退方向の
		z変換係数を計算する． #Zeroth, #First, #Second のときに，それ
		ぞれ in(n) = 1, in(n) = n, in(n) = n^2 に対する出力が
		1, 1, 2になるよう，全体のスケールも調整される．
  \return	このフィルタ自身
*/
template <size_t D, class T> BidirectionalIIRFilter<D, T>&
BidirectionalIIRFilter<D, T>::initialize(const T c[D+D], Order order)
{
  // Compute 0th, 1st and 2nd derivatives of the forward z-transform
  // functions at z = 1.
    T	n0 = 0, d0 = 1, n1 = 0, d1 = 0, n2 = 0, d2 = 0;
    for (size_t i = 0; i < D; ++i)
    {
	n0 +=	      c[i];
	d0 -=	      c[D+i];
	n1 +=	    i*c[D-1-i];
	d1 -=	(i+1)*c[D+D-1-i];
	n2 += (i-1)*i*c[D-1-i];
	d2 -= i*(i+1)*c[D+D-1-i];
    }
    const T	x0 = n0/d0, x1 = (n1 - x0*d1)/d0,
		x2 = (n2 - 2*x1*d1 - x0*d2)/d0;
    
  // Compute denominators.
    T	cF[D+D], cB[D+D];
    for (size_t i = 0; i < D; ++i)
	cB[D+D-1-i] = cF[D+i] = c[D+i];

  // Compute nominators.
    if (order == First)	// Antisymmetric filter
    {
	const T	k = -0.5/x1;
	cF[D-1] = cB[D-1] = 0;
	for (size_t i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = -cF[i];				// i(n+D-1-i)
	}
    }
    else		// Symmetric filter
    {
	const T	k = (order == Second ? 1.0 / (x1 + x2)
				     : 1.0 / (2.0*x0 - c[D-1]));
	cF[D-1] = k*c[D-1];				// i(n)
	cB[D-1] = cF[D-1] * c[D];			// i(n+D)
	for (size_t i = 0; i < D-1; ++i)
	{
	    cF[i]     = k*c[i];				// i(n-D+1+i)
	    cB[D-2-i] = cF[i] + cF[D-1] * cF[D+1+i];	// i(n+D-1-i)
	}
    }

    return initialize(cF, cB);
}
    
//! 特定の入力データ列に対してフィルタを適用した場合の極限値を求める
/*!
  \param limit0		一定入力 in(n) = 1 を与えたときの出力極限値を返す．
  \param limit1		傾き一定入力 in(n) = n を与えたときの出力極限値を返す．
  \param limit2		2次入力 in(n) = n^2 を与えたときの出力極限値を返す．
*/
template <size_t D, class T> void
BidirectionalIIRFilter<D, T>::limits(T& limit0, T& limit1, T& limit2) const
{
    T	limit0F, limit1F, limit2F;
    _iirF.limitsF(limit0F, limit1F, limit2F);

    T	limit0B, limit1B, limit2B;
    _iirB.limitsB(limit0B, limit1B, limit2B);

    limit0 = limit0F + limit0B;
    limit1 = limit1F + limit1B;
    limit2 = limit2F + limit2B;
}

//! フィルタによる畳み込みを行う. 
/*!
  \param ib	入力データ列の先頭を指す反復子
  \param ie	入力データ列の末尾の次を指す反復子
  \param out	出力データ列の先頭を指す反復子
*/
template <size_t D, class T> template <class IN, class OUT> inline OUT
BidirectionalIIRFilter<D, T>::convolve(IN ib, IN ie, OUT out) const
{
    typedef typename std::iterator_traits<OUT>::value_type	value_type;
#if defined(SIMD)
    typedef Array<value_type,
		  Buf<value_type, 0,
		      simd::allocator<value_type> > >		buf_type;
#else
    typedef Array<value_type>					buf_type;
#endif
    typedef typename buf_type::iterator				buf_iterator;
    
    buf_type	bufF(std::distance(ib, ie)), bufB(bufF.size());

    _iirF.forward(ib, ie, bufF.begin());
    _iirB.backward(std::reverse_iterator<IN>(ie),
		   std::reverse_iterator<IN>(ib),
		   std::reverse_iterator<buf_iterator>(bufB.end()));
    return std::transform(bufF.cbegin(), bufF.cend(), bufB.cbegin(),
			  out, std::plus<value_type>());
}

//! 与えられた長さの入力データ列に対する出力データ列の長さを返す
/*!
  \param inLength	入力データ列の長さ
  \return		出力データ列の長さ
*/
template <size_t D, class T> inline size_t
BidirectionalIIRFilter<D, T>::outLength(size_t inLength)
{
    return IIRFilter<D, T>::outLength(inLength);
}
    
/************************************************************************
*  class BidirectionalIIRFilter2<D, T>					*
************************************************************************/
//! 2次元両側Infinite Inpulse Response Filterを表すクラス
template <size_t D, class T=float>
class BidirectionalIIRFilter2
    : public SeparableFilter2<BidirectionalIIRFilter<D, T> >
{
  private:
    typedef BidirectionalIIRFilter<D, T>	biir_type;
    typedef SeparableFilter2<biir_type>		super;

  public:
    typedef typename biir_type::coeff_type	coeff_type;
    typedef typename biir_type::coeffs_type	coeffs_type;
    typedef typename biir_type::Order		Order;
    
  public:
    BidirectionalIIRFilter2&
			initialize(const T cHF[], const T cHB[],
				   const T cVF[], const T cVB[])	;
    BidirectionalIIRFilter2&
			initialize(const T cHF[], Order orderH,
				   const T cVF[], Order orderV)		;

    using		super::convolve;
    using		super::filterH;
    using		super::filterV;
    
    const coeffs_type&	ciHF()		const	{return filterH().ciF();}
    const coeffs_type&	coHF()		const	{return filterH().coF();}
    const coeffs_type&	ciHB()		const	{return filterH().ciB();}
    const coeffs_type&	coHB()		const	{return filterH().coB();}
    const coeffs_type&	ciVF()		const	{return filterV().ciF();}
    const coeffs_type&	coVF()		const	{return filterV().coF();}
    const coeffs_type&	ciVB()		const	{return filterV().ciB();}
    const coeffs_type&	coVB()		const	{return filterV().coB();}
};
    
//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param cHB	横方向後退z変換係数
  \param cVF	縦方向前進z変換係数
  \param cVB	縦方向後退z変換係数
  \return	このフィルタ自身
*/
template <size_t D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], const T cHB[],
					  const T cVF[], const T cVB[])
{
    filterH().initialize(cHF, cHB);
    filterV().initialize(cVF, cVB);

    return *this;
}

//! フィルタのz変換係数をセットする
/*!
  \param cHF	横方向前進z変換係数
  \param orderH 横方向微分階数
  \param cVF	縦方向前進z変換係数
  \param orderV	縦方向微分階数
  \return	このフィルタ自身
*/
template <size_t D, class T> inline BidirectionalIIRFilter2<D, T>&
BidirectionalIIRFilter2<D, T>::initialize(const T cHF[], Order orderH,
					  const T cVF[], Order orderV)
{
    filterH().initialize(cHF, orderH);
    filterV().initialize(cVF, orderV);

    return *this;
}

}
#endif	// !__TU_IIRFILTER_H
