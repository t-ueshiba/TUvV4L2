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
  \file		GuidedFilter.h
  \brief	guided filterに関するクラスの定義と実装
*/
#ifndef __TU_GUIDEDFILTER_H
#define __TU_GUIDEDFILTER_H

#include "TU/BoxFilter.h"

namespace TU
{
/************************************************************************
*  class GuidedFilter<T>						*
************************************************************************/
//! 1次元guided filterを表すクラス
template <class T>
class GuidedFilter : public BoxFilter
{
  public:
    typedef T		value_type;
    
    class Params
	: public boost::tuple<value_type, value_type, value_type, value_type>
    {
      private:
	typedef boost::tuple<value_type, value_type,
			     value_type, value_type>	super;
	
      public:
	typedef boost::tuple<value_type, value_type>	result_type;
	
	struct Init
	{
	    typedef Params	result_type;
	    
	    template <class TUPLE_>
	    result_type	operator ()(TUPLE_ t) const
			{
			    return result_type(boost::get<0>(t),
					       boost::get<1>(t));
			}
	};
	
      public:
	Params(value_type p=0, value_type g=0)	:super(p, g, p*g, g*g)	{}
    
	result_type	coeffs(size_t n, value_type e) const
			{
			    using namespace	boost;

			    value_type	a = (e == 0 ? 1 :
					     (n*get<2>(*this)
					      - get<0>(*this)*get<1>(*this)) /
					     (n*(get<3>(*this) + n*e)
					      -  get<1>(*this)*get<1>(*this)));
			    value_type	b = (get<0>(*this) - a*get<1>(*this))/n;

			    return result_type(a, b);
			}
    };
    
    class SimpleParams : public boost::tuple<value_type, value_type>
    {
      private:
	typedef boost::tuple<value_type, value_type>	super;
	
      public:
	typedef boost::tuple<value_type, value_type>	result_type;
	
	struct Init
	{
	    typedef SimpleParams	result_type;
	    
	    template <class IN_>
	    result_type	operator ()(IN_ p) const
			{
			    return result_type(p);
			}
	};
	
      public:
	SimpleParams(value_type p=0) :super(p, p*p)	{}

	result_type	coeffs(size_t n, value_type e) const
			{
			    using namespace	boost;
			    
			    value_type	var = n*get<1>(*this)
					    - get<0>(*this)*get<0>(*this);
			    value_type	a = (e == 0 ? 1 : var/(var + n*n*e));
			    value_type	b = (get<0>(*this) - a*get<1>(*this))/n;
			    
			    return result_type(a, b);
			}
    };

    class Coeff : public boost::tuple<value_type, value_type>
    {
      public:
	typedef boost::tuple<value_type, value_type>	super;
	
	template <class PARAMS_>
	class Init
	{
	  public:
	    typedef PARAMS_	argument_type;
	    typedef Coeff	result_type;

	  public:
	    Init(size_t n, value_type e) :_n(n), _e(e)			{}

	    result_type	operator ()(const argument_type& params) const
			{
			    return result_type(params.coeffs(_n, _e));
			}

	  private:
	    const size_t	_n;
	    const value_type	_e;
	};

	class Trans
	{
	  public:
	    typedef Coeff	first_argument_type;
	    typedef void	result_type;
	
	  public:
	    Trans(size_t n)	:_n(n)					{}

	    template <class TUPLE>
	    result_type	operator ()(first_argument_type coeffs, TUPLE t) const
			{
			    boost::get<1>(t)
				= coeffs.trans(boost::get<0>(t), _n);
			}
	
	  private:
	    const size_t	_n;
	};

      public:
	Coeff()								{}
	Coeff(const super& initial_coeffs)	:super(initial_coeffs)	{}
    
	template <class GUIDE_>
	value_type	trans(GUIDE_ g, size_t n) const
			{
			    using namespace	boost;
			    
			    return (get<0>(*this)*g + get<1>(*this))/n;
			}
    };
    
  private:
    typedef BoxFilter	super;

  public:
    GuidedFilter(size_t w, value_type e) :super(w), _e(e)	{}

    value_type		epsilon()			const	{return _e;}
    GuidedFilter&	setEpsilon(value_type e)
			{
			    _e = e;
			    return *this;
			}
    
    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie,
			 GUIDE gb, GUIDE ge, OUT out)	const	;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)		const	;

    size_t	outLength(size_t inLen) const
		{
		    return inLen + 2 - 2*winSize();
		}
    
  private:
    value_type	_e;
};

//! 1次元入力データ列と1次元ガイドデータ列にguided filterを適用する
/*!
  \param ib	1次元入力データ列の先頭を示す反復子
  \param ie	1次元入力データ列の末尾の次を示す反復子
  \param gb	1次元ガイドデータ列の先頭を示す反復子
  \param ge	1次元ガイドデータ列の末尾の次を示す反復子
  \param out	guided filterを適用したデータの出力先を示す反復子
*/
template <class T> template <class IN, class GUIDE, class OUT> void
GuidedFilter<T>::convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out) const
{
    typedef Array<Coeff>				carray_type;
    typedef typename Params::Init			params_init;
    typedef typename Coeff::template Init<Params>	coeff_init;
    typedef typename Coeff::Trans			coeff_trans;
    
  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(super::outLength(std::distance(ib, ie)));
    super::convolve(boost::make_transform_iterator(
			make_fast_zip_iterator(boost::make_tuple(ib, gb)),
			params_init()),
		    boost::make_transform_iterator(
			make_fast_zip_iterator(boost::make_tuple(ie, ge)),
			params_init()),
		    make_assignment_iterator(c.begin(),
					     coeff_init(winSize(), _e)));

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    std::advance(gb, winSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment2_iterator(
			make_fast_zip_iterator(boost::make_tuple(out, gb)),
			coeff_trans(winSize())));
}

//! 1次元入力データ列にguided filterを適用する
/*!
  ガイドデータ列は与えられた1次元入力データ列に同一とする．
  \param ib	1次元入力データ列の先頭を示す反復子
  \param ie	1次元入力データ列の末尾の次を示す反復子
  \param out	guided filterを適用したデータの出力先を示す反復子
  \param w	box filterのウィンドウ幅
  \param e	正則化のための微小定数
*/
template <class T> template <class IN, class OUT> void
GuidedFilter<T>::convolve(IN ib, IN ie, OUT out) const
{
    typedef Array<Coeff>				carray_type;
    typedef typename SimpleParams::Init			params_init;
    typedef typename Coeff::template Init<SimpleParams>	coeff_init;
    typedef typename Coeff::Trans			coeff_trans;

  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(super::outLength(std::distance(ib, ie)));
    super::convolve(boost::make_transform_iterator(ib, params_init()),
		    boost::make_transform_iterator(ie, params_init()),
		    make_assignment_iterator(c.begin(),
					     coeff_init(winSize(), _e)));

  // 係数ベクトルの平均値を求め，それによって入力データ列を線型変換する．
    std::advance(ib, winSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment2_iterator(
			make_fast_zip_iterator(boost::make_tuple(ib, out)),
			coeff_trans(winSize())));
}

/************************************************************************
*  class GuidedFilter2<T>						*
************************************************************************/
//! 2次元guided filterを表すクラス
template <class T>
class GuidedFilter2 : private BoxFilter2
{
  public:
    typedef T							value_type;
    typedef typename GuidedFilter<value_type>::Params		Params;
    typedef typename GuidedFilter<value_type>::SimpleParams	SimpleParams;
    typedef typename GuidedFilter<value_type>::Coeff		Coeff;
    
  private:
    typedef BoxFilter2						super;

  public:
    GuidedFilter2(size_t wrow, size_t wcol, value_type e)
	:super(wrow, wcol), _e(e)				{}

    using	super::grainSize;
    using	super::setGrainSize;
    using	super::rowWinSize;
    using	super::colWinSize;
    using	super::setRowWinSize;
    using	super::setColWinSize;
    
    value_type	epsilon()		const	{return _e;}
    GuidedFilter2&
		setEpsilon(value_type e)	{ _e = e; return *this; }
    
    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie,
			 GUIDE gb, GUIDE ge, OUT out)	const	;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)		const	;
    
  private:
    value_type	_e;
};

//! 2次元入力データと2次元ガイドデータにguided filterを適用する
/*!
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param gb	2次元ガイドデータの先頭の行を示す反復子
  \param ge	2次元ガイドデータの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
*/
template <class T> template <class IN, class GUIDE, class OUT> void
GuidedFilter2<T>::convolve(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out) const
{
    typedef Array2<Array<Coeff> >			carray2_type;
    typedef typename Params::Init			params_init;
    typedef typename Coeff::template Init<Params>	coeff_init;
    typedef typename Coeff::Trans			coeff_trans;

    if (ib == ie)
	return;
    
    const size_t	n = rowWinSize() * colWinSize();
    carray2_type	c(super::outRowLength(std::distance(ib, ie)),
			  super::outColLength(std::distance(ib->begin(),
							    ib->end())));

    super::convolve(make_row_transform_iterator(
			make_fast_zip_iterator(boost::make_tuple(ib, gb)),
			params_init()),
		    make_row_transform_iterator(
			make_fast_zip_iterator(boost::make_tuple(ie, ge)),
			params_init()),
		    make_row_uniarg_iterator<assignment_iterator>(
			c.begin(), coeff_init(n, _e)));

    std::advance(gb,  rowWinSize() - 1);
    std::advance(out, rowWinSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_row_uniarg_iterator<assignment2_iterator>(
			colWinSize() - 1, 0,
			make_fast_zip_iterator(boost::make_tuple(gb, out)),
			coeff_trans(n)));
}

//! 2次元入力データにguided filterを適用する
/*!
  ガイドデータは与えられた2次元入力データに同一とする．
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
*/
template <class T> template <class IN, class OUT> void
GuidedFilter2<T>::convolve(IN ib, IN ie, OUT out) const
{
    typedef Array2<Array<Coeff> >			carray2_type;
    typedef typename SimpleParams::Init			params_init;
    typedef typename Coeff::template Init<SimpleParams>	coeff_init;
    typedef typename Coeff::Trans			coeff_trans;

    if (ib == ie)
	return;
    
    const size_t	n = rowWinSize() * colWinSize();
    carray2_type	c(super::outRowLength(std::distance(ib, ie)),
			  super::outColLength(std::distance(ib->begin(),
							    ib->end())));

    super::convolve(make_row_transform_iterator(ib, params_init()),
		    make_row_transform_iterator(ie, params_init()),
		    make_row_uniarg_iterator<assignment_iterator>(
			c.begin(), coeff_init(n, _e)));
    
    std::advance(ib,  rowWinSize() - 1);
    std::advance(out, rowWinSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_row_uniarg_iterator<assignment2_iterator>(
			colWinSize() - 1, 0,
			make_fast_zip_iterator(boost::make_tuple(ib, out)),
			coeff_trans(n)));
}

}
#endif	// !__TU_GUIDEDFILTER_H
