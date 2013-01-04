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
    
    class Coeff
    {
      public:
	class Params
	{
	  public:
	    Params()	:_v0(0), _v1(0), _v2(0), _v3(0)			{}
    
	    template <class TUPLE>
	    Params(const TUPLE& t)
		:_v0(boost::get<0>(t)), _v1(boost::get<1>(t)),
		 _v2(_v0*_v1), _v3(_v1*_v1)				{}

	    Params&		operator +=(const Params& v)
				{
				    _v0 += v._v0;
				    _v1 += v._v1;
				    _v2 += v._v2;
				    _v3 += v._v3;
				    return *this;
				}
    
	    Params&		operator -=(const Params& v)
				{
				    _v0 -= v._v0;
				    _v1 -= v._v1;
				    _v2 -= v._v2;
				    _v3 -= v._v3;
				    return *this;
				}

	    value_type		a(size_t n, value_type e) const
				{
				    return (e == 0 ? 1 :
					    (n*_v2 - _v0*_v1) /
					    (n*(_v3 + n*e) - _v1*_v1));
				}

	    value_type		b(size_t n, value_type a) const
				{
				    return (_v0 - a*_v1) / n;
				}

	  private:
	    value_type		_v0, _v1, _v2, _v3;
	};
    
	class SimpleParams
	{
	  public:
	    SimpleParams()	:_v0(0), _v1(0)			{}

	    template <class IN>
	    SimpleParams(IN p)	:_v0(p), _v1(_v0*_v0)		{}

	    SimpleParams&	operator +=(const SimpleParams& v)
				{
				    _v0 += v._v0;
				    _v1 += v._v1;
				    return *this;
				}
    
	    SimpleParams&	operator -=(const SimpleParams& v)
				{
				    _v0 -= v._v0;
				    _v1 -= v._v1;
				    return *this;
				}

	    value_type		a(size_t n, value_type e) const
				{
				    value_type	var = n*_v1 - _v0*_v0;
				    return (e == 0 ? 1 : var/(var + n*n*e));
				}

	    value_type		b(size_t n, value_type a) const
				{
				    return (_v0 - a*_v0) / n;
				}

	  private:
	    value_type		_v0, _v1;
	};

	template <class PARAMS>
	class Init
	{
	  public:
	    typedef PARAMS	argument_type;
	    typedef Coeff	result_type;

	  public:
	    Init(size_t n, value_type e) :_n(n), _e(e)			{}

	    result_type	operator ()(const argument_type& params) const
			{
			    return result_type(params, _n, _e);
			}

	  private:
	    const size_t	_n;
	    const value_type	_e;
	};

	class Trans
	{
	  public:
	    typedef Coeff	second_argument_type;
	    typedef void	result_type;
	
	  public:
	    Trans(size_t n)	:_n(n)					{}

	    template <class TUPLE>
	    result_type	operator ()(TUPLE t,
				    const second_argument_type& coeffs) const
			{
			    boost::get<0>(t)
				= coeffs.trans(boost::get<1>(t), _n);
			}
	
	  private:
	    const size_t	_n;
	};

      public:
	Coeff()								{}

	template <class PARAMS>
	Coeff(const PARAMS& params, size_t n, value_type e)
	    :_a(params.a(n, e)), _b(params.b(n, _a))			{}
    
	Coeff&		operator +=(const Coeff& c)
			{
			    _a += c._a;
			    _b += c._b;
			    return *this;
			}
    
	Coeff&		operator -=(const Coeff& c)
			{
			    _a -= c._a;
			    _b -= c._b;
			    return *this;
			}
    
	template <class GUIDE>
	value_type	trans(GUIDE g, size_t n) const
			{
			    return (_a * g + _b) / n;
			}

	friend std::ostream&
			operator <<(std::ostream& out, const Coeff& coeff)
			{
			    out << '(' << coeff._a << ", " << coeff._b << ')';
			}

      private:
	value_type	_a, _b;
    };
    
  private:
    typedef BoxFilter	super;

  public:
    GuidedFilter(size_t w, value_type e) :super(w), _e(e)	{}

    value_type		epsilon()		const	{return _e;}
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
		    return inLen + 2 - 2*width();
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
    using namespace	boost;

    typedef Array<Coeff>					carray_type;
    typedef typename Coeff::template Init<typename Coeff::Params>
								coeff_init;
    typedef typename std::iterator_traits<GUIDE>::value_type	guide_type;
    typedef typename Coeff::Trans				coeff_trans;
    
  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(super::outLength(std::distance(ib, ie)));
    super::convolve(make_zip_iterator(make_tuple(ib, gb)),
		    make_zip_iterator(make_tuple(ie, ge)),
		    make_assignment_iterator(c.begin(),
					     coeff_init(width(), _e)));

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    std::advance(gb, width() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment2_iterator(
			make_zip_iterator(make_tuple(out, gb)),
			coeff_trans(width())));
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
    using namespace	boost;

    typedef Array<Coeff>					carray_type;
    typedef typename Coeff::template Init<typename Coeff::SimpleParams>
								coeff_init;
    typedef typename Coeff::Trans				coeff_trans;

  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(super::outLength(std::distance(ib, ie)));
    super::convolve(ib, ie,
		    make_assignment_iterator(c.begin(),
					     coeff_init(width(), _e)));

  // 係数ベクトルの平均値を求め，それによって入力データ列を線型変換する．
    std::advance(ib, width() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment2_iterator(
			make_zip_iterator(make_tuple(out, ib)),
			coeff_trans(width())));
}

/************************************************************************
*  class GuidedFilter2<T>						*
************************************************************************/
//! 2次元guided filterを表すクラス
template <class T>
class GuidedFilter2 : private BoxFilter2
{
  public:
    typedef T						value_type;
    typedef typename GuidedFilter<value_type>::Coeff	Coeff;
    
  private:
    typedef BoxFilter2					super;

  public:
    GuidedFilter2(size_t wrow, size_t wcol, value_type e)
	:super(wrow, wcol, 0), _e(e)				{}

    using	super::grainSize;
    using	super::setGrainSize;
    using	super::rowWidth;
    using	super::colWidth;
    using	super::setRowWidth;
    using	super::setColWidth;
    
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
    using namespace	boost;

    typedef Array2<Array<Coeff> >				carray2_type;
    typedef typename Coeff::template Init<typename Coeff::Params>
								coeff_init;
    typedef typename Coeff::Trans				coeff_trans;
    
    const size_t	n = rowWidth() * colWidth();
    carray2_type	c(super::outRowLength(std::distance(ib, ie)),
			  (ib != ie ?
			   super::outColLength(std::distance(ib->begin(),
							     ib->end())) :
			   0));
    super::convolve(make_row_transform_iterator(
			make_zip_iterator(make_tuple(ib, gb)), identity()),
		    make_row_transform_iterator(
			make_zip_iterator(make_tuple(ie, ge)), identity()),
		    make_row_iterator<assignment_iterator>(c.begin(),
							   coeff_init(n, _e)));
			   
    const_cast<GuidedFilter2*>(this)->setShift(colWidth() - 1);
    std::advance(gb,  rowWidth() - 1);
    std::advance(out, rowWidth() - 1);
    super::convolve(c.begin(), c.end(),
		    make_row_iterator<assignment2_iterator>(
			make_zip_iterator(make_tuple(out, gb)),
			coeff_trans(n)));
    const_cast<GuidedFilter2*>(this)->setShift(0);
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
    using namespace	boost;

    typedef Array2<Array<Coeff> >				carray2_type;
    typedef typename Coeff::template Init<typename Coeff::SimpleParams>
								coeff_init;
    typedef typename Coeff::Trans				coeff_trans;

    const size_t	n = rowWidth() * colWidth();
    carray2_type	c(super::outRowLength(std::distance(ib, ie)),
			  (ib != ie ?
			   super::outColLength(std::distance(ib->begin(),
							     ib->end())) :
			   0));
    super::convolve(ib, ie,
		    make_row_iterator<assignment_iterator>(c.begin(),
							   coeff_init(n, _e)));
    
    const_cast<GuidedFilter2*>(this)->setShift(colWidth() - 1);
    std::advance(ib,  rowWidth() - 1);
    std::advance(out, rowWidth() - 1);
    super::convolve(c.begin(), c.end(),
		    make_row_iterator<assignment2_iterator>(
			make_zip_iterator(make_tuple(out, ib)),
			coeff_trans(n)));
    const_cast<GuidedFilter2*>(this)->setShift(0);
}

}
