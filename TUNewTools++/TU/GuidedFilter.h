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
    using value_type	= T;
    using guide_type	= element_t<value_type>;
    using Params4	= std::tuple<value_type, guide_type,
				     value_type, guide_type>;
    using Params2	= std::tuple<value_type, value_type>;
    using Coeffs	= std::tuple<value_type, value_type>;
    
    struct init_params
    {
	template <class IN_, class GUIDE_>
	auto	operator ()(std::tuple<IN_, GUIDE_> t) const
		{
		    using	std::get;
		    
		    return Params4(get<0>(t),		get<1>(t),
				   get<0>(t)*get<1>(t), get<1>(t)*get<1>(t));
		}
	template <class IN_>
	auto	operator ()(IN_ p) const
		{
		    return Params2(p, p*p);
		}
    };

    class init_coeffs
    {
      public:
	init_coeffs(size_t n, guide_type e)	:_n(n), _sq_e(e*e)	{}
	
	auto	operator ()(const Params4& params) const
		{
		    using 	std::get;

		    const auto	a = (_n*get<2>(params)
				     -  get<0>(params)*get<1>(params))
				  / (_n*(get<3>(params) + _n*_sq_e)
				     -   get<1>(params)*get<1>(params));
		    return Coeffs(a, (get<0>(params) - a*get<1>(params))/_n);
		}
	auto	operator ()(const Params2& params) const
		{
		    using	std::get;
		    
		    const auto	var = _n*get<1>(params)
				       - get<0>(params)*get<0>(params);
		    const auto	a   = var/(var + _n*_n*_sq_e);
		    return Coeffs(a, (get<0>(params) - a*get<0>(params))/_n);
		}
	
      private:
	const size_t		_n;
	const guide_type	_sq_e;
    };

    class trans_guides
    {
      public:
	trans_guides(size_t n)	:_n(n)					{}

	template <class GUIDE_, class OUT_>
	void	operator ()(std::tuple<GUIDE_, OUT_> t,
			    const Coeffs& coeffs) const
		{
		    using	std::get;
		    
		    get<1>(t) = (get<0>(coeffs)*get<0>(t) + get<1>(coeffs))/_n;
		}
	template <class IN_, class GUIDE_, class OUT_>
	void	operator ()(std::tuple<std::tuple<IN_, GUIDE_>, OUT_> t,
			    const Coeffs& coeffs) const
		{
		    using	std::get;
		    
		    get<1>(t) = (get<0>(coeffs)*get<1>(get<0>(t)) +
				 get<1>(coeffs))/_n;
		}
	
      private:
	const size_t	_n;
    };
    
  private:
    using super	= BoxFilter;

  public:
    GuidedFilter(size_t w, guide_type e) :super(w), _e(e)		{}

    guide_type	epsilon()			const	{return _e;}
    GuidedFilter&
		setEpsilon(guide_type e)		{ _e = e; return *this; }
    
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
    guide_type	_e;
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
  // guided filterの2次元係数ベクトルを計算する．
    Array<Coeffs>	c(super::outLength(std::distance(ib, ie)));
    super::convolve(boost::make_transform_iterator(
			make_zip_iterator(std::make_tuple(ib, gb)),
			init_params()),
		    boost::make_transform_iterator(
			make_zip_iterator(std::make_tuple(ie, ge)),
			init_params()),
		    make_assignment_iterator(c.begin(),
					     init_coeffs(winSize(), _e)));
    
  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    std::advance(gb, winSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment_iterator(
			make_zip_iterator(std::make_tuple(gb, out)),
			trans_guides(winSize())));
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
  // guided filterの2次元係数ベクトルを計算する．
    Array<Coeffs>	c(super::outLength(std::distance(ib, ie)));
    super::convolve(boost::make_transform_iterator(ib, init_params()),
		    boost::make_transform_iterator(ie, init_params()),
		    make_assignment_iterator(c.begin(),
					     init_coeffs(winSize(), _e)));

  // 係数ベクトルの平均値を求め，それによって入力データ列を線型変換する．
    std::advance(ib, winSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_assignment_iterator(
			make_zip_iterator(std::make_tuple(ib, out)),
			trans_guides(winSize())));
}

/************************************************************************
*  class GuidedFilter2<T>						*
************************************************************************/
//! 2次元guided filterを表すクラス
template <class T>
class GuidedFilter2 : private BoxFilter2
{
  public:
    using value_type	= T;
    using guide_type	= element_t<value_type>;
    using Coeffs	= typename GuidedFilter<value_type>::Coeffs;
    using init_params	= typename GuidedFilter<value_type>::init_params;
    using init_coeffs	= typename GuidedFilter<value_type>::init_coeffs;
    using trans_guides	= typename GuidedFilter<value_type>::trans_guides;
    
  private:
    using super		= BoxFilter2;

  public:
    GuidedFilter2(size_t wrow, size_t wcol, guide_type e)
	:super(wrow, wcol), _e(e)				{}

    using	super::rowWinSize;
    using	super::colWinSize;
    using	super::grainSize;
    using	super::setRowWinSize;
    using	super::setColWinSize;
    using	super::setGrainSize;
    
    guide_type	epsilon()		const	{return _e;}
    GuidedFilter2&
		setEpsilon(guide_type e)	{ _e = e; return *this; }
    
    template <class IN, class GUIDE, class OUT>
    void	convolve(IN ib, IN ie,
			 GUIDE gb, GUIDE ge, OUT out)	const	;
    template <class IN, class OUT>
    void	convolve(IN ib, IN ie, OUT out)		const	;
    
  private:
    guide_type	_e;
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
    if (ib == ie)
	return;
    
    const auto		n = rowWinSize() * colWinSize();
    Array2<Coeffs>	c(super::outRowLength(std::distance(ib, ie)),
			  super::outColLength(std::distance(std::begin(*ib),
							    std::end(*ib))));

  // guided filterの2次元係数ベクトルを計算する．
    super::convolve(make_range_iterator(
			boost::make_transform_iterator(
			    std::begin(std::make_tuple(*ib, *gb)),
			    init_params()),
			ib.stride(), ib.size()),
		    make_range_iterator(
			boost::make_transform_iterator(
			    std::begin(std::make_tuple(*ie, *ge)),
			    init_params()),
			ie.stride(), ie.size()),
		    make_range_iterator(
			make_assignment_iterator(c.begin()->begin(),
						 init_coeffs(n, _e)),
			c.begin().stride(), c.begin().size()));

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    std::advance(gb,  rowWinSize() - 1);
    std::advance(out, rowWinSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_range_iterator(
			make_assignment_iterator(
			    std::begin(std::make_tuple(*gb, *out))
			    + colWinSize() - 1,
			    trans_guides(n)),
			out.stride(), out.size()));
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
    if (ib == ie)
	return;
    
    const auto		n = rowWinSize() * colWinSize();
    Array2<Coeffs>	c(super::outRowLength(std::distance(ib, ie)),
			  super::outColLength(std::distance(std::begin(*ib),
							    std::end(*ib))));

  // guided filterの2次元係数ベクトルを計算する．
    super::convolve(make_range_iterator(
			boost::make_transform_iterator(std::begin(*ib),
						       init_params()),
			ib.stride(), ib.size()),
		    make_range_iterator(
			boost::make_transform_iterator(std::begin(*ie),
						       init_params()),
			ie.stride(), ie.size()),
		    make_range_iterator(
			make_assignment_iterator(c.begin()->begin(),
						 init_coeffs(n, _e)),
			c.begin().stride(), c.begin().size()));

  // 係数ベクトルの平均値を求め，それによって入力データ列を線型変換する．
    std::advance(ib,  rowWinSize() - 1);
    std::advance(out, rowWinSize() - 1);
    super::convolve(c.begin(), c.end(),
		    make_range_iterator(
			make_assignment_iterator(
			    std::begin(std::make_tuple(*ib, *out))
			    + colWinSize() - 1,
			    trans_guides(n)),
			out.stride(), out.size()));
}

}
#endif	// !__TU_GUIDEDFILTER_H
