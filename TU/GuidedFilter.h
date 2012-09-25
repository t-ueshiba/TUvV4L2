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
#include "TU/Array++.h"
#include "TU/functional.h"
#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>
#include <boost/function_output_iterator.hpp>

namespace TU
{
/*!
  \namespace	detail
  \brief	実装の詳細を納める名前空間
*/ 
namespace detail
{
template <class T>
class GFCoeffs
{
  public:
    typedef T		value_type;

    class Params
    {
      public:
	struct Init
	{
	    typedef Params	result_type;
	
	    template <class IN, class GUIDE>
	    result_type	operator ()(IN p, GUIDE g) const
			{
			    return result_type(p, g);
			}
	};

      public:
	Params()						{}
    
	template <class IN, class GUIDE>
	Params(IN p, GUIDE g)
	    :_v0(p), _v1(g), _v2(_v0*_v1), _v3(_v1*_v1)		{}

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

	value_type	a(size_t n, value_type e) const
			{
			    return (e == 0 ? 1 :
				    (n*_v2 - _v0*_v1) /
				    (n*(_v3 + n*e) - _v1*_v1));
			}

	value_type	b(size_t n, value_type a) const
			{
			    return (_v0 - a*_v1) / n;
			}

      private:
	value_type	_v0, _v1, _v2, _v3;
    };
    
    class SimpleParams
    {
      public:
	struct Init
	{
	    typedef SimpleParams	result_type;
	
	    template <class IN>
	    result_type	operator ()(IN p) const
			{
			    return result_type(p);
			}
	};

      public:
	SimpleParams()						{}
    
	template <class IN>
	SimpleParams(IN p) :_v0(p), _v1(_v0*_v0)		{}

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

	value_type	a(size_t n, value_type e) const
			{
			    value_type	var = n*_v1 - _v0*_v0;
			    return (e == 0 ? 1 : var/(var + n*n*e));
			}

	value_type	b(size_t n, value_type a) const
			{
			    return (_v0 - a*_v0) / n;
			}

      private:
	value_type	_v0, _v1;
    };

    template <class CI>
    class Init
    {
      public:
	Init(CI ci, size_t n, value_type e) :_ci(ci), _n(n), _e(e) {}

	template <class P>
	void		operator ()(const P& params)
			{
			    _ci->init(params, _n, _e);
			    ++_ci;
			}

      private:
	CI			_ci;
	const size_t		_n;
	const value_type	_e;
    };

    template <class CI2>
    class InitIterator2
    {
      private:
	typedef typename std::iterator_traits<CI2>::value_type::iterator
								citerator;
	typedef boost::function_output_iterator<Init<citerator> >
								fiterator;
	
      public:
	InitIterator2(CI2 ci2, size_t n, value_type e)
	    :_ci2(ci2), _n(n), _e(e)	{}
	
	fiterator	begin()		{ return fiterator(
					    Init<citerator>(_ci2->begin(),
								_n, _e)); }
	InitIterator2*	operator ->()	{ return this; }
	void		operator ++()	{ ++_ci2; }
	InitIterator2	operator +(size_t d) const
			{
			    CI2	ci2 = _ci2;
			    std::advance(ci2, d);
			    return InitIterator2(ci2, _n, _e);
			}

      private:
	CI2			_ci2;
	const size_t		_n;
	const value_type	_e;
    };

    template <class GUIDE, class OUT>
    class Trans
    {
      public:
	typedef boost::zip_iterator<boost::tuple<GUIDE, OUT> >	ziterator;
	
      public:
	Trans(ziterator zip, size_t n) :_zip(zip), _n(n)	{}

	void		operator ()(const GFCoeffs& coeffs)
			{
			    boost::get<1>(*_zip)
				= coeffs.trans(boost::get<0>(*_zip), _n);
			    ++_zip;
			}
	
      private:
	ziterator	_zip;
	const size_t	_n;
    };

    template <class GUIDE2, class OUT2>
    class TransIterator2
    {
      private:
	typedef boost::zip_iterator<boost::tuple<GUIDE2,OUT2> >	ziterator2;
	typedef typename std::iterator_traits<GUIDE2>::value_type::iterator
								guide_iterator;
	typedef typename std::iterator_traits<OUT2>::value_type::iterator
								out_iterator;
	typedef Trans<guide_iterator, out_iterator>		trans_type;
	typedef boost::function_output_iterator<trans_type>	fiterator;
	
      public:
	TransIterator2(ziterator2 zip2, size_t n, size_t shift)
	    :_zip2(zip2), _n(n), _shift(shift)				{}
	
	fiterator	begin()
			{
			    using namespace	boost;

			    guide_iterator	gi = get<0>(*_zip2).begin();
			    std::advance(gi, _shift);
			    out_iterator	oi = get<1>(*_zip2).begin();
			    std::advance(oi, _shift);
			    
			    return fiterator(trans_type(
						 make_zip_iterator(
						     make_tuple(gi, oi)),
						 _n));
			}
	TransIterator2*	operator ->()	{ return this; }
	void		operator ++()	{ ++_zip2; }
	TransIterator2	operator +(size_t d) const
			{
			    ziterator2	zip2 = _zip2;
			    std::advance(zip2, d);
			    return TransIterator2(zip2, _n, _shift);
			}
	
      private:
	ziterator2	_zip2;
	const size_t	_n;
	const size_t	_shift;
    };

  public:
    GFCoeffs()							{}

    template <class P>
    void	init(const P& params, size_t n, value_type e)
		{
		    _a = params.a(n, e);
		    _b = params.b(n, _a);
		}
    
    GFCoeffs&	operator +=(const GFCoeffs& c)
		{
		    _a += c._a;
		    _b += c._b;
		    return *this;
		}
    
    GFCoeffs&	operator -=(const GFCoeffs& c)
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

  private:
    value_type	_a, _b;
};

}

//! 1次元入力データ列と1次元ガイドデータ列にguided filterを適用する
/*!
  \param ib	1次元入力データ列の先頭を示す反復子
  \param ie	1次元入力データ列の末尾の次を示す反復子
  \param gb	1次元ガイドデータ列の先頭を示す反復子
  \param ge	1次元ガイドデータ列の末尾の次を示す反復子
  \param out	guided filterを適用したデータの出力先を示す反復子
  \param w	box filterのウィンドウ幅
  \param e	正則化のための微小定数
*/
template <class IN, class GUIDE, class OUT> void
guidedFilter(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out,
	     size_t w, typename std::iterator_traits<OUT>::value_type e)
{
    using namespace	boost;

    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    typedef typename detail::GFCoeffs<value_type>		coeff_type;
    typedef Array<coeff_type>					carray_type;
    typedef unarize<typename coeff_type::SimpleParams::Init>	param_init;
    typedef typename coeff_type::template Init<
			typename carray_type::iterator>		coeff_init;
    typedef typename coeff_type::template Trans<GUIDE, OUT>	coeff_trans;
    
  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(std::distance(ib, ie));
    boxFilter(make_transform_iterator<param_init>(
		  make_zip_iterator(make_tuple(ib, gb))),
	      make_transform_iterator<param_init>(
		  make_zip_iterator(make_tuple(ie, ge))),
	      make_function_output_iterator(coeff_init(c.begin(), w, e)),
	      w);

  // 係数ベクトルの平均値を求め，それによってガイドデータ列を線型変換する．
    std::advance(gb,  w - 1);
    std::advance(out, w - 1);
    boxFilter(c.begin(), c.end() + 1 - w,
	      make_function_output_iterator(
		  coeff_trans(make_zip_iterator(make_tuple(gb, out)), w)),
	      w);
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
template <class IN, class OUT> void
guidedFilter(IN ib, IN ie, OUT out,
	     size_t w, typename std::iterator_traits<OUT>::value_type e)
{
    using namespace	boost;

    typedef typename std::iterator_traits<OUT>::value_type	value_type;
    typedef typename detail::GFCoeffs<value_type>		coeff_type;
    typedef Array<coeff_type>					carray_type;
    typedef typename coeff_type::SimpleParams::Init		param_init;
    typedef typename coeff_type::template Init<
			typename carray_type::iterator>		coeff_init;
    typedef typename coeff_type::template Trans<IN, OUT>	coeff_trans;

  // guided filterの2次元係数ベクトルを計算する．
    carray_type	c(std::distance(ib, ie));
    boxFilter(make_transform_iterator<param_init>(ib),
	      make_transform_iterator<param_init>(ie),
	      make_function_output_iterator(coeff_init(c.begin(), w, e)),
	      w);

  // 係数ベクトルの平均値を求め，それによって入力データ列を線型変換する．
    std::advance(ib,  w - 1);
    std::advance(out, w - 1);
    boxFilter(c.begin(), c.end() + 1 - w,
	      make_function_output_iterator(
		  coeff_trans(make_zip_iterator(make_tuple(ib, out)), w)),
	      w);
}

//! 2次元入力データと2次元ガイドデータにguided filterを適用する
/*!
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param gb	2次元ガイドデータの先頭の行を示す反復子
  \param ge	2次元ガイドデータの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
  \param w	box filterのウィンドウ幅
  \param e	正則化のための微小定数
*/
template <class IN, class GUIDE, class OUT> void
guidedFilter2(IN ib, IN ie, GUIDE gb, GUIDE ge, OUT out,
	      size_t wrow, size_t wcol,
	      typename std::iterator_traits<
	          typename std::iterator_traits<OUT>::value_type::iterator
	      >::value_type e)
{
    using namespace	boost;

    typedef typename std::iterator_traits<
		typename std::iterator_traits<OUT>
			    ::value_type::iterator>::value_type	value_type;
    typedef typename detail::GFCoeffs<value_type>		coeff_type;
    typedef Array2<Array<coeff_type> >				carray2_type;
    typedef typename coeff_type::Params				param_type;
    typedef unarize<seq_transform<Array<param_type>,
				  typename param_type::Init> >	param_init;
    typedef typename coeff_type::template InitIterator2<
		typename carray2_type::iterator>		coeff_init;
    typedef
      typename coeff_type::template TransIterator2<GUIDE, OUT>	coeff_trans;

    const size_t	n = wrow * wcol;
    carray2_type	c(std::distance(ib, ie),
			  (ib != ie ?
			   std::distance(ib->begin(), ib->end()) + 1 - wcol :
			   0));
    boxFilter2(make_transform_iterator<param_init>(
		   make_zip_iterator(make_tuple(ib, gb))),
	       make_transform_iterator<param_init>(
		   make_zip_iterator(make_tuple(ie, ge))),
	       coeff_init(c.begin(), n, e), wrow, wcol);

    std::advance(gb,  wrow - 1);
    std::advance(out, wrow - 1);
    boxFilter2(c.begin(), c.end() + 1 - wrow,
	       coeff_trans(make_zip_iterator(make_tuple(gb, out)),
			   n, wcol - 1),
	       wrow, wcol);
}

//! 2次元入力データにguided filterを適用する
/*!
  ガイドデータは与えられた2次元入力データに同一とする．
  \param ib	2次元入力データの先頭の行を示す反復子
  \param ie	2次元入力データの末尾の次の行を示す反復子
  \param out	guided filterを適用したデータの出力先の先頭行を示す反復子
  \param w	box filterのウィンドウ幅
  \param e	正則化のための微小定数
*/
template <class IN, class OUT> void
guidedFilter2(IN ib, IN ie, OUT out, size_t wrow, size_t wcol,
	      typename std::iterator_traits<
	          typename std::iterator_traits<OUT>::value_type::iterator
	      >::value_type e)
{
    using namespace	boost;

    typedef typename std::iterator_traits<
		typename std::iterator_traits<OUT>
			    ::value_type::iterator>::value_type	value_type;
    typedef typename detail::GFCoeffs<value_type>		coeff_type;
    typedef Array2<Array<coeff_type> >				carray2_type;
    typedef typename coeff_type::SimpleParams			param_type;
    typedef seq_transform<Array<param_type>,
			  typename param_type::Init>		param_init;
    typedef typename coeff_type::template InitIterator2<
		typename carray2_type::iterator>		coeff_init;
    typedef
	typename coeff_type::template TransIterator2<IN, OUT>	coeff_trans;

    const size_t	n = wrow * wcol;
    carray2_type	c(std::distance(ib, ie),
			  (ib != ie ?
			   std::distance(ib->begin(), ib->end()) + 1 - wcol
			   : 0));
    boxFilter2(make_transform_iterator<param_init>(ib),
	       make_transform_iterator<param_init>(ie),
	       coeff_init(c.begin(), n, e), wrow, wcol);

    std::advance(ib,  wrow - 1);
    std::advance(out, wrow - 1);
    boxFilter2(c.begin(), c.end() + 1 - wrow,
	       coeff_trans(make_zip_iterator(make_tuple(ib, out)),
			   n, wcol - 1),
	       wrow, wcol);
}

}
