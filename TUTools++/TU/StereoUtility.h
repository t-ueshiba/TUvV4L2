/*!
  \file		StereoUtility.h
  \author	Toshio UESHIBA
  \brief	ステレオビジョンをサポートする各種クラスの定義と実装
*/
#ifndef TU_STEREOUTILITY_H
#define TU_STEREOUTILITY_H

#include <limits>
#include "TU/Image++.h"
#include "TU/StereoBase.h"

namespace TU
{
inline u_int
diff(const RGB& x, const RGB& y)
{
    return diff(x.r, y.r) + diff(x.g, y.g) + diff(x.b, y.b);
}

inline u_int
diff(const ARGB& x, const ARGB& y)
{
    return diff(x.r, y.r) + diff(x.g, y.g) + diff(x.b, y.b);
}

/************************************************************************
*  struct StereoParameters						*
************************************************************************/
struct StereoParameters
{
    StereoParameters()
	:doHorizontalBackMatch(true),
	 disparitySearchWidth(64), disparityMax(64),
	 disparityInconsistency(2), grainSize(100),
	 windowSize(11), intensityDiffMax(20), sigma(20)		{}

    size_t		disparityMin() const
			{
			    return disparityMax - disparitySearchWidth + 1;
			}
    std::istream&	get(std::istream& in)
			{
			    return in >> disparitySearchWidth
				      >> disparityMax
				      >> disparityInconsistency
				      >> grainSize
				      >> windowSize
				      >> intensityDiffMax;
			}
    std::ostream&	put(std::ostream& out) const
			{
			    using namespace	std;
			    
			    cerr << "  disparity search width:             ";
			    out << disparitySearchWidth << endl;
			    cerr << "  maximum disparity:                  ";
			    out << disparityMax << endl;
			    cerr << "  allowable disparity inconsistency:  ";
			    out << disparityInconsistency << endl;
			    cerr << "  grain size for parallel processing: ";
			    out << grainSize << endl;
			    cerr << "  window size:                        ";
			    out << windowSize << endl;
			    cerr << "  maximum intensity difference:       ";
			    out << intensityDiffMax << endl;
			    cerr << "  sigma for guided/tree filtering:    ";
			    return out << sigma << endl;
			}

    bool	doHorizontalBackMatch;
    size_t	disparitySearchWidth;	//!< 視差の探索幅
    size_t	disparityMax;		//!< 視差の最大値
    size_t	disparityInconsistency;	//!< 最適視差の不一致の許容値
    size_t	grainSize;		//!< 並列処理の粒度
    size_t	windowSize;		//!< ウィンドウのサイズ
    size_t	intensityDiffMax;	//!< 輝度差の最大値
    float	sigma;
};
    
/************************************************************************
*  class MinIdx								*
************************************************************************/
class MinIdx
{
  public:
    MinIdx(size_t disparityMax)	:_disparityMax(disparityMax)	{}

    template <class SCORES_>
    size_t	operator ()(const SCORES_& R) const
		{
		    auto	RminL = std::cbegin(R);
		    for (auto iter = std::cbegin(R); iter != std::cend(R);
			 ++iter)
			if (*iter < *RminL)
			    RminL = iter;
		    return _disparityMax - (RminL - std::cbegin(R));
		}
  private:
    const size_t	_disparityMax;
};

/************************************************************************
*  struct Diff2<T>							*
************************************************************************/
template <class T>
struct Diff2
{
    Diff2(T x, T thresh)	:_x(x), _thresh(thresh)		{}

#if defined(SIMD)
    auto	operator ()(simd::vec<T> y) const
		{
		    using signed_type
			= typename std::conditional_t<
				std::is_integral<T>::value,
				std::make_signed<T>,
				detail::identity<T> >::type;
		    using namespace	simd;

		    return cast<signed_type>(min(diff(simd::vec<T>(_x), y),
						 simd::vec<T>(_thresh)));
		}
#endif
    auto	operator ()(T y) const
		{
		    using signed_type
			= typename std::conditional_t<
				std::is_integral<T>::value,
				std::make_signed<T>,
				detail::identity<T> >::type;

		    return static_cast<signed_type>(std::min(diff(_x, y),
							     _thresh));
		}

    auto	operator ()(T y, T z) const
		{
		    return (*this)(y) + (*this)(z);
		}
    
  private:
    const T	_x;
    const T	_thresh;
};

/************************************************************************
*  class diff_iterator<T, COL, COL_RV...>				*
************************************************************************/
template <class T, class COL, class... COL_RV>
class diff_iterator : public boost::iterator_adaptor<
			diff_iterator<T, COL, COL_RV...>,
			zip_iterator<std::tuple<COL, COL_RV...> >,
			range<decltype(make_map_iterator(
					   std::declval<Diff2<T> >(),
					   std::declval<COL_RV>()...))>,
			boost::use_default,
			range<decltype(make_map_iterator(
					   std::declval<Diff2<T> >(),
					   std::declval<COL_RV>()...))> >
{
  private:
    using super	= boost::iterator_adaptor<
			diff_iterator,
			zip_iterator<std::tuple<COL, COL_RV...> >,
			range<decltype(make_map_iterator(
					   std::declval<Diff2<T> >(),
					   std::declval<COL_RV>()...))>,
			boost::use_default,
			range<decltype(make_map_iterator(
					   std::declval<Diff2<T> >(),
					   std::declval<COL_RV>()...))> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::reference;
    
  public:
		diff_iterator(size_t dsw, T thresh, COL colL, COL_RV... colRV)
		    :super(std::make_tuple(colL, colRV...)),
		     _dsw(dsw), _thresh(thresh)
		{
		}
    
  private:
    reference	dereference() const
		{
		    return dereference(
				std::make_index_sequence<sizeof...(COL_RV)>());
		}
    template <size_t... IDX_>
    reference	dereference(std::index_sequence<IDX_...>) const
		{
		    const auto&	iter_tuple = super::base().get_iterator_tuple();
		    
		    return {make_map_iterator(
			        Diff2<T>(*std::get<0>(iter_tuple), _thresh),
				std::get<1 + IDX_>(iter_tuple)...),
			    _dsw};
		}
    
  private:
    size_t	_dsw;
    T		_thresh;
};

template <class T, class COL, class... COL_RV>
inline diff_iterator<T, COL, COL_RV...>
make_diff_iterator(size_t dsw, T thresh, COL colL, COL_RV... colRV)
{
    return {dsw, thresh, colL, colRV...};
}

/************************************************************************
*  class matching_iterator<ITER, SCORE>					*
************************************************************************/
namespace detail
{
  template <class ITER>
  class matching_proxy
  {
    public:
      matching_proxy(const ITER& iter)	:_iter(iter)	{}

      template <class SCORES_>
      matching_proxy&	operator =(const SCORES_& R)
			{
			    _iter.load_scores(R);
			    return *this;
			}
	
    private:
      const ITER&	_iter;
  };
}
    
template <class OUT, class SCORE>
class matching_iterator
    : public boost::iterator_adaptor<matching_iterator<OUT, SCORE>,
				     OUT,
				     size_t,
				     std::random_access_iterator_tag,
				     detail::matching_proxy<
					 matching_iterator<OUT, SCORE> > >
{
  private:
    using super		= boost::iterator_adaptor<
				matching_iterator,
				OUT,
				size_t,
				std::random_access_iterator_tag,
				detail::matching_proxy<matching_iterator> >;
    
    struct Element
    {
	size_t		dminL = 0;
	SCORE		RminR = _infty;
	size_t		dminR = 0;
    };
    using buf_type	= Array<Element>;
    using biterator	= ring_iterator<typename buf_type::iterator>;
    using const_biterator
			= ring_iterator<typename buf_type::const_iterator>;

  public:
    using		typename super::reference;
    using		typename super::difference_type;

    friend class	boost::iterator_core_access;
    friend class	detail::matching_proxy<matching_iterator>;

  public:
		matching_iterator(OUT out, size_t dsw,
				  size_t disparityMax, size_t thresh,
				  bool doHorizontalBackMatch)
		    :super(out),
		     _dsw(dsw),
		     _disparityMax(disparityMax),
		     _thresh(thresh),
		     _doHorizontalBackMatch(doHorizontalBackMatch),
		     _buf(2*_dsw - 1),
		     _curr(_buf.cbegin(), _buf.cend()),
		     _next(_buf.begin(),  _buf.end())
		{
		}
		matching_iterator(const matching_iterator& iter)
		    :super(iter),
		     _dsw(iter._dsw),
		     _disparityMax(iter._disparityMax),
		     _thresh(iter._thresh),
		     _doHorizontalBackMatch(iter._doHorizontalBackMatch),
		     _buf(iter._buf),
		     _curr(_buf.cbegin(), _buf.cend()),
		     _next(_buf.begin(),  _buf.end())
		{
		    _curr += iter._curr.position();
		    _next += iter._next.position();
		}
    matching_iterator&
		operator =(const matching_iterator& iter)
		{
		    if (&iter != this)
		    {
			super::operator =(iter);
			_dsw	      = iter._dsw;
			_disparityMax = iter._disparityMax;
			_thresh	      = iter._thresh;
			_buf	      = iter._buf;
			_curr	      = {_buf.cbegin(), _buf.cend()};
			_curr	     += iter._curr.position();
			_next	      = {_buf.begin(), _buf.end()};
			_next	     += iter._next.position();
		    }
		    return *this;
		}
		matching_iterator(matching_iterator&&)	= default;
    matching_iterator&
		operator =(matching_iterator&&)		= default;
		~matching_iterator()
		{
		    while (_curr != _next)
		  	store_disparity();
		}

  private:
    reference	dereference() const
		{
		    return reference(*this);
		}
    void	increment()
		{
		    if (_next == _curr + _dsw)
			store_disparity();
		}

    template <class SCORES_>
    void	load_scores(const SCORES_& R) const
		{
		    auto	RminL = _infty;
		    auto	bynd  = _next;
		    size_t	d     = 0;
		    for (auto val : R)
		    {
			if (val < RminL)
			{
			    RminL	 = val;
			    _next->dminL = d;
			}
			
			if (val < bynd->RminR)
			{
			    bynd->RminR = val;
			    bynd->dminR = d;
			}

			++bynd;
			++d;
		    }
		    bynd->RminR = _infty;

		    ++_next;
		}
    void	store_disparity()
		{
		    const auto	dminL = _curr->dminL;
		    const auto	dminR = (_curr + dminL)->dminR;
		    if (_doHorizontalBackMatch)
			*super::base_reference()
			    = (diff(dminL, dminR) <= _thresh ?
			       _disparityMax - dminL : 0);
		    else
			*super::base_reference() = _disparityMax - dminL;

		    ++_curr;
		    ++super::base_reference();
		}
    
  private:
    size_t			_dsw;	// 代入可能にするためconstを付けない
    size_t			_disparityMax;	// 同上
    size_t			_thresh;	// 同上
    bool			_doHorizontalBackMatch;
    buf_type			_buf;
    const_biterator		_curr;
    mutable biterator		_next;
    constexpr static SCORE	_infty = std::numeric_limits<SCORE>::max();
};

template <class SCORE, class OUT> inline matching_iterator<OUT, SCORE>
make_matching_iterator(OUT out, size_t dsw, size_t disparityMax,
		       size_t thresh, bool doHorizontalBackMatch=true)
{
    return {out, dsw, disparityMax, thresh, doHorizontalBackMatch};
}
    
}
#endif	// !TU_STEREOUTILITY_H
