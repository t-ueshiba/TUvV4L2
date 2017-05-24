/*!
  \file		StereoUtility.h
  \author	Toshio UESHIBA
  \brief	ステレオビジョンをサポートする各種クラスの定義と実装
*/
#ifndef TU_STEREOUTILITY_H
#define TU_STEREOUTILITY_H

#include <limits>
#include "TU/algorithm.h"	// for TU::diff()
#include "TU/Image++.h"

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
*  class diff_iterator<COL, T>						*
************************************************************************/
template <class COL, class T>
class diff_iterator
    : public boost::iterator_adaptor<diff_iterator<COL, T>,
				     zip_iterator<std::tuple<COL, COL> >,
				     Array<T>,
				     boost::use_default,
				     const Array<T>&>
{
  private:
    using super	= boost::iterator_adaptor<diff_iterator,
					  zip_iterator<std::tuple<COL, COL> >,
					  Array<T>,
					  boost::use_default,
					  const Array<T>&>;
    friend	class boost::iterator_core_access;

  public:
    using col_t	= iterator_value<COL>;
    using	typename super::base_type;
    using	typename super::reference;
    
  public:
		diff_iterator(base_type col, size_t dsw, col_t thresh)
		    :super(col), _P(dsw), _thresh(thresh)		{};
    
  private:
    reference	dereference() const
		{
		    const auto	iter_tuple = super::base().get_iterator_tuple();
		    const auto	colL = *std::get<0>(iter_tuple);
		    auto	colR =  std::get<1>(iter_tuple);
		    for (auto& val : _P)
		    {
			val = std::min(diff(colL, *colR), _thresh);
			++colR;
		    }

		    return _P;
		}
    
  private:
    mutable Array<T>	_P;
    const col_t		_thresh;
};

template <class T, class COL> inline diff_iterator<COL, T>
make_diff_iterator(COL colL, COL colR, size_t dsw, iterator_value<COL> thresh)
{
    return {make_zip_iterator(std::make_tuple(colL, colR)), dsw, thresh};
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
			    _iter.read(R);
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
				     boost::single_pass_traversal_tag,
				     detail::matching_proxy<
					 matching_iterator<OUT, SCORE> > >
{
  private:
    using super		= boost::iterator_adaptor<
				matching_iterator,
				OUT,
				size_t,
				boost::single_pass_traversal_tag,
				detail::matching_proxy<matching_iterator> >;
    
    struct Element
    {
	Element()	:dminL(0), RminR(_infty), dminR(0)	{}

	size_t		dminL;
	SCORE		RminR;
	size_t		dminR;
    };
    using buf_type	= Array<Element>;
    using biterator	= ring_iterator<typename buf_type::iterator>;

  public:
    using		typename super::reference;

    friend class	boost::iterator_core_access;
    friend class	detail::matching_proxy<matching_iterator>;

  public:
		matching_iterator(OUT out, size_t dsw,
				  size_t disparityMax, size_t thresh)
		    :super(out),
		     _dsw(dsw),
		     _disparityMax(disparityMax),
		     _thresh(thresh),
		     _buf(2*_dsw - 1),
		     _curr(_buf.begin(), _buf.end()),
		     _next(_curr)
		{
		}
		matching_iterator(const matching_iterator& iter)
		    :super(iter),
		     _dsw(iter._dsw),
		     _disparityMax(iter._disparityMax),
		     _thresh(iter._thresh),
		     _buf(iter._buf),
		     _curr(_buf.begin(), _buf.end()),
		     _next(_curr)
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
			_curr	      = biterator(_buf.begin(), _buf.end());
			_curr	     += iter._curr.position();
			_next	      = _curr;
			_next	     += iter._next.position();
		    }
		    return *this;
		}
		~matching_iterator()
		{
		    while (_curr != _next)
			assign();
		}
    
  private:
    reference	dereference() const
		{
		    return reference(*this);
		}
    void	increment()
		{
		}
    template <class SCORES_>
    void	read(const SCORES_& R) const
		{
		    auto	RminL = std::cbegin(R);
		    auto	b = _next;
		    for (auto iter = std::cbegin(R); iter != std::cend(R);
			 ++iter)
		    {
			if (*iter < *RminL)
			    RminL = iter;

			if (*iter < b->RminR)
			{
			    b->RminR = *iter;
			    b->dminR = iter - std::cbegin(R);
			}
			++b;
		    }
		    _next->dminL = RminL - R.begin();
		    b->RminR = _infty;
		    ++_next;
		    
		    if (_next - _curr == _dsw)
			const_cast<matching_iterator*>(this)->assign();
		}
    void	assign()
		{
		    auto	dminL = _curr->dminL;
		    auto	dminR = (_curr + dminL)->dminR;
		    *super::base_reference() = (diff(dminL, dminR) <= _thresh ?
						_disparityMax - dminL : 0);
		  //*super::base_reference() = _disparityMax - dminL;
		    ++super::base_reference();
		    ++_curr;
		}
    
  private:
    size_t			_dsw;	// 代入可能にするためconstを付けない
    size_t			_disparityMax;	// 同上
    size_t			_thresh;	// 同上
    buf_type			_buf;
    biterator			_curr;
    mutable biterator		_next;
    constexpr static SCORE	_infty = std::numeric_limits<SCORE>::max();
};

template <class SCORE, class OUT> inline matching_iterator<OUT, SCORE>
make_matching_iterator(OUT out, size_t dsw, size_t disparityMax, size_t thresh)
{
    return {out, dsw, disparityMax, thresh};
}
    
}
#endif	// !TU_STEREOUTILITY_H
