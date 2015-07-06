/*
 *  $Id$
 */
#ifndef __TU_STEREOUTILITY_H
#define __TU_STEREOUTILITY_H

#include <limits>
#include "TU/algorithm.h"	// for TU::diff()
#include "TU/Array++.h"

namespace TU
{
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
*  class MinIdx<A>							*
************************************************************************/
template <class A>
class MinIdx
{
  public:
    typedef A		argument_type;
    typedef size_t	result_type;

  public:
    MinIdx(size_t disparityMax)	:_disparityMax(disparityMax)	{}
    
    result_type	operator ()(const argument_type& R) const
		{
		    auto	RminL = R.begin();
		    for (auto iter = R.begin(); iter != R.end(); ++iter)
			if (*iter < *RminL)
			    RminL = iter;
		    return _disparityMax - (RminL - R.begin());
		}
  private:
    const size_t	_disparityMax;
};

/************************************************************************
*  class diff_iterator<COL, T>						*
************************************************************************/
template <class COL, class T>
class diff_iterator
    : public boost::iterator_adaptor<
	diff_iterator<COL, T>,
	fast_zip_iterator<
	    boost::tuples::cons<
		COL, boost::tuples::cons<COL, boost::tuples::null_type> > >,
	Array<T>,
	boost::use_default,
	const Array<T>&>
{
  private:
    typedef boost::iterator_adaptor<
	diff_iterator<COL, T>,
	fast_zip_iterator<
	    boost::tuples::cons<
		COL, boost::tuples::cons<COL, boost::tuples::null_type> > >,
	Array<T>,
	boost::use_default,
	const Array<T>&>				super;

  public:
    typedef typename super::value_type			value_type;
    typedef typename super::reference			reference;
    typedef typename super::base_type			base_type;
    
    friend class	boost::iterator_core_access;
    
  public:
		diff_iterator(const base_type& col, size_t dsw, T thresh)
		    :super(col), _P(dsw), _thresh(thresh)		{};
    
  private:
    reference	dereference() const
		{
		    const auto	iter_tuple = super::base().get_iterator_tuple();
		    const auto	colL = *boost::get<0>(iter_tuple);
		    auto	colR =  boost::get<1>(iter_tuple);
		    for (auto& val : _P)
		    {
			val = std::min(T(diff(colL, *colR)), _thresh);
			++colR;
		    }

		    return _P;
		}
    
  private:
    mutable value_type	_P;
    const T		_thresh;
};

/************************************************************************
*  class matching_iterator<ITER, A>					*
************************************************************************/
namespace detail
{
    template <class ITER, class A>
    class matching_proxy
    {
      public:
	typedef A				argument_type;
	typedef matching_proxy			self;
	
      public:
		matching_proxy(const ITER& iter)	:_iter(iter)	{}

	self&	operator =(const argument_type& R)
		{
		    _iter.read(R);
		    return *this;
		}
	
      private:
	const ITER&	_iter;
    };
}
    
template <class OUT, class A>
class matching_iterator
    : public boost::iterator_adaptor<matching_iterator<OUT, A>,
				     OUT,
				     size_t,
				     boost::single_pass_traversal_tag,
				     detail::matching_proxy<
					 matching_iterator<OUT, A>, A> >
{
  private:
    typedef boost::iterator_adaptor<
		matching_iterator,
		OUT,
		size_t,
		boost::single_pass_traversal_tag,
		detail::matching_proxy<matching_iterator, A> >	super;
    typedef typename A::value_type				score_type;
    
    struct Element
    {
	Element()	:dminL(0), RminR(_infty), dminR(0)	{}

	size_t		dminL;
	score_type	RminR;
	size_t		dminR;
    };
    typedef Array<Element>					buf_type;
    typedef ring_iterator<typename buf_type::iterator>		biterator;

  public:
    typedef A							argument_type;
    typedef typename super::reference				reference;

    friend class	boost::iterator_core_access;
    friend class	detail::matching_proxy<matching_iterator, A>;

  public:
		matching_iterator(const OUT& out, size_t dsw,
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
    void	read(const argument_type& R) const
		{
		    auto	RminL = R.begin();
		    auto	b = _next;
		    for (auto iter = R.begin(); iter != R.end(); ++iter)
		    {
			if (*iter < *RminL)
			    RminL = iter;

			if (*iter < b->RminR)
			{
			    b->RminR = *iter;
			    b->dminR = iter - R.begin();
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
    size_t		_dsw;		// 代入可能にするためconstを付けない
    size_t		_disparityMax;	// 同上
    size_t		_thresh;	// 同上
    buf_type		_buf;
    biterator		_curr;
    mutable biterator	_next;
    constexpr static score_type
			_infty = std::numeric_limits<score_type>::max();
};
    
}
#endif	// !__TU_STEREOUTILITY_H
