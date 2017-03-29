/*!
  \file		StereoBase.h
  \brief	ステレオマッチングクラスの定義と実装
*/
#ifndef __TU_STEREOBASE_H
#define __TU_STEREOBASE_H

#include "TU/Image++.h"		// Use RGBA
#include "TU/algorithm.h"	// Use std::min(), std::max() and TU::diff().
#include "TU/tuple.h"
#include "TU/simd/simd.h"
#include <limits>		// Use std::numeric_limits<T>.
#include <stack>
#include <tbb/blocked_range.h>

#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/spin_mutex.h>
#  include <tbb/scalable_allocator.h>
#endif
#include "TU/Profiler.h"

#if defined(PROFILE) && !defined(USE_TBB)
#  define ENABLE_PROFILER
#else
#  define ENABLE_PROFILER	void
#endif

namespace TU
{
/************************************************************************
*  struct Diff<T>							*
************************************************************************/
template <class T>
struct Diff
{
    using result_type = std::conditional_t<std::is_integral<T>::value, int, T>;

    Diff(T x, T thresh)	:_x(x), _thresh(thresh)		{}
    
    result_type	operator ()(T y) const
		{
		    return std::min(diff(_x, y), _thresh);
		}
    result_type	operator ()(std::tuple<T, T> y) const
		{
		    return (*this)(std::get<0>(y)) + (*this)(std::get<1>(y));
		}
    
  private:
    const T	_x;
    const T	_thresh;
};
    
#if defined(SIMD)
template <class T>
struct Diff<simd::vec<T> >
{
    using signed_type	= std::make_signed_t<T>;
    using result_type	= simd::vec<signed_type>;

    Diff(simd::vec<T> x, simd::vec<T> thresh)	:_x(x), _thresh(thresh)	{}
    
    result_type	operator ()(simd::vec<T> y) const
		{
		    using namespace	simd;

		    return cast<signed_type>(min(diff(_x, y), _thresh));
		}
    result_type	operator ()(std::tuple<simd::vec<T>, simd::vec<T> > y) const
		{
		    return (*this)(std::get<0>(y)) + (*this)(std::get<1>(y));
		}

  private:
    const simd::vec<T>	_x;
    const simd::vec<T>	_thresh;
};
#endif

template <class T, class S> inline Diff<T>
makeDiff(T x, S thresh)
{
    return Diff<T>(x, thresh);
}
    
/************************************************************************
*  struct Minus<T>							*
************************************************************************/
struct Minus
{
    template <class T>
    using result_t = std::conditional_t<std::is_integral<T>::value, int, T>;

    template <class T>
    result_t<T>	operator ()(T x, T y) const
		{
		    return result_t<T>(x) - result_t<T>(y);
		}

    template <class... T>
    auto	operator ()(std::tuple<T...> x, std::tuple<T...> y) const
		{
		    return tuple_transform(x, y, Minus());
		}

#if defined(SIMD)
    template <class T>
    auto	operator ()(simd::vec<T> x, simd::vec<T> y) const
		{
		    using signed_type	= std::make_signed_t<T>;
		    
		    return simd::cast<signed_type>(x) -
			   simd::cast<signed_type>(y);
		}
#endif
};
    
/************************************************************************
*  struct Blend<T>							*
************************************************************************/
template <class T>
struct Blend
{
    typedef std::tuple<T, T>	argument_type;
    typedef T			result_type;
    
    Blend(T alpha)	:_alpha(alpha)					{}

    result_type	operator ()(T x, T y) const
		{
		    return x + _alpha*(y - x);
		}
    result_type	operator ()(const argument_type& args) const
		{
		    return (*this)(std::get<0>(args), std::get<1>(args));
		}

  private:
    const T	_alpha;
};

/************************************************************************
*  col2ptr(COL)								*
************************************************************************/
template <class COL> inline COL
col2ptr(COL col)
{
    return col;
}
    
template <class ITER_TUPLE> inline auto
col2ptr(const zip_iterator<ITER_TUPLE>& col)
    -> decltype(make_zip_iterator(
		    std::make_tuple(
			std::get<0>(col.get_iterator_tuple()),
			std::get<1>(col.get_iterator_tuple()).operator ->())))
{
    return make_zip_iterator(
	       std::make_tuple(
		   std::get<0>(col.get_iterator_tuple()),
		   std::get<1>(col.get_iterator_tuple()).operator ->()));
}
    
/************************************************************************
*  class dummy_iterator<ITER>						*
************************************************************************/
template <class ITER>
class dummy_iterator
    : public boost::iterator_adaptor<dummy_iterator<ITER>, ITER>
{
  private:
    using super	= boost::iterator_adaptor<dummy_iterator, ITER>;

  public:
    using	typename super::difference_type;

    friend	class boost::iterator_core_access;

  public:
    dummy_iterator(ITER iter)	:super(iter)	{}

  private:
    void	advance(difference_type)	{}
    void	increment()			{}
    void	decrement()			{}
};
    
template <class ITER> dummy_iterator<ITER>
make_dummy_iterator(ITER iter)		{ return dummy_iterator<ITER>(iter); }

/************************************************************************
*  struct Idx<T>							*
************************************************************************/
template <class T>
struct Idx
{
		Idx()	:_i(0)		{}
		operator T()	const	{ return _i; }
    void	operator ++()		{ ++_i; }
    
  private:
    T		_i;
};

#if defined(SIMD)
template <class T>
struct Idx<simd::vec<T> > : simd::vec<T>
{
    typedef simd::vec<T>	super;
    
		Idx()	:super(std::make_index_sequence<super::size>())	{}
    void	operator ++()		{ *this += super(super::size); }
};
#endif

/************************************************************************
*  class mask_iterator<ITER, RV_ITER>					*
************************************************************************/
template <class ITER, class RV_ITER>
class mask_iterator : public boost::iterator_adaptor<
				 mask_iterator<ITER, RV_ITER>,
				 ITER,
				 tuple_replace<iterator_value<RV_ITER>, bool>,
				 boost::single_pass_traversal_tag,
				 tuple_replace<iterator_value<RV_ITER>, bool> >
{
  private:
    using rv_type	= tuple_decay_t<iterator_value<RV_ITER> >;
    using super		= boost::iterator_adaptor<
			      mask_iterator,
			      ITER,
			      tuple_replace<rv_type, bool>,
			      boost::single_pass_traversal_tag,
			      tuple_replace<rv_type, bool> >;
    using element_type	= tuple_head<rv_type>;
    
  public:
    using	typename super::difference_type;
    using	typename super::reference;

    friend	class boost::iterator_core_access;

  public:
		mask_iterator(ITER R, RV_ITER RminRV)
		    :super(R), _Rs(R), _RminL(_Rs), _RminRV(RminRV), _nextRV()
		{
		    init(std::numeric_limits<element_type>::max(), _nextRV);
		}
    int		dL() const
		{
		    return _RminL - _Rs;
		}
      
  private:
    void	init(element_type val, element_type& x)
		{
		    x = val;
		}
    template <class VEC_>
    void	init(element_type val, VEC_& x)
		{
		    x = std::make_tuple(val, val);
		}
    
    void	update(element_type R, bool& mask)
		{
		    element_type	RminR  = *_RminRV;
		    *_RminRV = _nextRV;
		    ++_RminRV;
		    
		    if (R < RminR)
		    {
			_nextRV = R;
			mask = true;
		    }
		    else
		    {
			_nextRV = RminR;
			mask = false;
		    }
		}
    template <class VEC_>
    void	update(element_type R, VEC_& mask)
		{
		    using 	std::get;

		    element_type	RminR = get<0>(*_RminRV),
					RminV = get<1>(*_RminRV);
		    *_RminRV = _nextRV;
		    ++_RminRV;

		    if (R < RminR)
		    {
			get<0>(_nextRV) = R;
			get<0>(mask) = true;
		    }
		    else
		    {
			get<0>(_nextRV) = RminR;
			get<0>(mask) = false;
		    }
		    
		    if (R < RminV)
		    {
			get<1>(_nextRV) = R;
			get<1>(mask) = true;
		    }
		    else
		    {
			get<1>(_nextRV) = RminV;
			get<1>(mask) = false;
		    }
		}

    void	cvtdown(reference& mask)
		{
		    element_type	R = *super::base();
		    if (R < *_RminL)
			_RminL = super::base();
		    ++super::base_reference();
		    update(R, mask);
		}

    reference	dereference() const
		{
		    reference	mask;
		    const_cast<mask_iterator*>(this)->cvtdown(mask);
		    return mask;
		}
    void	advance(difference_type)				{}
    void	increment()						{}
    void	decrement()						{}

  private:
    const ITER	_Rs;
    ITER	_RminL;
    RV_ITER	_RminRV;
    rv_type	_nextRV;
};

#if defined(SIMD)
namespace simd
{
#  if defined(MMX)
#    if !defined(SSE2)
  template <size_t I> inline int
  extract(Is32vec x)
  {					// short用の命令を無理に int に適用
      return _mm_extract_pi16(x, I);	// しているため，x が SHRT_MIN 以上
  }					// かつ SHRT_MAX 以下の場合のみ有効
#    elif !defined(SSE4)
  template <size_t I> inline int
  extract(Is32vec x)
  {					// short用の命令を無理に int に適用
      return _mm_extract_epi16(x, I);	// しているため，x が SHRT_MIN 以上
  }					// かつ SHRT_MAX 以下の場合のみ有効
#    endif
#  endif

#  if defined(WITHOUT_CVTDOWN)
  template <class ITER, class RV_ITER>
  class mask_iterator
      : public boost::iterator_adaptor<
	    mask_iterator<ITER, RV_ITER>,
	    ITER,
	    tuple_replace<
		iterator_value<RV_ITER>,
		vec<mask_type<typename tuple_head<
				  iterator_value<RV_ITER> >::element_type> > >,
	    boost::single_pass_traversal_tag,
	    tuple_replace<
		iterator_value<RV_ITER>,
		vec<mask_type<typename tuple_head<
				  iterator_value<RV_ITER> >::element_type> > > >
#  else
  template <class T, class ITER, class RV_ITER>
  class mask_iterator
      : public boost::iterator_adaptor<
	    mask_iterator<T, ITER, RV_ITER>,
	    ITER,
	    tuple_replace<iterator_value<RV_ITER>, vec<T> >,
	    boost::single_pass_traversal_tag,
	    tuple_replace<iterator_value<RV_ITER>, vec<T> > >
#  endif
  {
    private:
      typedef tuple_decay_t<iterator_value<RV_ITER> >		score_vec;
      typedef typename tuple_head<score_vec>::element_type	score_element;
      typedef mask_type<score_element>				S;
      typedef tuple_replace<score_vec, vec<S> >			mask_vec;
#  if defined(WITHOUT_CVTDOWN)
      typedef S							T;
#endif
      typedef boost::iterator_adaptor<
		  mask_iterator,
		  ITER,
		  tuple_replace<score_vec, vec<T> >,
		  boost::single_pass_traversal_tag,
		  tuple_replace<score_vec, vec<T> > >		super;

      template <class T_, size_t I_=vec<T_>::size/2> static inline int
		minIdx(vec<T_> d, vec<T_> x,
		       std::integral_constant<size_t, I_>
		       dummy=std::integral_constant<size_t, I_>())
		{
		    const auto	y = shift_r<I_>(x);
		    return minIdx<T_>(select(x < y, d, shift_r<I_>(d)),
				      min(x, y),
				      std::integral_constant<size_t, I_/2>());
		}
      template <class T_> static inline int
		minIdx(vec<T_> d, vec<T_>, std::integral_constant<size_t, 0>)
		{
		    return extract<0>(d);
		}
      
    public:
      typedef typename super::difference_type		difference_type;
      typedef typename super::reference			reference;

      friend class	boost::iterator_core_access;

    public:
		mask_iterator(ITER R, RV_ITER RminRV)
		    :super(R),
		     _index(),
		     _dminL(_index),
		     _RminL(std::numeric_limits<score_element>::max()),
		     _RminRV(RminRV),
		     _nextRV(init(std::numeric_limits<score_element>::max(),
				  is_tuple<mask_vec>()))
		{
		}
      int	dL()	const	{ return minIdx(_dminL, _RminL); }
	
    private:
    // _nextRV の初期化
      static vec<score_element>
		init(score_element val, std::false_type)
		{
		    return vec<score_element>(val);
		}
      static std::tuple<vec<score_element>, vec<score_element> >
		init(score_element val, std::true_type)
		{
		    return std::make_tuple(vec<score_element>(val),
					   vec<score_element>(val));
		}

    // mask と mask tuple に対するupdate
      template <class T_>
      typename std::enable_if<(vec<T_>::size == vec<S>::size),
			      tuple_replace<mask_vec, vec<T_> > >::type
		cvtdown()
		{
		    const auto	R = *super::base();
		    ++super::base_reference();

		    _dminL = select(R < _RminL, _index, _dminL);
		    _RminL = min(R, _RminL);
		    ++_index;

		    constexpr size_t	N = vec<score_element>::size;
		    const score_vec	RminRV = *_RminRV;
		    const auto		minval = min(R, RminRV);
		    *_RminRV = shift_r<N-1>(_nextRV, minval);
		    ++_RminRV;
		    _nextRV  = minval;

		    return cvt<T_, false, true>(R < RminRV);
		}
#  if !defined(WITHOUT_CVTDOWN)
      template <class T_>
      typename std::enable_if<(vec<T_>::size > vec<S>::size),
			      tuple_replace<mask_vec, vec<T_> > >::type
		cvtdown()
		{
		    using	A = cvt_above_type<T_, S, true>;
		    
		    auto	x = cvtdown<A>();
		    auto	y = cvtdown<A>();
		    return cvt<T_, true>(x, y);
		}
#  endif
      reference	dereference() const
		{
		    return const_cast<mask_iterator*>(this)->cvtdown<T>();
		}
      void	advance(difference_type)				{}
      void	increment()						{}
      void	decrement()						{}

    private:
      Idx<vec<score_element> >	_index;
      vec<score_element>	_dminL;
      vec<score_element>	_RminL;
      RV_ITER			_RminRV;
      score_vec			_nextRV;
  };

#  if defined(WITHOUT_CVTDOWN)
  template <class ITER, class RV_ITER> mask_iterator<ITER, RV_ITER>
  make_mask_iterator(ITER R, RV_ITER RminRV)
  {
      return mask_iterator<ITER, RV_ITER>(R, RminRV);
  }
#  endif

  template <class T> inline simd::vec<T>
  fast_select(simd::vec<mask_type<T> > mask,
	      simd::vec<T> index, simd::vec<T> dminRV)
  {
      return select(mask, index, dminRV);
  }
  template <class MASK, class T, class DMIN_RV> inline DMIN_RV
  fast_select(const MASK& mask, simd::vec<T> index, const DMIN_RV& dminRV)
  {
      using namespace 	boost;

      return make_tuple(select(get<0>(mask), index, get<0>(dminRV)),
			select(get<1>(mask), index, get<1>(dminRV)));
  }
}	// end of namespace simd
#else
template <class S, class T, class U> inline auto
fast_select(const S& s, const T& x, const U& y) -> decltype(select(s, x, y))
{
    return select(s, x, y);
}
#endif
    
/************************************************************************
*  class StereoBase<STEREO>						*
************************************************************************/
template <class STEREO>
class StereoBase : public Profiler<ENABLE_PROFILER>
{
  public:
  //! ステレオ対応探索の各種パラメータを収めるクラス．
    struct Parameters
    {
	Parameters()
	    :doHorizontalBackMatch(true), doVerticalBackMatch(true),
	     disparitySearchWidth(64), disparityMax(64),
	     disparityInconsistency(2), grainSize(100)			{}

      //! 視差の最小値を返す．
	size_t		disparityMin() const
			{
			    return disparityMax - disparitySearchWidth + 1;
			}
	std::istream&	get(std::istream& in)
			{
			    return in >> disparitySearchWidth
				      >> disparityMax
				      >> disparityInconsistency
				      >> grainSize;
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

			    return out;
			}

	bool	doHorizontalBackMatch;	//!< 右画像から基準画像への逆探索
	bool	doVerticalBackMatch;	//!< 上画像から基準画像への逆探索
	size_t	disparitySearchWidth;	//!< 視差の探索幅
	size_t	disparityMax;		//!< 視差の最大値
	size_t	disparityInconsistency;	//!< 最適視差の不一致の許容値
	size_t	grainSize;		//!< 並列処理の粒度
    };

  protected:
    template <class T>
    class Pool
    {
      public:
		~Pool()
		{
		    while (!_values.empty())
		    {
			T*	value = _values.top();
			_values.pop();
			delete value;
		    }
		}
    
	T*	get()
		{
#if defined(USE_TBB)
		    tbb::spin_mutex::scoped_lock	lock(_mutex);
#endif
		    if (_values.empty())
			_values.push(new T);
		    T*	value = _values.top();
		    _values.pop();
		    return value;
		}
	void	put(T* value)
		{
#if defined(USE_TBB)
		    tbb::spin_mutex::scoped_lock	lock(_mutex);
#endif
		    _values.push(value);
		}
    
      private:
	std::stack<T*>	_values;
#if defined(USE_TBB)
	tbb::spin_mutex	_mutex;
#endif
    };

  private:
#if defined(USE_TBB)
    template <class ROW, class ROW_D>
    class Match
    {
      public:
		Match(STEREO& stereo,
		      ROW rowL, ROW rowLlast, ROW rowR, ROW rowV, ROW_D rowD)
		    :_stereo(stereo), _rowL(rowL), _rowLlast(rowLlast),
		     _rowR(rowR), _rowV(rowV), _rowD(rowD)
		{
		}
	
	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    if (_rowR == _rowV)
			_stereo.match(_rowL + r.begin(),
				      std::min(_rowL + r.end() +
					       _stereo.getOverlap(),
					       _rowLlast),
				      _rowR + r.begin(), _rowD + r.begin());
		    else
			_stereo.match(_rowL + r.begin(),
				      std::min(_rowL + r.end() +
					       _stereo.getOverlap(), 
					       _rowLlast),
				      _rowLlast, _rowR + r.begin(),
				      _rowV, _rowD + r.begin());
		}
	
      private:
	STEREO&		_stereo;
	const ROW	_rowL;
	const ROW	_rowLlast;
	const ROW	_rowR;
	const ROW	_rowV;
	const ROW_D	_rowD;
    };
#endif
    
    template <class DMIN, class DELTA, class DISP, bool HOR_BACKMATCH>
    class CorrectDisparity
    {
      public:
	typedef typename std::iterator_traits<DMIN>::value_type	argument_type;
	typedef DISP						result_type;

      private:
	typedef typename std::is_floating_point<result_type>::type
								is_floating_point;
	typedef std::integral_constant<bool, HOR_BACKMATCH>	hor_backmatch;
	
      public:
	CorrectDisparity(DMIN dminR, DELTA delta,
			 argument_type dmax, argument_type thr)
	    :_dminR(dminR), _delta(delta), _dmax(dmax), _thr(thr)	{}

	result_type	operator ()(argument_type dL) const
			{
			    result_type	val = filter(dL, hor_backmatch());
			    ++_dminR;
			    ++_delta;
			    return val;
			}

      private:
	result_type	filter(argument_type dL, std::true_type) const
			{
			    return (diff(dL, *(_dminR + dL)) <= _thr ?
				    correct(dL, is_floating_point()) : 0);
			}
	result_type	filter(argument_type dL, std::false_type) const
			{
			    return correct(dL, is_floating_point());
			}
	result_type	correct(argument_type dL, std::true_type) const
			{
			    return result_type(_dmax - dL) - *_delta;
			}
	result_type	correct(argument_type dL, std::false_type) const
			{
			    return _dmax - dL;
			}
	
      private:
	mutable DMIN		_dminR;
	mutable DELTA		_delta;
	const argument_type	_dmax;
	const argument_type	_thr;
    };

  public:
    template <class ROW, class ROW_D>
    void	operator ()(ROW rowL, ROW rowLe,
			    ROW rowR, ROW_D rowD)		const	;
    template <class ROW, class ROW_D>
    void	operator ()(ROW rowL, ROW rowLe, ROW rowLlast,
			    ROW rowR, ROW rowV, ROW_D rowD)	const	;

  protected:
    StereoBase(STEREO& stereo, size_t ntimers)
	:Profiler(ntimers), _stereo(stereo)				{}

    template <class DMIN, class DELTA, class COL_D>
    void	selectDisparities(DMIN dminL, DMIN dminLe, DMIN dminR,
				  DELTA delta, COL_D colD)	const	;
    template <class DMINV, class COL_D>
    void	pruneDisparities(DMINV dminV,
				 DMINV dminVe, COL_D colD)	const	;

  private:
    STEREO&	_stereo;
};

template <class STEREO> template <class ROW, class ROW_D> inline void
StereoBase<STEREO>::operator ()(ROW rowL, ROW rowLe,
				ROW rowR, ROW_D rowD) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(
			  0, std::distance(rowL, rowLe),
			  _stereo.getParameters().grainSize),
		      Match<ROW, ROW_D>(_stereo,
					rowL, rowLe, rowR, rowR, rowD));
#else
    _stereo.match(rowL, rowLe, rowR, rowD);
#endif
}
    
template <class STEREO> template <class ROW, class ROW_D> inline void
StereoBase<STEREO>::operator ()(ROW rowL, ROW rowLe, ROW rowLlast,
				ROW rowR, ROW rowV, ROW_D rowD) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(
			  0, std::distance(rowL, rowLe),
			  _stereo.getParameters().grainSize),
		      Match<ROW, ROW_D>(_stereo,
					rowL, rowLlast, rowR, rowV, rowD));
#else
    _stereo.match(rowL, rowLe, rowLlast, rowR, rowV, rowD);
#endif
}
    
//! 右画像からの逆方向視差探索と視差補間を行う
template <class STEREO>
template <class DMIN, class DELTA, class COL_D> inline void
StereoBase<STEREO>::selectDisparities(DMIN dminL, DMIN dminLe, DMIN dminR,
				      DELTA delta, COL_D colD) const
{
    typedef typename std::iterator_traits<COL_D>::value_type	DISP;
    
    const Parameters&	params = _stereo.getParameters();
    
    if (params.doHorizontalBackMatch)
	std::transform(dminL, dminLe, colD,
		       CorrectDisparity<DMIN, DELTA, DISP, true>(
			   dminR, delta,
			   params.disparityMax, params.disparityInconsistency));
    else
	std::transform(dminL, dminLe, colD,
		       CorrectDisparity<DMIN, DELTA, DISP, false>(
			   dminR, delta,
			   params.disparityMax, params.disparityInconsistency));
}

//! 上画像からの逆方向視差探索を行う
template <class STEREO> template <class DMINV, class COL_D> void
StereoBase<STEREO>::pruneDisparities(DMINV dminV,
				     DMINV dminVe, COL_D colD) const
{
    for (; dminV != dminVe; ++dminV)
    {
	if (*colD != 0)
	{
	    const Parameters&	params = _stereo.getParameters();
	    const size_t	dL = params.disparityMax - size_t(*colD);
	    const size_t	dV = *(dminV.operator ->() + dL);
	    if (diff(dL, dV) > params.disparityInconsistency)
		*colD = 0;
	}
	++colD;
    }
}

}
#if defined(PROFILE_BAK)
#  define PROFILE
#endif

#endif	// !__TU_STEREOBASE_H
