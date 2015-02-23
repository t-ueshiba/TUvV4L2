/*!
  \file		StereoBase.h
  \brief	ステレオマッチングクラスの定義と実装
*/
#ifndef __TU_STEREOBASE_H
#define __TU_STEREOBASE_H

#include "TU/Image++.h"
#include "TU/algorithm.h"	// Use std::min(), std::max() and TU::diff().
#include "TU/tuple.h"
#include <limits>		// Use std::numeric_limits<T>.
#include <stack>
#include <tbb/blocked_range.h>

#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/spin_mutex.h>
#  include <tbb/scalable_allocator.h>
#  undef PROFILE
#endif

#if defined(PROFILE)
#  include "TU/Profiler.h"
#else
struct Profiler
{
    Profiler(size_t)				{}

    void	reset()			const	{}
    void	print(std::ostream&)	const	{}
    void	start(int)		const	{}
    void	nextFrame()		const	{}
};
#endif

namespace TU
{
/************************************************************************
*  class Diff<T>							*
************************************************************************/
template <class T>
class Diff
{
  public:
    typedef T						first_argument_type;
    typedef first_argument_type				second_argument_type;
    typedef typename std::conditional<
	std::is_integral<T>::value,
	typename std::make_signed<T>::type, T>::type	result_type;
    
  public:
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
    
#if defined(SSE)
template <class T>
class Diff<mm::vec<T> >
{
  public:
    typedef mm::vec<T>					first_argument_type;
    typedef first_argument_type				second_argument_type;
    typedef typename std::make_signed<T>::type		signed_type;
    typedef mm::vec<signed_type>			result_type;

  public:
    Diff(mm::vec<T> x, mm::vec<T> thresh)	:_x(x), _thresh(thresh)	{}
    
    result_type	operator ()(mm::vec<T> y) const
		{
		    using namespace	mm;

		    return cast<signed_type>(min(diff(_x, y), _thresh));
		}
    result_type	operator ()(std::tuple<mm::vec<T>, mm::vec<T> > y) const
		{
		    return (*this)(std::get<0>(y)) + (*this)(std::get<1>(y));
		}

  private:
    const mm::vec<T>	_x;
    const mm::vec<T>	_thresh;
};
#endif

/************************************************************************
*  class rvcolumn_iterator<COL>						*
************************************************************************/
namespace detail
{
    template <class COL>
    class rvcolumn_proxy
    {
      public:
  	typedef COL	base_iterator;
  	typedef COL	iterator;
	
      public:
  	rvcolumn_proxy(base_iterator col) :_col(col)	{}

  	iterator	begin()			 const	{ return _col; }
	    
      private:
  	const base_iterator	_col;
    };
    
    template <class COL, class COLV>
    class rvcolumn_proxy<fast_zip_iterator<std::tuple<COL, COLV> > >
    {
      public:
  	typedef fast_zip_iterator<std::tuple<COL, COLV> >	base_iterator;
  	typedef fast_zip_iterator<
  	    std::tuple<
  		COL,
		typename std::iterator_traits<COLV>::pointer> >	iterator;

      public:
  	rvcolumn_proxy(base_iterator col)
	    :_rv(col.get_iterator_tuple())			{}

	iterator	begin() const
			{
			    return std::make_tuple(
				       std::get<0>(_rv),
				       std::get<1>(_rv).operator ->());
			}
	
      private:
	const std::tuple<COL, COLV>	_rv;
    };
}

template <class COL>
class rvcolumn_iterator
    : public boost::iterator_adaptor<rvcolumn_iterator<COL>,
				     COL,
				     detail::rvcolumn_proxy<COL>,
				     boost::use_default,
				     detail::rvcolumn_proxy<COL> >
{
  private:
    typedef boost::iterator_adaptor<rvcolumn_iterator<COL>,
				    COL,
				    detail::rvcolumn_proxy<COL>,
				    boost::use_default,
				    detail::rvcolumn_proxy<COL> >	super;

  public:
    typedef typename super::reference	reference;

    friend class			boost::iterator_core_access;

  public:
    rvcolumn_iterator(COL col)	:super(col)			{}
    
  private:
    reference	dereference()	const	{ return reference(super::base()); }
};

template <class COL> rvcolumn_iterator<COL>
make_rvcolumn_iterator(COL col)		{ return rvcolumn_iterator<COL>(col); }

/************************************************************************
*  class dummy_iterator<ITER>						*
************************************************************************/
template <class ITER>
class dummy_iterator
    : public boost::iterator_adaptor<dummy_iterator<ITER>, ITER>
{
  private:
    typedef boost::iterator_adaptor<dummy_iterator, ITER>	super;

  public:
    typedef typename super::difference_type	difference_type;

    friend class				boost::iterator_core_access;

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
*  class mask_iterator<ITER, RV_ITER>					*
************************************************************************/
template <class ITER, class RV_ITER>
class mask_iterator
    : public boost::iterator_adaptor<
		 mask_iterator<ITER, RV_ITER>,
		 ITER,
		 tuple_replace<iterator_value<RV_ITER>, bool>,
		 boost::single_pass_traversal_tag,
		 tuple_replace<iterator_value<RV_ITER>, bool> >
{
  private:
    typedef iterator_value<RV_ITER>			rv_type;
    typedef boost::iterator_adaptor<
		mask_iterator,
		ITER,
		tuple_replace<rv_type, bool>,
		boost::single_pass_traversal_tag,
		tuple_replace<rv_type, bool> >		super;
    typedef tuple_head<rv_type>				element_type;
    
  public:
    typedef typename super::difference_type		difference_type;
    typedef typename super::reference			reference;

    friend class	boost::iterator_core_access;

  public:
		mask_iterator(ITER R, RV_ITER RminRV)
		    :super(R), _Rs(R), _RminL(_Rs), _RminRV(RminRV), _nextRV()
		{
		    setRMost(std::numeric_limits<element_type>::max(), _nextRV);
		}
    int		dL() const
		{
		    return _RminL - _Rs;
		}
      
  private:
    void	setRMost(element_type val, element_type& x)
		{
		    x = val;
		}
    template <class VEC_>
    void	setRMost(element_type val, VEC_& x)
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
		    using namespace	std;

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

/************************************************************************
*  class Idx<T>								*
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

#if defined(SSE)
template <class T>
struct Idx<mm::vec<T> > : mm::vec<T>
{
    typedef mm::vec<T>	super;
    
		Idx()	:super(make_index_sequence<super::size>())	{}
    void	operator ++()		{ *this += super(super::size); }
};

namespace mm
{
  /**********************************************************************
  *  SIMD functions							*
  **********************************************************************/
#  if !defined(SSE2)
  template <u_int I> inline int
  extract(Is32vec x)
  {					// short用の命令を無理に int に適用
      return _mm_extract_pi16(x, I);	// しているため，x が SHRT_MIN 以上かつ
  }					// SHRT_MAX 以下の場合しか有効でない
#  elif !defined(SSE4)
  template <u_int I> inline int
  extract(Is32vec x)
  {					// short用の命令を無理に int に適用
      return _mm_extract_epi16(x, I);	// しているため，x が SHRT_MIN 以上かつ
  }					// SHRT_MAX 以下の場合しか有効でない
#  endif
    
  template <class T, u_int I=vec<T>::size/2> static inline vec<T>
  minIdx(vec<T> d, vec<T> x)
  {
      if (I > 0)
      {
	  const vec<T>	y = shift_r<I>(x);
	  return minIdx<T, (I >> 1)>(select(x < y, d, shift_r<I>(d)), min(x, y));
      }
      else
	  return d;
  }

#  if defined(WITHOUT_CVTDOWN)
  template <class ITER, class RV_ITER>
  class mask_iterator
      : public boost::iterator_adaptor<mask_iterator<ITER, RV_ITER>,
				       ITER,
				       tuple_replace<iterator_value<RV_ITER> >,
				       boost::single_pass_traversal_tag,
				       tuple_replace<iterator_value<RV_ITER> > >
#  else
  template <class T, class ITER, class RV_ITER>
  class mask_iterator
      : public boost::iterator_adaptor<mask_iterator<T, ITER, RV_ITER>,
				       ITER,
				       tuple_replace<iterator_value<RV_ITER>,
						     vec<T> >,
				       boost::single_pass_traversal_tag,
				       tuple_replace<iterator_value<RV_ITER>,
						     vec<T> > >
#  endif
  {
    private:
      typedef iterator_value<RV_ITER>			elementary_vec;
      typedef typename tuple_head<elementary_vec>::element_type
							element_type;
#  if defined(WITHOUT_CVTDOWN)
      typedef boost::iterator_adaptor<
	  mask_iterator,
	  ITER,
	  tuple_replace<elementary_vec>,
	  boost::single_pass_traversal_tag,
	  tuple_replace<elementary_vec> >		super;
#  else
      typedef boost::iterator_adaptor<
	  mask_iterator,
	  ITER,
	  tuple_replace<elementary_vec, vec<T> >,
	  boost::single_pass_traversal_tag,
	  tuple_replace<elementary_vec, vec<T> > >	super;
      typedef typename type_traits<element_type>::complementary_mask_type
							complementary_type;
      typedef tuple_replace<elementary_vec, vec<complementary_type> >
							complementary_vec;
      typedef typename std::conditional<
	  std::is_floating_point<element_type>::value,
	  complementary_type, element_type>::type	integral_type;
      typedef tuple_replace<elementary_vec, vec<integral_type> >
							integral_vec;
      typedef typename std::conditional<
	  std::is_signed<integral_type>::value,
	  typename type_traits<integral_type>::unsigned_type,
	  typename type_traits<integral_type>::signed_type>::type
							flipped_type;
      typedef tuple_replace<elementary_vec, vec<flipped_type> >
							flipped_vec;
      typedef typename type_traits<flipped_type>::lower_type
							flipped_lower_type;
      typedef tuple_replace<elementary_vec, vec<flipped_lower_type> >
							flipped_lower_vec;
#  endif
      
    public:
      typedef typename super::difference_type		difference_type;
      typedef typename super::reference			reference;

      friend class	boost::iterator_core_access;

    public:
		mask_iterator(ITER R, RV_ITER RminRV)
		    :super(R),
		     _index(),
		     _dminL(_index),
		     _RminL(std::numeric_limits<element_type>::max()),
		     _RminRV(RminRV),
		     _nextRV()
		{
		    setRMost(std::numeric_limits<element_type>::max(), _nextRV);
		}
      int	dL()	const	{ return extract<0>(minIdx(_dminL, _RminL)); }
	
    private:
    // mask と mask tuple に対するsetRMost
      void	setRMost(element_type val, vec<element_type>& x)
		{
		    x = set_rmost<element_type>(val);
		}
      template <class VEC_>
      void	setRMost(element_type val, VEC_& x)
		{
		    vec<element_type>	next = set_rmost<element_type>(val);
		    x = std::make_tuple(next, next);
		}

    // mask と mask tuple に対するupdate
      void	update(vec<element_type> R, vec<element_type>& x)
		{
		    vec<element_type>	RminR  = _RminRV();
		    vec<element_type>	minval = min(R, RminR);
		    *_RminRV = shift_l<1>(minval) | _nextRV;
		    ++_RminRV;
		    _nextRV  = shift_lmost_to_rmost(minval);
		    
		    x = (R < RminR);
		}
      template <class VEC_>
      void	update(vec<element_type> R, VEC_& x)
		{
		    using namespace	std;

		    vec<element_type>
			RminR = get<0>(_RminRV.get_iterator_tuple())(),
			RminV = get<1>(_RminRV.get_iterator_tuple())();
		    vec<element_type>	minvalR = min(R, RminR),
					minvalV = min(R, RminV);
		    *_RminRV = make_tuple(
				   shift_l<1>(minvalR) | get<0>(_nextRV),
				   shift_l<1>(minvalV) | get<1>(_nextRV));
		    ++_RminRV;
		    _nextRV = make_tuple(shift_lmost_to_rmost(minvalR),
					 shift_lmost_to_rmost(minvalV));
		    
		    x = make_tuple(R < RminR, R < RminV);
		}

      void	cvtdown(elementary_vec& x)
		{
		    vec<element_type>	R = *super::base();
		    ++super::base_reference();

		    _dminL = select(R < _RminL, _index, _dminL);
		    _RminL = min(R, _RminL);
		    ++_index;

		    update(R, x);
		}
#  if !defined(WITHOUT_CVTDOWN)
      void	cvtdown(complementary_vec& x)
		{
		    elementary_vec	y;
		    cvtdown(y);
		    x = cvt_mask<complementary_type>(y);
		}
      void	cvtdown(flipped_vec& x)
		{
		    integral_vec	y;
		    cvtdown(y);
		    x = cvt_mask<flipped_type>(y);
		}
      void	cvtdown(flipped_lower_vec& x)
		{
		    integral_vec	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<flipped_lower_type>(y, z);
		}
      template <class VEC_>
      void	cvtdown(VEC_& x)
		{
		    typedef
			typename tuple_head<VEC_>::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;
		    
		    tuple_replace<elementary_vec, vec<upper_type> >	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<S>(y, z);
		}
#  endif
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
      Idx<vec<element_type> >	_index;
      vec<element_type>		_dminL;
      vec<element_type>		_RminL;
      RV_ITER			_RminRV;
      elementary_vec		_nextRV;
  };

# if defined(WITHOUT_CVTDOWN)
  template <class ITER, class RV_ITER> mask_iterator<ITER, RV_ITER>
  make_mask_iterator(ITER R, RV_ITER RminRV)
  {
      return mask_iterator<ITER, RV_ITER>(R, RminRV);
  }
# endif
}	// end of namespace mm
#endif

/************************************************************************
*  class StereoBase<STEREO>						*
************************************************************************/
template <class STEREO>
class StereoBase : public Profiler
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
#endif	// !__TU_STEREOBASE_H
