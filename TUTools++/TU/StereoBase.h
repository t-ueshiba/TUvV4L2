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
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef int		result_type;
    
  public:
    Diff(T thresh)	:_thresh(thresh)		{}
    
    result_type	operator ()(T x, T y) const
		{
		    return std::min(diff(x, y), _thresh);
		}
    
  private:
    const T	_thresh;
};
    
template <>
class Diff<RGBA>
{
  public:
    typedef RGBA	first_argument_type;
    typedef RGBA	second_argument_type;
    typedef int		result_type;
    
  public:
    Diff(u_char thresh)	:_diff(thresh)			{}
    
    result_type	operator ()(RGBA x, RGBA y) const
		{
		    return _diff(x.r, y.r) + _diff(x.g, y.g) + _diff(x.b, y.b);
		}
    
  private:
    const Diff<u_char>	_diff;
};

template <class T>
class Diff<boost::tuples::cons<
	       T, boost::tuples::cons<T, boost::tuples::null_type> > >
{
  public:
    typedef T						first_argument_type;
    typedef boost::tuple<const T&, const T&>		second_argument_type;
    typedef typename Diff<T>::result_type		result_type;

  public:
    Diff(T thresh)	:_diff(thresh)			{}

    result_type	operator ()(T x, boost::tuple<const T&, const T&> y) const
		{
		    using namespace	boost;
		    
		    return _diff(x, get<0>(y)) + _diff(x, get<1>(y));
		}

  private:
    const Diff<T>	_diff;
};

#if defined(SSE)
template <class T>
class Diff<mm::vec<T> >
{
  public:
    typedef mm::vec<T>					first_argument_type;
    typedef mm::vec<T>					second_argument_type;
    typedef typename mm::type_traits<T>::signed_type	signed_type;
    typedef mm::vec<signed_type>			result_type;

  public:
    Diff(mm::vec<T> thresh)	:_thresh(thresh)	{}
    
    result_type	operator ()(mm::vec<T> x, mm::vec<T> y) const
		{
		    using namespace	mm;

		    return cast<signed_type>(min(diff(x, y), _thresh));
		}

  private:
    const mm::vec<T>	_thresh;
};
#endif

/************************************************************************
*  exec_assignment							*
************************************************************************/
template <template <class, class> class ASSIGN, class S, class T> void
exec_assignment(S x, T y)
{
    ASSIGN<S, T>()(x, y);
}

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
    class rvcolumn_proxy<fast_zip_iterator<boost::tuple<COL, COLV> > >
    {
      public:
  	typedef fast_zip_iterator<boost::tuple<COL, COLV> >	base_iterator;
  	typedef fast_zip_iterator<
  	    boost::tuple<
  		COL,
		typename std::iterator_traits<COLV>::pointer> >	iterator;

      public:
  	rvcolumn_proxy(base_iterator col)
	    :_rv(col.get_iterator_tuple())			{}

	iterator	begin() const
			{
			    using namespace	boost;
			    
			    return make_tuple(get<0>(_rv),
					      get<1>(_rv).operator ->());
			}
	
      private:
	const boost::tuple<COL, COLV>	_rv;
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
class dummy_iterator : public boost::iterator_adaptor<dummy_iterator<ITER>,
						      ITER>
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
	  typename tuple2cons<
	      typename iterator_value<RV_ITER>::type, bool>::type,
	  boost::single_pass_traversal_tag,
	  typename tuple2cons<
	      typename iterator_value<RV_ITER>::type, bool>::type>
{
  private:
    typedef typename iterator_value<RV_ITER>::type	rv_type;
    typedef boost::iterator_adaptor<
	mask_iterator,
	ITER,
	typename tuple2cons<rv_type, bool>::type,
	boost::single_pass_traversal_tag,
	typename tuple2cons<rv_type, bool>::type>	super;
    typedef typename tuple_head<rv_type>::type		element_type;
    
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
    template <class _VEC>
    void	setRMost(element_type val, _VEC& x)
		{
		    x = boost::make_tuple(val, val);
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
    template <class _VEC>
    void	update(element_type R, _VEC& mask)
		{
		    using namespace	boost;

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
namespace mm
{
  /**********************************************************************
  *  SIMD functions							*
  **********************************************************************/
    template <class TUPLE, class T> inline TUPLE
    select(const TUPLE& mask, vec<T> index, const TUPLE& dmin)
    {
	using namespace	boost;
      
	return make_tuple(select(get<0>(mask), index, get<0>(dmin)),
			  select(get<1>(mask), index, get<1>(dmin)));
    }

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
    
    template <class T> static vec<T>	initIdx()			;
#  define MM_INITIDX32(type)						\
    template <> inline vec<type>					\
    initIdx<type>()							\
    {									\
	return vec<type>(31,30,29,28,27,26,25,24,			\
			 23,22,21,20,19,18,17,16,			\
			 15,14,13,12,11,10, 9, 8,			\
			  7, 6, 5, 4, 3, 2, 1, 0);			\
    }
#  define MM_INITIDX16(type)						\
    template <> inline vec<type>					\
    initIdx<type>()							\
    {									\
	return vec<type>(15,14,13,12,11,10, 9, 8,			\
			  7, 6, 5, 4, 3, 2, 1, 0);			\
    }
#  define MM_INITIDX8(type)						\
    template <> inline vec<type>					\
    initIdx<type>()							\
    {									\
	return vec<type>(7, 6, 5, 4, 3, 2, 1, 0);			\
    }
#  define MM_INITIDX4(type)						\
    template <> inline vec<type>					\
    initIdx<type>()							\
    {									\
	return vec<type>(3, 2, 1, 0);					\
    }
#  define MM_INITIDX2(type)						\
    template <> inline vec<type>					\
    initIdx<type>()							\
    {									\
	return vec<type>(1, 0);						\
    }
    
#  if defined(AVX2)
    MM_INITIDX32(u_char)
    MM_INITIDX16(short)
    MM_INITIDX16(u_short)
    MM_INITIDX8(int)
    MM_INITIDX8(u_int)
#  elif defined(SSE2)
    MM_INITIDX16(u_char)
    MM_INITIDX8(short)
    MM_INITIDX8(u_short)
    MM_INITIDX4(int)
    MM_INITIDX4(u_int)
#  else
    MM_INITIDX8(u_char)
    MM_INITIDX4(short)
    MM_INITIDX4(u_short)
    MM_INITIDX2(int)
    MM_INITIDX2(u_int)
#  endif
#  if defined(AVX)
    MM_INITIDX8(float)
#  elif defined(SSE)
    MM_INITIDX4(float)
#  endif
#  undef MM_INITIDX32
#  undef MM_INITIDX16
#  undef MM_INITIDX8
#  undef MM_INITIDX4
#  undef MM_INITIDX2

    template <class T, u_int I=vec<T>::size/2> static inline vec<T>
    minIdx(vec<T> d, vec<T> x)
    {
	if (I > 0)
	{
	    const vec<T>	y = shift_r<I>(x);
	    return minIdx<T, (I >> 1)>(select(x < y, d, shift_r<I>(d)),
				     min(x, y));
	}
	else
	    return d;
    }

#  if defined(WITHOUT_CVTDOWN)
    template <class ITER, class RV_ITER>
    class mask_iterator
	: public boost::iterator_adaptor<
		mask_iterator<ITER, RV_ITER>,
		ITER,
		typename tuple2cons<
		    typename iterator_value<RV_ITER>::type>::type,
		boost::single_pass_traversal_tag,
		typename tuple2cons<
		    typename iterator_value<RV_ITER>::type>::type>
#  else
    template <class T, class ITER, class RV_ITER>
    class mask_iterator
	: public boost::iterator_adaptor<
		mask_iterator<T, ITER, RV_ITER>,
		ITER,
		typename tuple2cons<
		    typename iterator_value<RV_ITER>::type, vec<T> >::type,
		boost::single_pass_traversal_tag,
		typename tuple2cons<
		    typename iterator_value<RV_ITER>::type, vec<T> >::type>
#  endif
    {
      private:
	typedef typename iterator_value<RV_ITER>::type	elementary_vec;
	typedef typename tuple_head<elementary_vec>
				::type::element_type	element_type;
#  if defined(WITHOUT_CVTDOWN)
	typedef boost::iterator_adaptor<
	    mask_iterator,
	    ITER,
	    typename tuple2cons<elementary_vec>::type,
	    boost::single_pass_traversal_tag,
	    typename tuple2cons<elementary_vec>::type>	super;
#  else
	typedef boost::iterator_adaptor<
	    mask_iterator,
	    ITER,
	    typename tuple2cons<
		elementary_vec, vec<T> >::type,
	    boost::single_pass_traversal_tag,
	    typename tuple2cons<
		elementary_vec, vec<T> >::type>		super;
	typedef typename type_traits<element_type>::complementary_mask_type
							complementary_type;
	typedef typename tuple2cons<
	    elementary_vec,
	    vec<complementary_type> >::type		complementary_vec;
	typedef typename boost::mpl::if_<
	    boost::is_floating_point<element_type>,
	    complementary_type, element_type>::type	integral_type;
	typedef typename tuple2cons<
	    elementary_vec, vec<integral_type> >::type	integral_vec;
	typedef typename boost::mpl::if_<
	    boost::is_signed<integral_type>,
	    typename type_traits<integral_type>::unsigned_type,
	    typename type_traits<integral_type>::signed_type>::type
							flipped_type;
	typedef typename tuple2cons<
	    elementary_vec, vec<flipped_type> >::type	flipped_vec;
	typedef typename type_traits<flipped_type>::lower_type
							flipped_lower_type;
	typedef typename tuple2cons<
	    elementary_vec,
	    vec<flipped_lower_type> >::type		flipped_lower_vec;
#  endif
      
      public:
	typedef typename super::difference_type		difference_type;
	typedef typename super::reference		reference;

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
	template <class _VEC>
	void	setRMost(element_type val, _VEC& x)
		{
		    vec<element_type>	next = set_rmost<element_type>(val);
		    x = boost::make_tuple(next, next);
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
	template <class _VEC>
	void	update(vec<element_type> R, _VEC& x)
		{
		    using namespace	boost;

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
	template <class _VEC>
	void	cvtdown(_VEC& x)
		{
		    typedef typename tuple_head<_VEC>
					::type::element_type	S;
		    typedef typename type_traits<S>::upper_type	upper_type;
		    
		    typename tuple2cons<
			elementary_vec, vec<upper_type> >::type	y, z;
		    cvtdown(y);
		    cvtdown(z);
		    x = cvt_mask<S>(y, z);
		}

    // mask と mask tuple に対するcvt
	template <class _S, class _T> static
        vec<_S>	cvt_mask(vec<_T> x)
		{
		    return mm::cvt_mask<_S>(x);
		}
	template <class _S, class _T> static
	vec<_S>	cvt_mask(vec<_T> x, vec<_T> y)
		{
		    return mm::cvt_mask<_S>(x, y);
		}
	template <class _S, class _TUPLE> static boost::tuple<vec<_S>, vec<_S> >
		cvt_mask(const _TUPLE& x)
		{
		    using namespace	boost;
		    
		    return make_tuple(cvt_mask<_S>(get<0>(x)),
				      cvt_mask<_S>(get<1>(x)));
		}
	template <class _S, class _TUPLE> static boost::tuple<vec<_S>, vec<_S> >
		cvt_mask(const _TUPLE& x, const _TUPLE& y)
		{
		    using namespace	boost;
		    
		    return make_tuple(cvt_mask<_S>(get<0>(x), get<0>(y)),
				      cvt_mask<_S>(get<1>(x), get<1>(y)));
		}
#  endif
	reference
		dereference() const
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
	vec<element_type>	_dminL;
	vec<element_type>	_RminL;
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

template <class T>
struct Idx<mm::vec<T> > : mm::vec<T>
{
    typedef mm::vec<T>	super;
    
		Idx() :super(mm::initIdx<T>())	{}
    void	operator ++()			{*this += super(super::size);}
};

#else
template <class T> inline T
select(bool mask, Idx<T> index, T dmin)
{
    return (mask ? index : dmin);
}

template <class MASK, class T, class DMIN> inline DMIN
select(const MASK& mask, Idx<T> index, const DMIN& dmin)
{
    using namespace	boost;

    return make_tuple(select(get<0>(mask), index, get<0>(dmin)),
		      select(get<1>(mask), index, get<1>(dmin)));
}
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
    struct Allocator
    {
#if defined(USE_TBB)
	typedef tbb::scalable_allocator<T>	type;
#elif defined(SSE)
	typedef mm::allocator<T>		type;
#else
	typedef std::allocator<T>		type;
#endif
    };

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
	typedef boost::mpl::bool_<
	  boost::is_floating_point<result_type>::value>	is_floating_point;
	typedef boost::mpl::bool_<HOR_BACKMATCH>	hor_backmatch;
	
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
	result_type	filter(argument_type dL, boost::mpl::true_) const
			{
			    return (diff(dL, *(_dminR + dL)) <= _thr ?
				    correct(dL, is_floating_point()) : 0);
			}
	result_type	filter(argument_type dL, boost::mpl::false_) const
			{
			    return correct(dL, is_floating_point());
			}
	result_type	correct(argument_type dL, boost::mpl::true_) const
			{
			    return result_type(_dmax - dL) - *_delta;
			}
	result_type	correct(argument_type dL, boost::mpl::false_) const
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
