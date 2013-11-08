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
  \file		Filter2.h
  \brief	水平/垂直方向に分離不可能な2次元フィルタを表すクラスの定義
*/
#ifndef __TUFilter2_h
#define __TUFilter2_h

#include "TU/iterator.h"
#if defined(USE_TBB)
#  include <tbb/parallel_for.h>
#  include <tbb/blocked_range.h>
#endif

namespace TU
{
/************************************************************************
*  strength<T>								*
************************************************************************/
//! 水平/垂直方向1階微分値からその強度を計算
template <class T>
struct strength
{
    typedef T	first_argument_type;
    typedef T	second_argument_type;
    typedef T	result_type;
	
    result_type
    operator ()(first_argument_type eH, second_argument_type eV) const
    {
	return std::sqrt(eH*eH + eV*eV);
    }

#if defined(SSE)
    struct mm_
    {
	typedef mm::vec<T>	first_argument_type;
	typedef mm::vec<T>	second_argument_type;
	typedef mm::vec<T>	result_type;

	mm_()			{}
	mm_(const strength&)	{}

	result_type
	operator ()(first_argument_type eH, second_argument_type eV) const
	{
	    return mm::sqrt(eH*eH + eV*eV);
	}
    };
#endif
};
    
/************************************************************************
*  dir4<T>								*
************************************************************************/
//! 2次元ベクトルの方向を右/下/左/上の4種に分類
template <class T>
struct dir4
{
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef u_int	result_type;
	
    result_type
    operator ()(first_argument_type u, second_argument_type v) const
    {
	return (u < -v ? (u < v ? 4 : 6) : (u < v ? 2 : 0));
    }

#if defined(AVX2) || (!defined(AVX) && defined(SSE2))
    struct mm_
    {
	typedef mm::vec<T>			first_argument_type;
	typedef mm::vec<T>			second_argument_type;
	typedef typename mm::type_traits<T>::signed_type
						value_type;
	typedef mm::vec<value_type>		result_type;
    
	mm_()					{}
	mm_(const dir4&)			{}
	
	result_type
	operator ()(first_argument_type u, second_argument_type v) const
	{
	    using namespace	mm;
		    
	    const result_type	l4 = cast<signed_type>(u < -v);
	    return (l4				    & result_type(0x4))
		 | ((cast<signed_type>(u < v) ^ l4) & result_type(0x2));
	}
    };
#endif
};
    
/************************************************************************
*  dir4x<T>								*
************************************************************************/
//! 2次元ベクトルの方向を右下/左下/左上/右上の4種に分類
template <class T>
struct dir4x
{
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef T		third_argument_type;
    typedef u_int	result_type;
	
    result_type
    operator ()(first_argument_type u, second_argument_type v) const
    {
	return (v < 0 ? (u < 0 ? 5 : 7) : (u < 0 ? 3 : 1));
    }

    result_type
    operator ()(first_argument_type  u,
		second_argument_type v, third_argument_type lambda) const
    {
	const u_int	l = (u < 0) ^ (v < 0);
	return (((lambda < 0) ^ l) << 2) | (l << 1) | 0x1;
    }
    
#if defined(AVX2) || (!defined(AVX) && defined(SSE2))
    struct mm_
    {
	typedef mm::vec<T>			first_argument_type;
	typedef mm::vec<T>			second_argument_type;
	typedef mm::vec<T>			third_argument_type;
	typedef typename mm::type_traits<T>::signed_type
						value_type;
	typedef mm::vec<value_type>		result_type;
    
	mm_()					{}
	mm_(const dir4x&)			{}
	
	result_type
	operator ()(first_argument_type u, second_argument_type v) const
	{
	    using namespace	mm;
		    
	    const result_type	l4 = cast<value_type>(v < zero<T>());
	    return (l4 & result_type(0x4))
		 | ((cast<value_type>(u < zero<T>()) ^ l4) & result_type(0x2))
		 | result_type(0x1);
	}

	result_type
	operator ()(first_argument_type  u,
		    second_argument_type v, third_argument_type lambda) const
	{
	    using namespace	mm;
		    
	    const result_type	l = cast<value_type>((u < zero<T>()) ^
						     (v < zero<T>()));
	    return ((l ^ cast<value_type>(lambda < zero<T>())) &
		    result_type(0x4))
		 | (l & result_type(0x2)) | result_type(0x1);
	}
    };
#endif    
};
    
/************************************************************************
*  dir8<T>								*
************************************************************************/
//! 8次元ベクトルの方向を右/右下/下/左下/左/左上/上/右上の8種に分類
template <class T>
struct dir8
{
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef u_int	result_type;
	
    result_type
    operator ()(first_argument_type u, second_argument_type v) const
    {
	const T	su = _slant * u, sv = _slant * v;
	return (su < -v ?
		( u < -sv ?
		  (u <  sv ? (su <   v ? 4 : 5) : 6) : 7) :
		(su <   v ?
		 (u <  sv ? ( u < -sv ? 3 : 2) : 1) : 0));
    }
    
#if defined(AVX2) || (!defined(AVX) && defined(SSE2))
    struct mm_
    {
	typedef mm::vec<T>			first_argument_type;
	typedef mm::vec<T>			second_argument_type;
	typedef typename mm::type_traits<T>::signed_type
						value_type;
	typedef mm::vec<value_type>		result_type;

	mm_()					{}
	mm_(const dir8&)			{}
	
	result_type
	operator ()(first_argument_type u, second_argument_type v) const
	{
	    using namespace	mm;

	    const vec<T>	su = vec<T>(_slant) * u,
				sv = vec<T>(_slant) * v;
	    const result_type	l2 = cast<value_type>( u <  sv),
				l4 = cast<value_type>(su <  -v);
	    return (l4					& result_type(0x4))
		 | ((l2 ^ l4)				& result_type(0x2))
		 | (((cast<value_type>(su <   v) ^ l2) |
		     (cast<value_type>( u < -sv) ^ l4)) & result_type(0x1));
	}
    };
#endif
  private:
    static const T	_slant;
};

template <class T> const T	dir8<T>::_slant = 0.41421356;	// tan(M_PI/8)
    
/************************************************************************
*  dir8x<T>								*
************************************************************************/
//! 8次元ベクトルの方向を右右下/右下下/左下下/左左下/左左上/左上上/右上上/右右上の8種に分類
template <class T>
struct dir8x
{
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef T		third_argument_type;
    typedef u_int	result_type;
	
    result_type
    operator ()(first_argument_type u, second_argument_type v) const
    {
	return (v < 0 ?
		(u < -v ? (u < 0 ? (u <  v ? 4 : 5) : 6) : 7) :
		(u <  v ? (u < 0 ? (u < -v ? 3 : 2) : 1) : 0));
    }

    result_type
    operator ()(first_argument_type  u,
		second_argument_type v, third_argument_type lambda) const
    {
	const u_int	l2 = (u < 0), l4 = (v < 0), l = l2 ^ l4;
	return (((lambda < 0) ^ l) << 2) | (l << 1)
	     | ((u <  v) ^ l2)		 | ((u < -v) ^ l4);
    }
    
#if defined(AVX2) || (!defined(AVX) && defined(SSE2))
    struct mm_
    {
	typedef mm::vec<T>			first_argument_type;
	typedef mm::vec<T>			second_argument_type;
	typedef mm::vec<T>			third_argument_type;
	typedef typename mm::type_traits<T>::signed_type
						value_type;
	typedef mm::vec<value_type>		result_type;
    
	mm_()					{}
	mm_(const dir8x&)			{}

	result_type
	operator ()(first_argument_type u, second_argument_type v) const
	{
	    using namespace	mm;
		    
	    const result_type	l2 = cast<value_type>(u < zero<T>()),
				l4 = cast<value_type>(v < zero<T>()),
				l  = l2 ^ l4;
	    return (l4				      & result_type(0x4))
		 | (l				      & result_type(0x2))
		 | (((cast<value_type>(u <  v) ^ l2) |
		     (cast<value_type>(u < -v) ^ l4)) & result_type(0x1));
	}
	
	result_type
	operator ()(first_argument_type  u,
		    second_argument_type v, third_argument_type lambda) const
	{
	    using namespace	mm;
	    
	    const result_type	l2 = cast<value_type>(u < zero<T>()),
				l4 = cast<value_type>(v < zero<T>()),
				l  = l2 ^ l4;
	    return ((cast<value_type>(lambda < zero<T>()) ^ l)
						      & result_type(0x4))
		 | (l				      & result_type(0x2))
		 | (((cast<value_type>(u <  v) ^ l2) |
		     (cast<value_type>(u < -v) ^ l4)) & result_type(0x1));
	}
    };
#endif    
};
    
/************************************************************************
*  eigen<T>								*
************************************************************************/
//! 2x2対称行列の絶対値が大きい方の固有値を求める
template <class T>
struct eigen
{
    typedef T		first_argument_type;
    typedef T		second_argument_type;
    typedef T		third_argument_type;
    typedef T		result_type;
	
    result_type
    operator ()(first_argument_type  a,
		second_argument_type b, third_argument_type c) const
    {
	const T	avrg = T(0.5)*(a + c), diff = T(0.5)*(a - c),
		frac = std::sqrt(diff*diff + b*b);
	return (avrg > T(0) ? avrg + frac : avrg - frac);
    }
    
#if defined(SSE)
    struct mm_
    {
	typedef mm::vec<T>	first_argument_type;
	typedef mm::vec<T>	second_argument_type;
	typedef mm::vec<T>	third_argument_type;
	typedef mm::vec<T>	result_type;

	mm_()			{}
	mm_(const eigen&)	{}

	result_type
	operator ()(first_argument_type  a,
		    second_argument_type b, third_argument_type c) const
	{
	    using namespace	mm;
		    
	    const vec<T>	avrg = avg(a, c), diff = sub_avg(a, c),
				frac = sqrt(diff*diff + b*b);
	    return select(avrg > zero<T>(), avrg + frac, avrg - frac);
	}
    };
#endif
};

/************************************************************************
*  class Filter2							*
************************************************************************/
//! 水平/垂直方向に分離不可能な2次元フィルタを表すクラス
class Filter2
{
  public:
#if defined(USE_TBB)
  private:
    template <class IN, class OUT>
    class FilterRows
    {
      public:
	FilterRows(IN const& in, OUT const& out) :_in(in), _out(out)	{}

	void	operator ()(const tbb::blocked_range<size_t>& r) const
		{
		    IN	ib = _in, ie = _in;
		    std::advance(ib, r.begin());
		    std::advance(ie, r.end());
		    OUT	out = _out;
		    std::advance(out, r.begin());
		    Filter2::filterRows(ib, ie, out);
		}

      private:
	IN     const&	_in;
	OUT    const&	_out;
    };
#endif
  public:
    Filter2()	:_grainSize(1)				{}

    template <class IN, class OUT>
    void	operator ()(IN ib, IN ie, OUT out) const;
    size_t	grainSize()			const	{ return _grainSize; }
    void	setGrainSize(size_t gs)			{ _grainSize = gs; }

  private:
    template <class IN, class OUT>
    static void	filterRows(IN ib, IN ie, OUT out)	;
    
  private:
    size_t	_grainSize;
};
    
//! 与えられた2次元配列にこのフィルタを適用する
/*!
  \param ib	入力2次元データ配列の先頭行を指す反復子
  \param ie	入力2次元データ配列の末尾の次の行を指す反復子
  \param out	出力2次元データ配列の先頭行を指す反復子
*/
template <class IN, class OUT> void
Filter2::operator ()(IN ib, IN ie, OUT out) const
{
#if defined(USE_TBB)
    tbb::parallel_for(tbb::blocked_range<size_t>(0, std::distance(ib, ie),
						 _grainSize),
		      FilterRows<IN, OUT>(ib, out));
#else
    filterRows(ib, ie, out);
#endif
}

template <class IN, class OUT> void
Filter2::filterRows(IN ib, IN ie, OUT out)
{
    for (; ib != ie; ++ib, ++out)
	std::copy(ib->begin(), ib->end(), out->begin());
}

}
#endif	// !__TUFilter2_h
