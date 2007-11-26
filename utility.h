/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: utility.h,v 1.14 2007-11-26 07:55:48 ueshiba Exp $
 */
#ifndef __TUutility_h
#define __TUutility_h

#include <algorithm>
#include <iterator>

namespace std
{
/************************************************************************
*  generic algorithms							*
************************************************************************/
//! 3つの引数のうち最小のものを返す．
template <class T> inline const T&
min(const T& a, const T& b, const T& c)
{
    return min(min(a, b), c);
}

//! 3つの引数のうち最大のものを返す．
template <class T> inline const T&
max(const T& a, const T& b, const T& c)
{
    return max(max(a, b), c);
}
    
//! 4つの引数のうち最小のものを返す．
template <class T> inline const T&
min(const T& a, const T& b, const T& c, const T& d)
{
    return min(min(a, b, c), d);
}

//! 4つの引数のうち最大のものを返す．
template <class T> inline const T&
max(const T& a, const T& b, const T& c, const T& d)
{
    return max(max(a, b, c), d);
}

}

/*!
  \namespace	TU
  \brief	本ライブラリで定義されたクラスおよび関数を納める名前空間
*/
namespace TU
{
/************************************************************************
*  generic algorithms							*
************************************************************************/
//! 条件を満たす要素が前半に，そうでないものが後半になるように並べ替える．
/*!
  \param first	データ列の先頭を示す反復子
  \param last	データ列の末尾を示す反復子
  \param pred	条件を指定する単項演算子
  \return	条件を満たさない要素の先頭を示す反復子
*/
template <class Iter, class Pred> Iter
pull_if(Iter first, Iter last, Pred pred)
{
    for (Iter iter = first; iter != last; ++iter)
	if (pred(*iter))
	    std::iter_swap(first++, iter);
    return first;
}

//! 2つの引数の差の絶対値を返す．
template <class T> inline T
diff(const T& a, const T& b)
{
    return (a > b ? a - b : b - a);
}

/************************************************************************
*  class mem_iterator							*
************************************************************************/
//! コンテナの要素の特定のメンバにアクセスする反復子
/*!
  \param Iterator	本反復子のベースとなる反復子
*/
template <class Iterator, class T>
class mbr_iterator
  : public
      std::iterator<typename std::iterator_traits<Iterator>::iterator_category,
		    T,
		    typename std::iterator_traits<Iterator>::difference_type,
		    T*, T&>
{
  public:
    typedef typename std::iterator_traits<Iterator>::iterator_category
							iterator_categoty;
    typedef T						value_type;
    typedef typename std::iterator_traits<Iterator>::difference_type
							difference_type;
    typedef value_type*					pointer;
    typedef value_type&					reference;
    typedef value_type std::iterator_traits<Iterator>::value_type::*
							mbr_pointer;
    
    mbr_iterator(Iterator i, mbr_pointer m)	:_i(i), _m(m)	{}

    Iterator		base() const
			{
			    return _i;
			}
    reference		operator * () const
			{
			    return (*_i).*_m;
			}
    pointer		operator ->() const
			{
			    return &(operator *());
			}
    mbr_iterator&	operator ++()
			{
			    ++_i;
			    return *this;
			}
    mbr_iterator	operator ++(int)
			{
			    mbr_iterator	tmp = *this;
			    ++_i;
			    return tmp;
			}
    mbr_iterator&	operator --()
			{
			    --_i;
			    return *this;
			}
    mbr_iterator	operator --(int)
			{
			    mbr_iterator	tmp = *this;
			    --_i;
			    return tmp;
			}
    mbr_iterator&	operator +=(difference_type n)
			{
			    _i += n;
			    return *this;
			}
    mbr_iterator&	operator -=(difference_type n)
			{
			    _i -= n;
			    return *this;
			}
    mbr_iterator	operator +(difference_type n) const
			{
			    mbr_iterator	tmp = *this;
			    return tmp += n;
			}
    mbr_iterator	operator -(difference_type n) const
			{
			    mbr_iterator	tmp = *this;
			    return tmp -= n;
			}
    reference		operator [](difference_type n) const
			{
			    return *(*this + n);
			}
	
  private:
    Iterator		_i;
    const mbr_pointer	_m;
};

template <class Iterator, class T> inline bool 
operator ==(const mbr_iterator<Iterator, T>& x,
	    const mbr_iterator<Iterator, T>& y) 
{
    return x.base() == y.base();
}

template<class Iterator, class T> inline bool 
operator !=(const mbr_iterator<Iterator, T>& x, 
	    const mbr_iterator<Iterator, T>& y) 
{
    return !(x == y);
}

template<class Iterator, class T> inline bool 
operator <(const mbr_iterator<Iterator, T>& x, 
	   const mbr_iterator<Iterator, T>& y) 
{
    return x.base() < y.base();
}

template<class Iterator, class T> inline bool 
operator >(const mbr_iterator<Iterator, T>& x, 
	   const mbr_iterator<Iterator, T>& y) 
{
    return y < x;
}

template<class Iterator, class T> inline bool 
operator <=(const mbr_iterator<Iterator, T>& x, 
	    const mbr_iterator<Iterator, T>& y) 
{
    return !(y < x);
}

template<class Iterator, class T> inline bool 
operator >=(const mbr_iterator<Iterator, T>& x, 
	    const mbr_iterator<Iterator, T>& y) 
{
    return !(x < y);
}

template<class Iterator, class T>
inline typename mbr_iterator<Iterator, T>::difference_type
operator -(const mbr_iterator<Iterator, T>& x, 
	   const mbr_iterator<Iterator, T>& y) 
{
    return x.base() - y.base();
}

template<class Iterator, class T> inline mbr_iterator<Iterator, T> 
operator +(typename mbr_iterator<Iterator, T>::difference_type n,
	   const mbr_iterator<Iterator, T>& x) 
{
    return x + n;
}

//! T型のメンバを持つオブジェクトを要素とするコンテナについてそのメンバにアクセス(R/W)する反復子を作る．
template <class Iterator, class T> inline mbr_iterator<Iterator, T>
make_mbr_iterator(Iterator i, T std::iterator_traits<Iterator>::value_type::* m)
{
    return mbr_iterator<Iterator, T>(i, m);
}
    
//! T型のメンバを持つオブジェクトを要素とするコンテナについてそのメンバにアクセス(R)する反復子を作る．
template <class Iterator, class T> inline mbr_iterator<Iterator, const T>
make_const_mbr_iterator(Iterator i,
			const T std::iterator_traits<Iterator>::value_type::* m)
{
    return mbr_iterator<Iterator, const T>(i, m);
}

//! std::pairを要素とするコンテナについてpairの第1要素にアクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mbr_iterator<Iterator, typename std::iterator_traits<Iterator>
					  ::value_type::first_type>
make_first_iterator(Iterator i)
{
    return make_mbr_iterator(i, &std::iterator_traits<Iterator>
				    ::value_type::first);
}
    
//! std::pairを要素とするコンテナについてpairの第1要素にアクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mbr_iterator<Iterator, const typename std::iterator_traits<Iterator>
						::value_type::first_type>
make_const_first_iterator(Iterator i)
{
    return make_const_mbr_iterator(i, &std::iterator_traits<Iterator>
					  ::value_type::first);
}
    
//! std::pairを要素とするコンテナについてpairの第2要素にアクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mbr_iterator<Iterator, typename std::iterator_traits<Iterator>
					  ::value_type::second_type>
make_second_iterator(Iterator i)
{
    return make_mbr_iterator(i, &std::iterator_traits<Iterator>
				    ::value_type::second);
}

//! std::pairを要素とするコンテナについてpairの第2要素にアクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mbr_iterator<Iterator, const typename std::iterator_traits<Iterator>
						::value_type::second_type>
make_const_second_iterator(Iterator i)
{
    return make_const_mbr_iterator(i, &std::iterator_traits<Iterator>
					  ::value_type::second);
}
    
}

#endif	/* __TUutility_h */
