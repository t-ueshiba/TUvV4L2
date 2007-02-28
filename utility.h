/*
 *  $Id: utility.h,v 1.8 2007-02-28 00:16:06 ueshiba Exp $
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
class mem_iterator
    : public std::iterator<typename Iterator::iterator_category, T,
			   typename Iterator::difference_type, T*, T&>
{
  public:
    typedef typename Iterator::iterator_category	iterator_categoty;
    typedef T						value_type;
    typedef typename Iterator::difference_type		difference_type;
    typedef value_type*					pointer;
    typedef value_type&					reference;
    typedef value_type Iterator::value_type::*		mem_pointer;
    
    mem_iterator(const Iterator& i, mem_pointer m)
	:_i(i), _m(m)						{}

    bool		operator ==(const mem_iterator& i) const
			{
			    return _i == i._i;
			}
    bool		operator !=(const mem_iterator& i) const
			{
			    return !(*this == i);
			}
    reference		operator * () const
			{
			    return (*_i).*_m;
			}
    pointer		operator ->() const
			{
			    return &(operator *());
			}
    mem_iterator&	operator ++()
			{
			    ++_i;
			    return *this;
			}
    mem_iterator	operator ++(int)
			{
			    mem_iterator	tmp = *this;
			    ++_i;
			    return tmp;
			}
    mem_iterator&	operator --()
			{
			    --_i;
			    return *this;
			}
    mem_iterator	operator --(int)
			{
			    mem_iterator	tmp = *this;
			    --_i;
			    return tmp;
			}
	
  private:
    Iterator		_i;
    const mem_pointer	_m;
};

//! std::pairを要素とするコンテナについてpairの第1要素にアクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mem_iterator<Iterator, typename Iterator::value_type::first_type>
make_first_iterator(const Iterator& i)
{
    return mem_iterator<Iterator, typename Iterator::value_type::first_type>
		(i, &Iterator::value_type::first);
}
    
//! std::pairを要素とするコンテナについてpairの第1要素にアクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mem_iterator<Iterator, const typename Iterator::value_type::first_type>
make_const_first_iterator(const Iterator& i)
{
    return mem_iterator<Iterator,
			const typename Iterator::value_type::first_type>
		(i, &Iterator::value_type::first);
}
    
//! std::pairを要素とするコンテナについてpairの第2要素にアクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mem_iterator<Iterator, typename Iterator::value_type::second_type>
make_second_iterator(const Iterator& i)
{
    return mem_iterator<Iterator, typename Iterator::value_type::second_type>
		(i, &Iterator::value_type::second);
}

//! std::pairを要素とするコンテナについてpairの第2要素にアクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline mem_iterator<Iterator, const typename Iterator::value_type::second_type>
make_const_second_iterator(const Iterator& i)
{
    return mem_iterator<Iterator,
			const typename Iterator::value_type::second_type>
		(i, &Iterator::value_type::second);
}
    
}

#endif	/* __TUutility_h */
