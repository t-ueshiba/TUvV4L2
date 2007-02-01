/*
 *  $Id: utility.h,v 1.1 2007-02-01 01:27:19 ueshiba Exp $
 */
#ifndef __TUutility_h
#define __TUutility_h

#include <iterator>

namespace TU
{
template <class T> static inline T	min(T a, T b) {return (a < b ? a : b);}
template <class T> static inline T	max(T a, T b) {return (a > b ? a : b);}
template <class T> static inline T	min(T a, T b, T c)
					{
					    return min(min(a, b), c);
					}
template <class T> static inline T	max(T a, T b, T c)
					{
					    return max(max(a, b), c);
					}
template <class T> static inline T	min(T a, T b, T c, T d)
					{
					    return min(min(a, b, c), d);
					}
template <class T> static inline T	max(T a, T b, T c, T d)
					{
					    return max(max(a, b, c), d);
					}
template <class T> static inline T	diff(T a, T b)
					{
					    return (a > b ? a - b : b - a);
					}
template <class T> static inline T	abs(T a) {return (a > 0 ? a : -a);}
template <class T> static inline void	swap(T& a, T& b)
					{
					    const T tmp = a;
					    a = b;
					    b = tmp;
					}

//! std::pairを要素とするコンテナにおいてpairの第1要素にアクセスする反復子
/*!
  \param Container	std::pair<First, Second>を要素とするコンテナ
  \param Iterator	本反復子のベースとなるContainerの反復子
*/
template <class Container, class Iterator=typename Container::iterator>
class first_iterator : public std::iterator_traits<Iterator>
{
  public:
    typedef typename Iterator::value_type::first_type	value_type;
    typedef value_type*					pointer_type;
    typedef value_type&					reference_type;
	
    first_iterator(const Iterator& i) :_i(i)		{}

    bool		operator ==(const first_iterator& i) const
			{
			    return _i == i._i;
			}
    bool		operator !=(const first_iterator& i) const
			{
			    return !(*this == i);
			}
    reference_type	operator * () const
			{
			    return _i->first;
			}
    pointer_type	operator ->() const
			{
			    return &(operator *());
			}
    first_iterator&	operator ++()
			{
			    ++_i;
			    return *this;
			}
    first_iterator	operator ++(int)
			{
			    first_iterator	tmp = *this;
			    ++_i;
			    return tmp;
			}
    first_iterator&	operator --()
			{
			    --_i;
			    return *this;
			}
    first_iterator	operator --(int)
			{
			    first_iterator	tmp = *this;
			    --_i;
			    return tmp;
			}
	
  private:
    Iterator	_i;
};

//! std::pairを要素とするコンテナにおいてpairの第2要素にアクセスする反復子
/*!
  \param Container	std::pair<First, Second>を要素とするコンテナ
  \param Iterator	本反復子のベースとなるContainerの反復子
*/
template <class Container, class Iterator=typename Container::iterator>
class second_iterator : public std::iterator_traits<Iterator>
{
  public:
    typedef typename Iterator::value_type::second_type	value_type;
    typedef value_type*					pointer_type;
    typedef value_type&					reference_type;
	
    second_iterator(const Iterator& i) :_i(i)		{}

    bool		operator ==(const second_iterator& i) const
			{
			    return _i == i._i;
			}
    bool		operator !=(const second_iterator& i) const
			{
			    return !(*this == i);
			}
    reference_type	operator * () const
			{
			    return _i->second;
			}
    pointer_type	operator ->() const
			{
			    return &(operator *());
			}
    second_iterator&	operator ++()
			{
			    ++_i;
			    return *this;
			}
    second_iterator	operator ++(int)
			{
			    second_iterator	tmp = *this;
			    ++_i;
			    return tmp;
			}
    second_iterator&	operator --()
			{
			    --_i;
			    return *this;
			}
    second_iterator	operator --(int)
			{
			    second_iterator	tmp = *this;
			    --_i;
			    return tmp;
			}
	
  private:
    Iterator	_i;
};

}

#endif	/* __TUutility_h */
