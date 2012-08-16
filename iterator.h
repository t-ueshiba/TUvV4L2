/*
 *  $Id: iterator.h,v 1.1 2012-08-16 01:30:37 ueshiba Exp $
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#ifndef __TUiterator_h
#define __TUiterator_h

#include <iterator>
#include <boost/iterator/transform_iterator.hpp>
#include "TU/functional.h"

namespace TU
{
//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R/W)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<mem_var_ref_t<S, T>, Iterator>
make_mbr_iterator(Iterator i, S T::* m)
{
    return boost::make_transform_iterator(i, mem_var_ref(m));
}
    
//! S型のメンバ変数を持つT型オブジェクトへの反復子からそのメンバに直接アクセス(R)する反復子を作る．
template <class Iterator, class S, class T>
inline boost::transform_iterator<const_mem_var_ref_t<S, T>, Iterator>
make_const_mbr_iterator(Iterator i, S const T::* m)
{
    return boost::make_transform_iterator(i, mem_var_ref(m));
}

//! std::pairへの反復子からその第1要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_first_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第1要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::first_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_first_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::first);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R/W)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::second_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_second_iterator(Iterator i)
{
    return make_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
//! std::pairへの反復子からその第2要素に直接アクセス(R)する反復子を作る．
/*!
  \param i	ベースとなる反復子
*/
template <class Iterator>
inline boost::transform_iterator<
    const_mem_var_ref_t<
	typename std::iterator_traits<Iterator>::value_type::second_type,
	typename std::iterator_traits<Iterator>::value_type>,
    Iterator>
make_const_second_iterator(Iterator i)
{
    return make_const_mbr_iterator(
		i, &std::iterator_traits<Iterator>::value_type::second);
}
    
/************************************************************************
*  class box_filter_iterator						*
************************************************************************/
//! コンテナ中の指定された要素に対してbox filterを適用した結果を返す反復子
/*!
  \param Iterator	コンテナ中の要素を指す定数反復子の型
*/
template <class Iterator>
class box_filter_iterator
    : public std::iterator<std::input_iterator_tag,
			   typename std::iterator_traits<Iterator>::value_type>
{
  private:
    typedef std::iterator<std::input_iterator_tag,
			  typename std::iterator_traits<Iterator>::value_type>
							super;
    
  public:
    typedef typename super::difference_type		difference_type;
    typedef typename super::value_type			value_type;
    typedef typename super::reference			reference;
    typedef typename super::pointer			pointer;

  public:
			box_filter_iterator(Iterator i, size_t w=0)
			    :_head(i), _tail(_head), _valid(true), _val()
			{
			    if (w > 0)
			    {
				_val = *_tail;
				
				while (--w > 0)
				    _val += *++_tail;
			    }
			}

    reference		operator *() const
			{
			    if (!_valid)
			    {
				_val += *_tail;
				_valid = true;
			    }
			    return _val;
			}
    
    const pointer	operator ->() const
			{
			    return &operator *();
			}
    
    box_filter_iterator&
			operator ++()
			{
			    _val -= *_head;
			    if (!_valid)
				_val += *_tail;
			    else
				_valid = false;
			    ++_head;
			    ++_tail;
			    return *this;
			}
    
    box_filter_iterator	operator ++(int)
			{
			    box_filter_iterator	tmp = *this;
			    operator ++();
			    return tmp;
			}
    
    bool		operator ==(const box_filter_iterator& a) const
			{
			    return _head == a._head;
			}
    
    bool		operator !=(const box_filter_iterator& a) const
			{
			    return !operator ==(a);
			}

  private:
    Iterator		_head;
    Iterator		_tail;
    mutable bool	_valid;	//!< _val が [_head, _tail] の総和ならtrue
    mutable value_type	_val;	//!< [_head, _tail) または [_head, _tail] の総和
};

}
#endif
