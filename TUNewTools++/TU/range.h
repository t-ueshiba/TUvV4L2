/*
 *  $Id$
 */
/*!
  \file		range.h
  \brief	2つの反復子によって指定されるレンジの定義と実装
*/
#ifndef __TU_RANGE_H
#define __TU_RANGE_H

#include <iostream>
#include <cstddef>	// for size_t
#include <cassert>
#include <algorithm>
#include <initializer_list>
#include <type_traits>
#include <boost/iterator/iterator_adaptor.hpp>

namespace std
{
#if __cplusplus < 201700L
/************************************************************************
*  function std::size(T)						*
************************************************************************/
template <class T> inline size_t
size(const T& x)
{
    return x.size();
}
template <class T, size_t N> inline constexpr size_t
size(const T (&array)[N]) noexcept
{
    return N;
}
#endif

#if __cplusplus <= 201103L
/************************************************************************
*  function std::make_reverse_iterator(ITER)				*
************************************************************************/
template <class ITER> inline std::reverse_iterator<ITER>
make_reverse_iterator(ITER iter)
{
    return std::reverse_iterator<ITER>(iter);
}
#endif
}	// namespace std

namespace TU
{
namespace detail
{
/************************************************************************
*  predicate is_range<E>						*
************************************************************************/
  struct is_range
  {
      template <class E> static auto
      check(const E& x) -> decltype(x.begin(), x.end(),
				    std::true_type())			;
      static std::false_type
      check(...)							;
  };
}	// namespace detail

template <class E>
using is_range = decltype(detail::is_range::check(std::declval<E>()));

/************************************************************************
*  class range<ITER, SIZE>						*
************************************************************************/
//! 2つの反復子によって指定される範囲(レンジ)を表すクラス
/*!
  \param ITER	反復子の型
  \param SIZE	レンジに含まれる要素数(0ならば可変長)
*/
template <class ITER, size_t SIZE=0>	class range;

//! 固定長レンジ
/*!
  \param ITER	反復子の型
  \param SIZE	レンジに含まれる要素数
*/
template <class ITER, size_t SIZE>
class range
{
  public:
    using iterator		 = ITER;
    using const_iterator	 = iterator;
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
					       ::value_type;
    using reference		 = typename std::iterator_traits<iterator>
					       ::reference;
    using const_reference	 = const reference&;

  public:
		range(iterator begin)	:_begin(begin)	{}
    
		range()					= delete;
		range(const range&)			= default;
    range&	operator =(const range& r)
		{
		    std::copy_n(r._begin, SIZE, _begin);
		    return *this;
		}
		range(range&&)				= default;
    range&	operator =(range&&)			= default;
    
    template <class E_>
    typename std::enable_if<is_range<E_>::value, range&>::type
		operator =(const E_& expr)
		{
		    assert(std::size(expr) == SIZE);
		    std::copy_n(std::begin(expr), SIZE, _begin);
		    return *this;
		}

		range(std::initializer_list<value_type> args)
		    :_begin(args.begin())
    		{
		    assert(args.size() == SIZE);
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == SIZE);
		    std::copy(args.begin(), args.end(), begin());
		    return *this;
		}
		
    constexpr static
    size_t	size()	  	{ return SIZE; }
    auto	begin()	  	{ return _begin; }
    auto	end()	  	{ return _begin + SIZE; }
    auto	begin()	  const	{ return _begin; }
    auto	end()	  const	{ return _begin + SIZE; }
    auto	cbegin()  const	{ return begin(); }
    auto	cend()    const	{ return end(); }
    auto	rbegin()  	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  	{ return std::make_reverse_iterator(begin()); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	crend()	  const	{ return rend(); }
    reference	operator [](size_t i) 
		{
		    assert(i < size());
		    return *(_begin + i);
		}
    const auto&	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
		}
    template <size_t I_>
    auto	begin() const
		{
		    return begin(std::integral_constant<size_t, I_>());
		}

  private:
    const auto*	begin(std::integral_constant<size_t, 0>) const
		{
		    return this;
		}
    template <size_t I_>
    auto	begin(std::integral_constant<size_t, I_>) const
		{
		    return begin(std::integral_constant<size_t, I_-1>())
			 ->begin();
		}
    
  private:
    const iterator	_begin;
};

//! 可変長レンジ
/*!
  \param ITER	反復子の型
*/
template <class ITER>
class range<ITER, 0>
{
  public:
    using iterator		 = ITER;
    using const_iterator	 = iterator;
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
					       ::value_type;
    using reference		 = typename std::iterator_traits<iterator>
					       ::reference;
    using const_reference	 = const reference&;

  public:
		range(iterator begin, iterator end)
		    :_begin(begin), _end(end)		{}
    
		range()					= delete;
		range(const range&)			= default;
    range&	operator =(const range& r)
		{
		    assert(r.size() == size());
		    std::copy(r._begin, r._end, _begin);
		    return *this;
		}
		range(range&&)				= default;
    range&	operator =(range&&)			= default;
    
    template <class E_>
    typename std::enable_if<is_range<E_>::value, range&>::type
		operator =(const E_& expr)
		{
		    assert(std::size(expr) == size());
		    std::copy(std::begin(expr), std::end(expr), _begin);
		    return *this;
		}
		
		range(std::initializer_list<value_type> args)
		    :_begin(args.begin()), _end(args.end())
    		{
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == size());
		    std::copy(args.begin(), args.end(), begin());
		    return *this;
		}
		
    size_t	size()	  const	{ return std::distance(_begin, _end); }
    auto	begin()	  	{ return _begin; }
    auto	end()	  	{ return _end; }
    auto	begin()	  const	{ return _begin; }
    auto	end()	  const	{ return _end; }
    auto	cbegin()  const	{ return begin(); }
    auto	cend()    const	{ return end(); }
    auto	rbegin()  	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  	{ return std::make_reverse_iterator(begin()); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crbegin() const	{ return std::make_reverse_iterator(end()); }
    auto	crend()	  const	{ return std::make_reverse_iterator(begin()); }
    reference	operator [](size_t i) 
		{
		    assert(i < size());
		    return *(_begin + i);
		}
    const auto&	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
		}
    const auto*	begin(std::integral_constant<size_t, 0>) const
		{
		    return this;
		}
    template <size_t I_>
    auto	begin(std::integral_constant<size_t, I_>) const
		{
		    return begin(std::integral_constant<size_t, I_-1>())
			 ->begin();
		}
    
  private:
    const iterator	_begin;
    const iterator	_end;
};

//! 固定長レンジを生成する
/*!
  \param SIZE	レンジ長
  \param iter	レンジの先頭要素を指す反復子
*/
template <size_t SIZE, class ITER> inline range<ITER, SIZE>
make_range(ITER iter)
{
    return {iter};
}
    
//! 可変長レンジを生成する
/*!
  \param begin	レンジの先頭要素を指す反復子
  \param end	レンジの末尾要素の次を指す反復子
*/
template <class ITER> inline range<ITER>
make_range(ITER begin, ITER end)
{
    return {begin, end};
}

/************************************************************************
*  class range_iterator<ITER, SIZE, STRIDE>				*
************************************************************************/
namespace detail
{
  template <size_t SIZE, size_t STRIDE>
  struct size_and_stride
  {
      constexpr static size_t	size()		{ return SIZE; }
      constexpr static size_t	stride()	{ return STRIDE; }
  };
  template <size_t SIZE>
  struct size_and_stride<SIZE, 0>
  {
      size_and_stride(size_t stride)
	  :_stride(stride)			{}
      constexpr static size_t	size()		{ return SIZE; }
      size_t			stride() const	{ return _stride; }

    private:
      const size_t	_stride;
  };
  template <size_t STRIDE>
  struct size_and_stride<0, STRIDE>
  {
      size_and_stride(size_t size)
	  :_size(size)				{}
      size_t			size()	 const	{ return _size; }
      constexpr static size_t	stride()	{ return STRIDE; }

    private:
      const size_t	_size;
  };
  template <>
  struct size_and_stride<0, 0>
  {
      size_and_stride(size_t size, size_t stride)
	  :_size(size), _stride(stride)		{}
      size_t			size()	 const	{ return _size; }
      size_t			stride() const	{ return _stride; }

    private:
      const size_t	_size;
      const size_t	_stride;
  };
}	// namespace detail
    
//! 配列を一定間隔に切り分けたレンジを指す反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param SIZE	レンジ長(0ならば可変長)
  \param STRIDE	インクリメントしたときに進める要素数(0ならば可変)
*/
template <class ITER, size_t SIZE=0, size_t STRIDE=0>
class range_iterator
    : public boost::iterator_adaptor<range_iterator<ITER, SIZE, STRIDE>,
						    ITER,
						    range<ITER, SIZE>,
						    boost::use_default,
						    range<ITER, SIZE> >,
      public detail::size_and_stride<SIZE, STRIDE>
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER, SIZE>,
					  boost::use_default,
					  range<ITER, SIZE> >;
    using ss	= detail::size_and_stride<SIZE, STRIDE>;
    
  public:
    using reference	  = typename super::reference;
    using difference_type = typename super::difference_type;

    friend class	boost::iterator_core_access;
	  
  public:
    template <size_t SIZE_=SIZE, size_t STRIDE_=STRIDE,
	      typename std::enable_if<
		  (SIZE_ != 0) && (STRIDE_ != 0)>::type* = nullptr>
	range_iterator(ITER iter)
	    :super(iter), ss()						{}
    template <size_t SIZE_=SIZE, size_t STRIDE_=STRIDE,
	      typename std::enable_if<
		  (SIZE_ != 0) && (STRIDE_ == 0)>::type* = nullptr>
	range_iterator(ITER iter, size_t stride)
	    :super(iter), ss(stride)					{}
    template <size_t SIZE_=SIZE, size_t STRIDE_=STRIDE,
	      typename std::enable_if<
		  (SIZE_ == 0) && (STRIDE_ != 0)>::type* = nullptr>
	range_iterator(ITER iter, size_t size)
	    :super(iter), ss(size)					{}
    template <size_t SIZE_=SIZE, size_t STRIDE_=STRIDE,
	      typename std::enable_if<
		  (SIZE_ == 0) && (STRIDE_ == 0)>::type* = nullptr>
	range_iterator(ITER iter, size_t size, size_t stride)
	    :super(iter), ss(size, stride)				{}

    using	ss::size;
    using	ss::stride;
    
  private:
    template <size_t SIZE_=SIZE>
    typename std::enable_if<SIZE_ == 0, reference>::type
		dereference() const
		{
		    return {super::base(), super::base() + size()};
		}
    template <size_t SIZE_=SIZE>
    typename std::enable_if<SIZE_ != 0, reference>::type
		dereference() const
		{
		    return {super::base()};
		}
    void	increment()
		{
		    std::advance(super::base_reference(), stride());
		}
    void	decrement()
		{
		    std::advance(super::base_reference(), -stride());
		}
    void	advance(difference_type n)
		{
		    std::advance(super::base_reference(), n*stride());
		}
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return std::distance(super::base(), iter.base()) / stride();
		}
};

/************************************************************************
*  fixed size & fixed stride ranges and associated iterators		*
************************************************************************/
//! 固定長レンジを指し，インクリメント時に固定した要素数だけ進める反復子を生成する
/*!
  \param SIZE	レンジ長
  \param STRIDE	インクリメント時に進める要素数
  \param iter	レンジの先頭要素を指す反復子
*/
template <size_t SIZE, size_t STRIDE, class ITER>
inline range_iterator<ITER, SIZE, STRIDE>
make_range_iterator(ITER iter)
{
    return {iter};
}

//! 多次元固定長レンジを指し，インクリメント時に固定したブロック数だけ進める反復子を生成する
/*!
  \param SIZE	最上位次元のレンジ長
  \param STRIDE	インクリメント時に進める最上位次元のブロック数
  \param SS	2番目以降の次元の(レンジ長，ストライド)の並び
  \param iter	レンジの先頭要素を指す反復子
*/
template <size_t SIZE, size_t STRIDE, size_t... SS, class ITER,
	  typename std::enable_if<sizeof...(SS) != 0>::type* = nullptr>
inline auto
make_range_iterator(ITER iter)
{
    return make_range_iterator<SIZE, STRIDE>(make_range_iterator<SS...>(iter));
}

template <size_t SIZE, size_t... SS, class ITER,
	  typename std::enable_if<sizeof...(SS) != 0>::type* = nullptr>
inline auto
make_range(ITER iter)
{
    return make_range<SIZE>(make_range_iterator<SS...>(iter));
}

/************************************************************************
*  fixed size & variable stride ranges and associated iterators		*
************************************************************************/
//! 固定長レンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param SIZE	レンジ長
  \param iter	レンジの先頭要素を指す反復子
  \param stride	インクリメント時に進める要素数
*/
template <size_t SIZE, class ITER> inline range_iterator<ITER, SIZE>
make_range_iterator(ITER iter, size_t stride)
{
    return {iter, stride};
}
    
//! 多次元固定長レンジを指し，インクリメント時に指定したブロック数だけ進める反復子を生成する
/*!
  \param SIZE		最上位次元のレンジ長
  \param SIZES		2番目以降の次元のレンジ長の並び
  \param stride		最上位次元のストライド
  \param strides	2番目以降の次元のストライドの並び
  \param iter		レンジの先頭要素を指す反復子
*/
template <size_t SIZE, size_t... SIZES, class ITER, class... STRIDES,
	  typename std::enable_if<
	      sizeof...(SIZES) == sizeof...(STRIDES)>::type* = nullptr>
inline auto
make_range_iterator(ITER iter, size_t stride, STRIDES... strides)
{
    return make_range_iterator<SIZE>(
	       make_range_iterator<SIZES...>(iter, strides...), stride);
}

template <size_t SIZE, size_t... SIZES, class ITER, class... STRIDES,
	  typename std::enable_if<
	      sizeof...(SIZES) == sizeof...(STRIDES)>::type* = nullptr>
inline auto
make_range(ITER iter, STRIDES... strides)
{
    return make_range<SIZE>(make_range_iterator<SIZES...>(iter, strides...));
}

/************************************************************************
*  variable size & variable stride ranges and associated iterators	*
************************************************************************/
//! 指定された長さのレンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param iter	レンジの先頭要素を指す反復子
  \param size	レンジ長
  \param stride	インクリメント時に進める要素数
*/
template <class ITER> inline range_iterator<ITER>
make_range_iterator(ITER iter, size_t size, size_t stride)
{
    return {iter, size, stride};
}
    
//! 多次元固定長レンジを指し，インクリメント時に指定したブロック数だけ進める反復子を生成する
/*!
  \param iter		レンジの先頭要素を指す反復子
  \param size		最上位次元のレンジ長
  \param stride		最上位次元のストライド
  \param ss		2番目以降の次元の(レンジ長, ストライド)の並び
*/
template <class ITER, class... SS> inline auto
make_range_iterator(ITER iter, size_t size, size_t stride, SS... ss)
{
    return make_range_iterator(make_range_iterator(iter, ss...),
			       size, stride);
}

template <class ITER> inline range<ITER>
make_range(ITER iter, size_t size)
{
    return {iter, iter + size};
}

template <class ITER, class... SS> inline auto
make_range(ITER iter, size_t size, SS... ss)
{
    return make_range(make_range_iterator(iter, ss...), size);
}

/************************************************************************
*  ranges with variable but identical size and stride			*
*  and associated iterators						*
************************************************************************/
template <class ITER> inline range_iterator<ITER>
make_dense_range_iterator(ITER iter, size_t size)
{
    return {iter, size, size};
}
    
template <class ITER, class... SIZES> inline auto
make_dense_range_iterator(ITER iter, size_t size, SIZES... sizes)
{
    return make_dense_range_iterator(make_dense_range_iterator(iter, sizes...),
				     size);
}
    
template <class ITER> inline range<ITER>
make_dense_range(ITER iter, size_t size)
{
    return {iter, iter + size};
}

template <class ITER, class... SIZES> inline auto
make_dense_range(ITER iter, size_t size, SIZES... sizes)
{
    return make_dense_range(make_dense_range_iterator(iter, sizes...), size);
}

/************************************************************************
*  sizes and strides of multidimensional ranges				*
************************************************************************/
template <size_t I=0, class ITER, size_t SIZE> inline size_t
size(const range<ITER, SIZE>& r)
{
    return r.begin(std::integral_constant<size_t, I>())->size();
}
/*
template <size_t I, class ITER, size_t SIZE> inline size_t
stride(const range<ITER, SIZE>& r)
{
    return r.begin(std::integral_constant<size_t, I>()).stride();
}
*/
/************************************************************************
*  subrange extraction							*
************************************************************************/
template <class ITER> inline ITER
make_subrange_iterator(ITER iter)
{
    return iter;
}

template <class ITER, class... IS> inline auto
make_subrange_iterator(const range_iterator<ITER>& iter,
		       size_t idx, size_t size, IS... is)
{
    return make_range_iterator(make_subrange_iterator(
				   iter->begin() + idx, is...),
			       size, iter.stride());
}

template <class RANGE, class... IS> inline auto
make_subrange(const RANGE& r, size_t idx, size_t size, IS... is)
{
    return make_range(make_subrange_iterator(r.begin() + idx, is...), size);
}

template <size_t SIZE, size_t... SIZES, class ITER, class... INDICES,
	  typename std::enable_if<
	      sizeof...(SIZES) == sizeof...(INDICES)>::type* = nullptr>
inline auto
make_subrange_iterator(const ITER& iter, size_t idx, INDICES... indices)
{
    return make_range_iterator<SIZE>(make_subrange_iterator<SIZES...>(
					 iter->begin() + idx, indices...),
				     iter.stride());
}
    
template <size_t SIZE, size_t... SIZES, class RANGE, class... INDICES,
	  typename std::enable_if<
	      sizeof...(SIZES) == sizeof...(INDICES)>::type* = nullptr>
inline auto
make_subrange(const RANGE& r, size_t idx, INDICES... indices)
{
    return make_range<SIZE>(make_subrange_iterator<SIZES...>(
				r.begin() + idx, indices...));
}

/************************************************************************
*  generic algorithms for ranges					*
************************************************************************/
template <class E>
typename std::enable_if<is_range<E>::value, std::ostream&>::type
operator <<(std::ostream& out, const E& expr)
{
    for (const auto& elm : expr)
	out << ' ' << elm;
    return out << std::endl;
}
    
template <class E, class T>
typename std::enable_if<!is_range<typename std::decay<E>::type>::value>::type
fill(E&& expr, const T& val)
{
    expr = val;
}
template <class E, class T>
typename std::enable_if<is_range<typename std::decay<E>::type>::value>::type
fill(E&& expr, const T& val)
{
    for (auto iter = std::begin(expr); iter != std::end(expr); ++iter)
	fill(*iter, val);
}
    
}	// namespace TU
#endif	// !__TU_RANGE_H
