/*
 *  $Id$
 */
/*!
  \file		iterator.h
  \brief	各種反復子の定義と実装
*/
#include <iostream>
#include <cstddef>	// for size_t
#include <cassert>
#include <algorithm>
#include <type_traits>
#include <boost/iterator/iterator_adaptor.hpp>

namespace std
{
#if __cplusplus < 201700L
/************************************************************************
*  function std::size(E)						*
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

template <class E>
typename std::enable_if<is_range<E>::value, std::ostream&>::type
operator <<(std::ostream& out, const E& expr)
{
    for (const auto& elm : expr)
	out << ' ' << elm;
    return out << std::endl;
}
    
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
		
    static constexpr
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
//! 配列を一定間隔に切り分けたレンジを指す反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param SIZE	レンジ長(0ならば可変長)
  \param STRIDE	インクリメントしたときに進める要素数(0ならば可変)
*/
template <class ITER, size_t SIZE=0, size_t STRIDE=0>    class range_iterator;
    
//! 固定長のレンジを指し，インクリメント時に固定した要素数だけ進める反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param SIZE	レンジ長
  \param STRIDE	インクリメントしたときに進める要素数
*/
template <class ITER, size_t SIZE, size_t STRIDE>
class range_iterator
    : public boost::iterator_adaptor<range_iterator<ITER, SIZE, STRIDE>,
						    ITER,
						    range<ITER, SIZE>,
						    boost::use_default,
						    range<ITER, SIZE> >
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER, SIZE>,
					  boost::use_default,
					  range<ITER, SIZE> >;

  public:
    using reference	  = typename super::reference;
    using difference_type = typename super::difference_type;

    friend class	boost::iterator_core_access;
	  
  public:
		range_iterator()			= default;
		range_iterator(ITER iter) :super(iter)	{}

    static constexpr size_t	size()			{ return SIZE; }
    static constexpr size_t	stride()		{ return STRIDE; }
	      
  private:
    reference	dereference() const
		{
		    return {super::base()};
		}
    void	increment()
		{
		    std::advance(super::base_reference(), STRIDE);
		}
    void	decrement()
		{
		    std::advance(super::base_reference(), -STRIDE);
		}
    void	advance(difference_type n)
		{
		    std::advance(super::base_reference(), n*STRIDE);
		}
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return std::distance(super::base(), iter.base()) / STRIDE;
		}
};

//! 固定長のレンジを指し，インクリメント時に指定した要素数(可変)だけ進める反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param SIZE	レンジ長
*/
template <class ITER, size_t SIZE>
class range_iterator<ITER, SIZE, 0>
    : public boost::iterator_adaptor<range_iterator<ITER, SIZE, 0>,
						    ITER,
						    range<ITER, SIZE>,
						    boost::use_default,
						    range<ITER, SIZE> >
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER, SIZE>,
					  boost::use_default,
					  range<ITER, SIZE> >;

  public:
    using reference	  = typename super::reference;
    using difference_type = typename super::difference_type;

    friend class	boost::iterator_core_access;
	  
  public:
		range_iterator() :super(), _stride(0)	{}
		range_iterator(ITER iter, size_t stride)
		    :super(iter), _stride(stride)	{}

    static constexpr size_t	size()			{ return SIZE; }
    size_t			stride()	const	{ return _stride; }
	      
  private:
    reference	dereference() const
		{
		    return {super::base()};
		}
    void	increment()
		{
		    std::advance(super::base_reference(), _stride);
		}
    void	decrement()
		{
		    std::advance(super::base_reference(), -_stride);
		}
    void	advance(difference_type n)
		{
		    std::advance(super::base_reference(), n*_stride);
		}
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return std::distance(super::base(), iter.base()) / _stride;
		}

  private:
    size_t	_stride;
};

//! 可変長のレンジを指し，インクリメント時に指定した要素数(可変)だけ進める反復子
/*!
  \param ITER	配列の要素を指す反復子の型
*/
template <class ITER>
class range_iterator<ITER, 0, 0>
    : public boost::iterator_adaptor<range_iterator<ITER, 0, 0>,
						    ITER,
						    range<ITER>,
						    boost::use_default,
						    range<ITER> >
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER>,
					  boost::use_default,
					  range<ITER> >;

  public:
    using reference	  = typename super::reference;
    using difference_type = typename super::difference_type;

    friend class	boost::iterator_core_access;
	  
  public:
    range_iterator()	:super(), _size(0), _stride(0)			{}
    range_iterator(ITER iter, size_t size, size_t stride)
	:super(iter), _size(size), _stride(stride)			{}

    size_t	size()			const	{ return _size; }
    size_t	stride()		const	{ return _stride; }
	      
  private:
    reference	dereference() const
		{
		    return {super::base(), super::base() + _size};
		}
    void	increment()
		{
		    std::advance(super::base_reference(), _stride);
		}
    void	decrement()
		{
		    std::advance(super::base_reference(), -_stride);
		}
    void	advance(difference_type n)
		{
		    std::advance(super::base_reference(), n*_stride);
		}
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return std::distance(super::base(), iter.base()) / _stride;
		}
    bool	equal(const range_iterator& iter) const
		{
		    return (super::base() == iter.base() &&
			    _size	  == iter._size  &&
			    _stride	  == iter._stride);
		}

  private:
    size_t	_size;
    size_t	_stride;
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
  \param args		2番目以降の次元の(レンジ長, ストライド)の並び
*/
template <class ITER, class... ARGS> inline auto
make_range_iterator(ITER iter, size_t size, size_t stride, ARGS... args)
{
    return make_range_iterator(make_range_iterator(iter, args...),
			       size, stride);
}

template <class ITER> inline range<ITER>
make_range(ITER iter, size_t size)
{
    return {iter, iter + size};
}

template <class ITER, class... ARGS> inline auto
make_range(ITER iter, size_t size, ARGS... args)
{
    return make_range(make_range_iterator(iter, args...), size);
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
template <size_t I=0, class ITER> inline size_t
size(const range<ITER>& r)
{
    return r.begin(std::integral_constant<size_t, I>())->size();
}

template <size_t I, class ITER> inline size_t
stride(const range<ITER>& r)
{
    return r.begin(std::integral_constant<size_t, I>()).stride();
}

/************************************************************************
*  subrange extraction							*
************************************************************************/
template <class RANGE> inline auto
subrange(const RANGE& r, size_t idx, size_t size)
{
    return make_range(r.begin() + idx, size);
}

template <class RANGE, class... ARGS> inline auto
subrange(const RANGE& r, size_t idx, size_t size, ARGS... args)
{
    return subrange(make_range(r.begin() + idx, size), args...);
}

}	// namespace TU
	
