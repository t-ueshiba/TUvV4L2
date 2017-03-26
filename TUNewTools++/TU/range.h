/*!
  \file		range.h
  \brief	2つの反復子によって指定されるレンジの定義と実装
*/
#ifndef __TU_RANGE_H
#define __TU_RANGE_H

#include <cassert>
#include <initializer_list>
#include "TU/algorithm.h"	// for copy<N>(IN, ARG, OUT), etc...
#include "TU/tuple.h"		// required before defining iterator_t<E>

namespace TU
{
/************************************************************************
*  rank<E>(), size0<E>(), size<E>() and stride<E>()			*
************************************************************************/
//! 式の次元数(軸の個数)を返す
/*!
  \param E	式の型
  \return	式の次元数
 */
template <class E>
constexpr std::enable_if_t<!has_begin<E>::value, size_t>
rank()
{
    return 0;
}
template <class E>
constexpr std::enable_if_t<has_begin<E>::value, size_t>
rank()
{
    return 1 + rank<value_t<E> >();
}
    
namespace detail
{
  template <class E>
  auto		  has_size0(E) -> decltype(E::size0(), std::true_type());
  std::false_type has_size0(...)					;

  /*
   *  A ^ b において演算子ノード product_opnode<bit_xor, L, R> が
   *  生成されるが，これを評価して2次元配列に代入する際に代入先の領域確保のため
   *  size<0>(opnode), size<1>(opnode) が呼び出される．後者はopnodeの反復子が
   *  指す先を評価するが，これは2つの3次元ベクトルのベクトル積を評価することに
   *  等しく，コストが高い処理である．そこで，ベクトル積を評価せずその評価結果の
   *  サイズだけを得るために，以下のオーバーロードを導入する．
   */
  template <class OP, class L, class R>	class product_opnode;
  struct bit_xor;
    
  template <class L, class R> constexpr size_t
  size(const product_opnode<bit_xor, L, R>&, std::integral_constant<size_t, 1>)
  {
      return 3;
  }

  template <class E> inline size_t
  size(const E& expr, std::integral_constant<size_t, 0>)
  {
      return std::size(expr);
  }
  template <size_t I, class E> inline size_t
  size(const E& expr, std::integral_constant<size_t, I>)
  {
      return size(*std::begin(expr), std::integral_constant<size_t, I-1>());
  }

  template <class E> inline size_t
  stride(const E& expr, std::integral_constant<size_t, 1>)
  {
      return std::begin(expr).stride();
  }
  template <size_t I, class E> inline size_t
  stride(const E& expr, std::integral_constant<size_t, I>)
  {
      return stride(*std::begin(expr), std::integral_constant<size_t, I-1>());
  }
}	// namespace detail

template <class E>
using has_size0 = decltype(detail::has_size0(std::declval<E>()));
    
//! 式の最上位軸の要素数を返す
/*!
  本関数で返されるのは静的なサイズであり，式が可変個の要素を持つ場合，0が返される
  \param E	式の型
  \return	最上位軸の静的な要素数(可変サイズの場合0)
 */
template <class E> constexpr std::enable_if_t<!has_size0<E>::value, size_t>
size0()
{
  // Array2<T, R, C> * Array<T, 0, 0> の評価結果の型が Array<T, R, 0>
  // ではなく Array<T, 0, 0> になるために，0ではなく1を返すことが必要．
    return 1;
}
template <class E> constexpr std::enable_if_t<has_size0<E>::value, size_t>
size0()
{
    return E::size0();
}
    
//! 与えられた式について，指定された軸の要素数を返す
/*!
  \param I	軸を指定するindex
  \param E	式の型
  \return	軸Iの要素数
 */
template <size_t I, class E> inline std::enable_if_t<rank<E>() != 0, size_t>
size(const E& expr)
{
    return detail::size(expr, std::integral_constant<size_t, I>());
}

//! 与えられた式について，指定された軸のストライドを返す
/*!
  \param I	軸を指定するindex
  \param E	式の型
  \return	軸Iのストライド
 */
template <size_t I, class E> inline std::enable_if_t<rank<E>() != 0, size_t>
stride(const E& expr)
{
    return detail::stride(expr, std::integral_constant<size_t, I>());
}

/************************************************************************
*  manipulator print_sizes() and print_sizes_and_strides()		*
************************************************************************/
template <class E>
class sizes_holder
{
  public:
			sizes_holder(const E& expr)	:_expr(expr)	{}

    std::ostream&	operator ()(std::ostream& out) const
			{
			    return print_size(out, &_expr);
			}
    
  protected:
    template <class ITER_> constexpr
    static size_t	rank()
			{
			    using value_type = iterator_value<ITER_>;
			    return TU::rank<value_type>();
			}
    template <class ITER_>
    static std::enable_if_t<rank<ITER_>() == 0, std::ostream&>
			print_x(std::ostream& out, ITER_)
			{
			    return out;
			}
    template <class ITER_>
    static std::enable_if_t<rank<ITER_>() != 0, std::ostream&>
			print_x(std::ostream& out, ITER_)
			{
			    return out << 'x';
			}

  private:
    template <class ITER_>
    static std::enable_if_t<rank<ITER_>() == 0, std::ostream&>
			print_size(std::ostream& out, ITER_)
			{
			    return out;
			}
    template <class ITER_>
    static std::enable_if_t<rank<ITER_>() != 0, std::ostream&>
			print_size(std::ostream& out, ITER_ iter)
			{
			    const auto&	val = *iter;
			    return print_size(print_x(out << std::size(val),
						      std::begin(val)),
					      std::begin(val));
			}

  protected:
    const E&	_expr;
};
    
template <class E>
class sizes_and_strides_holder : public sizes_holder<E>
{
  private:
    using super	= sizes_holder<E>;
    
  public:
			sizes_and_strides_holder(const E& expr)
			    :super(expr)				{}

    std::ostream&	operator ()(std::ostream& out) const
			{
			    return print_stride(super::operator ()(out) << ':',
						std::begin(super::_expr));
			}
    
  private:
    template <class ITER_>
    static std::enable_if_t<super::template rank<ITER_>() == 0, std::ostream&>
			print_stride(std::ostream& out, ITER_)
			{
			    return out;
			}
    template <class ITER_>
    static std::enable_if_t<super::template rank<ITER_>() != 0, std::ostream&>
			print_stride(std::ostream& out, ITER_ iter)
			{
			    const auto&	val = *iter;
			    return print_stride(super::print_x(
						    out << iter.stride(),
						    std::begin(val)),
						std::begin(val));
			}
};

template <class E> sizes_holder<E>
print_sizes(const E& expr)
{
    return sizes_holder<E>(expr);
}

template <class E> std::ostream&
operator <<(std::ostream& out, const sizes_holder<E>& holder)
{
    return holder(out);
}

template <class E> sizes_and_strides_holder<E>
print_sizes_and_strides(const E& expr)
{
    return sizes_and_strides_holder<E>(expr);
}

template <class E> std::ostream&
operator <<(std::ostream& out, const sizes_and_strides_holder<E>& holder)
{
    return holder(out);
}

/************************************************************************
*  class range<ITER, SIZE>						*
************************************************************************/
//! 先頭要素を指す反復子と要素数によって指定される範囲(レンジ)を表すクラス
/*!
  \param ITER	反復子の型
  \param SIZE	レンジに含まれる要素数(0ならば可変長)
*/
template <class ITER, size_t SIZE=0>
class range
{
  public:
    using iterator		= ITER;
    using reverse_iterator	= std::reverse_iterator<iterator>;
    using value_type		= iterator_value<iterator>;
    using reference		= iterator_reference<iterator>;
    using element_type		= element_t<value_type>;
    
  public:
		range(iterator begin)		:_begin(begin)	{}
		range(iterator begin, size_t)	:_begin(begin)	{}
    
		range()						= delete;
		range(const range&)				= default;
    range&	operator =(const range& r)
		{
		    copy<SIZE>(r.begin(), size(), begin());
		    return *this;
		}
		range(range&&)					= default;
    range&	operator =(range&&)				= default;
    
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, iterator>::value>*
	      = nullptr>
		range(const range<ITER_, SIZE>& r)
		    :_begin(r.begin())
		{
		}

    template <class E_> std::enable_if_t<rank<E_>() != 0, range&>
		operator =(const E_& expr)
		{
		    assert(std::size(expr) == size());
		    copy<SIZE>(std::begin(expr), size(), begin());
		    return *this;
		}

		range(std::initializer_list<value_type> args)
		    :_begin(const_cast<iterator>(args.begin()))
    		{
		    assert(args.size() == size());
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == size());
		    copy<SIZE>(args.begin(), size(), begin());
		    return *this;
		}

    template <class T_> std::enable_if_t<rank<T_>() == 0, range&>
		operator =(const T_& c)
		{
		    fill<SIZE>(begin(), size(), c);
		    return *this;
		}

    constexpr static
    size_t	size0()			{ return SIZE; }
    constexpr static
    size_t	size()			{ return SIZE; }
    auto	begin()		const	{ return _begin; }
    auto	end()		const	{ return _begin + SIZE; }
    auto	cbegin()	const	{ return begin(); }
    auto	cend()		const	{ return end(); }
    auto	rbegin()	const	{ return reverse_iterator(end()); }
    auto	rend()		const	{ return reverse_iterator(begin()); }
    auto	crbegin()	const	{ return rbegin(); }
    auto	crend()		const	{ return rend(); }
    reference	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
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
    using iterator		= ITER;
    using reverse_iterator	= std::reverse_iterator<iterator>;
    using value_type		= iterator_value<iterator>;
    using reference		= iterator_reference<iterator>;
    using element_type		= element_t<value_type>;
    
  public:
		range(iterator begin, size_t size)
		    :_begin(begin), _size(size)			{}
    
		range()						= delete;
		range(const range&)				= default;
    range&	operator =(const range& r)
		{
		    assert(r.size() == size());
		    copy<0>(r._begin, size(), begin());
		    return *this;
		}
		range(range&&)					= default;
    range&	operator =(range&&)				= default;
    
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, iterator>::value>*
	      = nullptr>
		range(const range<ITER_, 0>& r)
		    :_begin(r.begin()), _size(r.size())
		{
		}

    template <class E_> std::enable_if_t<rank<E_>() != 0, range&>
		operator =(const E_& expr)
		{
		    assert(std::size(expr) == size());
		    copy<TU::size0<E_>()>(std::begin(expr), size(), begin());
		    return *this;
		}
		
		range(std::initializer_list<value_type> args)
		    :_begin(const_cast<iterator>(args.begin())),
		     _size(args.size())
    		{
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == size());
		    copy<0>(args.begin(), size(), begin());
		    return *this;
		}
		
    template <class T_> std::enable_if_t<rank<T_>() == 0, range&>
		operator =(const T_& c)
		{
		    fill<0>(begin(), size(), c);
		    return *this;
		}

    constexpr static
    size_t	size0()			{ return 0; }
    size_t	size()		const	{ return _size; }
    auto	begin()		const	{ return _begin; }
    auto	end()		const	{ return _begin + _size; }
    auto	cbegin()	const	{ return begin(); }
    auto	cend()		const	{ return end(); }
    auto	rbegin()	const	{ return reverse_iterator(end()); }
    auto	rend()		const	{ return reverse_iterator(begin()); }
    auto	crbegin()	const	{ return rbegin(); }
    auto	crend()		const	{ return rend(); }
    reference	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
		}

  private:
    const iterator	_begin;
    const size_t	_size;
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
  \param iter	レンジの先頭要素を指す反復子
  \param size	レンジの要素数
*/
template <size_t SIZE=0, class ITER> inline range<ITER, SIZE>
make_range(ITER iter, size_t size)
{
    return {iter, size};
}

//! 出力ストリームにレンジの内容を書き出す
/*!
  \param out	出力ストリーム
  \param r	レンジ
  \return	outで指定した出力ストリーム
*/
template <class ITER, size_t SIZE> std::ostream&
operator <<(std::ostream& out, const range<ITER, SIZE>& r)
{
    for (const auto& val : r)
	out << ' ' << val;
    return out << std::endl;
}
    
/************************************************************************
*  class range_iterator<ITER, STRIDE, SIZE>				*
************************************************************************/
namespace detail
{
  template <size_t STRIDE, size_t SIZE>
  struct stride_and_size
  {
      constexpr static size_t	stride()	{ return STRIDE; }
      constexpr static size_t	size()		{ return SIZE; }
  };
  template <size_t SIZE>
  struct stride_and_size<0, SIZE>
  {
      stride_and_size(size_t stride)
	  :_stride(stride)			{}
      size_t			stride() const	{ return _stride; }
      constexpr static size_t	size()		{ return SIZE; }

    private:
      size_t	_stride;
  };
  template <size_t STRIDE>
  struct stride_and_size<STRIDE, 0>
  {
      stride_and_size(size_t size)
	  :_size(size)				{}
      constexpr static size_t	stride()	{ return STRIDE; }
      size_t			size()	 const	{ return _size; }

    private:
      size_t	_size;
  };
  template <>
  struct stride_and_size<0, 0>
  {
      stride_and_size(size_t stride, size_t size)
	  :_stride(stride), _size(size)		{}
      size_t			stride() const	{ return _stride; }
      size_t			size()	 const	{ return _size; }

    private:
      size_t	_stride;
      size_t	_size;
  };
}	// namespace detail
    
//! 配列を一定間隔に切り分けたレンジを指す反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param STRIDE	インクリメントしたときに進める要素数(0ならば可変)
  \param SIZE	レンジ長(0ならば可変長)
*/
template <class ITER, size_t STRIDE, size_t SIZE>
class range_iterator
    : public boost::iterator_adaptor<range_iterator<ITER, STRIDE, SIZE>,
				     ITER,
				     range<ITER, SIZE>,
				     boost::use_default,
				     range<ITER, SIZE> >,
      public detail::stride_and_size<STRIDE, SIZE>
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER, SIZE>,
					  boost::use_default,
					  range<ITER, SIZE> >;
    using ss	= detail::stride_and_size<STRIDE, SIZE>;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;

    friend class	boost::iterator_core_access;
	  
  public:
    template <size_t STRIDE_=STRIDE, size_t SIZE_=SIZE,
	      std::enable_if_t<(STRIDE_ != 0) && (SIZE_ != 0)>* = nullptr>
		range_iterator(ITER iter)
		    :super(iter), ss()					{}
    template <size_t STRIDE_=STRIDE, size_t SIZE_=SIZE,
	      std::enable_if_t<(STRIDE_ == 0) && (SIZE_ != 0)>* = nullptr>
		range_iterator(ITER iter, size_t stride)
		    :super(iter), ss(stride)				{}
    template <size_t STRIDE_=STRIDE, size_t SIZE_=SIZE,
	      std::enable_if_t<(STRIDE_ != 0) && (SIZE_ == 0)>* = nullptr>
		range_iterator(ITER iter, size_t size)
		    :super(iter), ss(size)				{}
    template <size_t STRIDE_=STRIDE, size_t SIZE_=SIZE,
	      std::enable_if_t<(STRIDE_ == 0) && (SIZE_ == 0)>* = nullptr>
		range_iterator(ITER iter, size_t stride, size_t size)
		    :super(iter), ss(stride, size)			{}

    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		range_iterator(
		    const range_iterator<ITER_, SIZE, STRIDE>& iter)
		    :super(iter), ss(iter)				{}

    using	ss::size;
    using	ss::stride;
    
  private:
    reference	dereference() const
		{
		    return {super::base(), size()};
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

//! 固定長レンジを指し，インクリメント時に固定した要素数だけ進める反復子を生成する
/*!
  \param STRIDE	インクリメント時に進める要素数
  \param SIZE	レンジ長
  \param iter	レンジの先頭要素を指す反復子
*/
template <size_t STRIDE, size_t SIZE, class ITER>
inline range_iterator<ITER, STRIDE, SIZE>
make_range_iterator(ITER iter)
{
    return {iter};
}

//! 固定長レンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param SIZE	レンジ長
  \param iter	レンジの先頭要素を指す反復子
  \param stride	インクリメント時に進める要素数
*/
template <size_t SIZE, class ITER> inline range_iterator<ITER, 0, SIZE>
make_range_iterator(ITER iter, size_t stride)
{
    return {iter, stride};
}
    
//! 指定された長さのレンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param iter	レンジの先頭要素を指す反復子
  \param stride	インクリメント時に進める要素数
  \param size	レンジ長
*/
template <class ITER> inline range_iterator<ITER, 0, 0>
make_range_iterator(ITER iter, size_t stride, size_t size)
{
    return {iter, stride, size};
}
    
/************************************************************************
*  fixed size & fixed stride ranges and associated iterators		*
************************************************************************/
//! 多次元固定長レンジを指し，インクリメント時に固定したブロック数だけ進める反復子を生成する
/*!
  \param STRIDE	インクリメント時に進める最上位軸のブロック数
  \param SIZE	最上位軸のレンジ長
  \param SS	2番目以降の軸の{ストライド, レンジ長}の並び
  \param iter	レンジの先頭要素を指す反復子
*/
template <size_t STRIDE, size_t SIZE, size_t... SS, class ITER,
	  std::enable_if_t<sizeof...(SS) != 0>* = nullptr>
inline auto
make_range_iterator(ITER iter)
{
    return make_range_iterator<STRIDE, SIZE>(make_range_iterator<SS...>(iter));
}

template <size_t SIZE, size_t... SS, class ITER,
	  std::enable_if_t<sizeof...(SS) != 0>* = nullptr>
inline auto
make_range(ITER iter)
{
    return make_range<SIZE>(make_range_iterator<SS...>(iter));
}

/************************************************************************
*  fixed size & variable stride ranges and associated iterators		*
************************************************************************/
//! 多次元固定長レンジを指し，インクリメント時に指定したブロック数だけ進める反復子を生成する
/*!
  \param SIZE		最上位軸のレンジ長
  \param SIZES		2番目以降の軸のレンジ長の並び
  \param stride		最上位軸のストライド
  \param strides	2番目以降の軸のストライドの並び
  \param iter		レンジの先頭要素を指す反復子
*/
template <size_t SIZE, size_t... SIZES, class ITER, class... STRIDES,
	  std::enable_if_t<sizeof...(SIZES) == sizeof...(STRIDES)>* = nullptr>
inline auto
make_range_iterator(ITER iter, size_t stride, STRIDES... strides)
{
    return make_range_iterator<SIZE>(
	       make_range_iterator<SIZES...>(iter, strides...), stride);
}

template <size_t SIZE, size_t... SIZES, class ITER, class... STRIDES,
	  std::enable_if_t<sizeof...(SIZES) == sizeof...(STRIDES)>* = nullptr>
inline auto
make_range(ITER iter, STRIDES... strides)
{
    return make_range<SIZE>(make_range_iterator<SIZES...>(iter, strides...));
}

/************************************************************************
*  variable size & variable stride ranges and associated iterators	*
************************************************************************/
//! 多次元可変長レンジを指し，インクリメント時に指定したブロック数だけ進める反復子を生成する
/*!
  \param iter		レンジの先頭要素を指す反復子
  \param stride		最上位軸のストライド
  \param size		最上位軸のレンジ長
  \param ss		2番目以降の軸の{ストライド, レンジ長}の並び
*/
template <class ITER, class... SS> inline auto
make_range_iterator(ITER iter, size_t stride, size_t size, SS... ss)
{
    return make_range_iterator(make_range_iterator(iter, ss...),
			       stride, size);
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
namespace detail
{
  template <class ITER> inline ITER
  make_dense_range_iterator(ITER iter)
  {
      return iter;
  }
    
  template <class ITER, class... SIZES> inline auto
  make_dense_range_iterator(ITER iter, size_t size, SIZES... sizes)
  {
      return make_range_iterator(make_dense_range_iterator(iter, sizes...),
				 size, size);
  }
}	// namespace detail

template <class ITER, class... SIZES> inline auto
make_dense_range(ITER iter, size_t size, SIZES... sizes)
{
    return make_range(detail::make_dense_range_iterator(iter, sizes...), size);
}

/************************************************************************
*  slice extraction							*
************************************************************************/
namespace detail
{
  template <class ITER> inline ITER
  make_slice_iterator(ITER iter)
  {
      return iter;
  }

  template <class ITER, class... IS> inline auto
  make_slice_iterator(ITER iter, size_t idx, size_t size, IS... is)
  {
      return make_range_iterator(make_slice_iterator(
				     iter->begin() + idx, is...),
				 iter.stride(), size);
  }

  template <size_t SIZE, size_t... SIZES, class ITER, class... INDICES,
	    std::enable_if_t<sizeof...(SIZES) == sizeof...(INDICES)>* = nullptr>
  inline auto
  make_slice_iterator(ITER iter, size_t idx, INDICES... indices)
  {
      return make_range_iterator<SIZE>(make_slice_iterator<SIZES...>(
					   iter->begin() + idx, indices...),
				       iter.stride());
  }
}	// namespace detail
    
template <class RANGE, class... IS,
	  std::enable_if_t<rank<std::decay_t<RANGE>>() != 0>* = nullptr>
inline auto
slice(RANGE&& r, size_t idx, size_t size, IS... is)
{
    return make_range(detail::make_slice_iterator(
			  std::begin(r) + idx, is...), size);
}

template <size_t SIZE, size_t... SIZES, class RANGE, class... INDICES,
	  std::enable_if_t<rank<std::decay_t<RANGE>>() != 0 &&
			   sizeof...(SIZES) == sizeof...(INDICES)>* = nullptr>
inline auto
slice(RANGE&& r, size_t idx, INDICES... indices)
{
    return make_range<SIZE>(detail::make_slice_iterator<SIZES...>(
				std::begin(r) + idx, indices...));
}

/************************************************************************
*  supports for range tuples						*
************************************************************************/
template <size_t... SIZES, class ITER_TUPLE, class... ARGS> inline auto
make_range(zip_iterator<ITER_TUPLE> zip_iter, ARGS... args)
{
    return tuple_transform(zip_iter.get_iterator_tuple(),
			   [args...](auto iter)
			   {
			       return make_range<SIZES...>(iter, args...);
			   });
}
    
template <size_t... SIZES, class TUPLE, class... ARGS,
	  std::enable_if_t<all_has_begin<TUPLE>::value>* = nullptr>
inline auto
slice(TUPLE&& t, ARGS... args)
{
    return tuple_transform(t, [args...](auto&& x)
			      {
				  return slice<SIZES...>(
				      std::forward<decltype(x)>(x), args...);
			      });
}
    
/************************************************************************
*  class column_iterator<ROW, NROWS>					*
************************************************************************/
//! 2次元配列の列を指す反復子
/*!
  \param ROW	begin(), end()をサポートするコンテナを指す反復子の型
  \param NROWS	begin()とend()間の距離(0ならば可変長)
*/ 
template <class ROW, size_t NROWS>
class column_iterator
    : public boost::iterator_adaptor<column_iterator<ROW, NROWS>,
				     size_t,
				     range<vertical_iterator<ROW>, NROWS>,
				     std::random_access_iterator_tag,
				     range<vertical_iterator<ROW>, NROWS>,
				     ptrdiff_t>
{
  private:
    using super	= boost::iterator_adaptor<column_iterator,
					  size_t,
					  range<vertical_iterator<ROW>, NROWS>,
					  std::random_access_iterator_tag,
					  range<vertical_iterator<ROW>, NROWS>,
					  ptrdiff_t>;

  public:
    using	typename super::reference;

    friend	class boost::iterator_core_access;

  public:
		column_iterator(ROW row, size_t nrows, size_t col)
		    :super(col), _row(row), _nrows(nrows)
		{
		}

  private:
    reference	dereference() const
		{
		    return {{_row, super::base()}, _nrows};
		}
    
  private:
    ROW		_row;
    size_t	_nrows;
};

template <size_t NROWS=0, class ROW> inline column_iterator<ROW, NROWS>
make_column_iterator(ROW row, size_t nrows, size_t col)
{
    return {row, nrows, col};
}

template <class E> inline auto
column_begin(E& expr)
{
    return make_column_iterator<size0<E>()>(std::begin(expr), std::size(expr),
					    0);
}
    
template <class E> inline auto
column_begin(const E& expr)
{
    return make_column_iterator<size0<E>()>(std::begin(expr), std::size(expr),
					    0);
}
    
template <class E> inline auto
column_cbegin(const E& expr)
{
    return column_begin(expr);
}
    
template <class E> inline auto
column_end(E& expr)
{
    return make_column_iterator<size0<E>()>(std::begin(expr), std::size(expr),
					    size<1>(expr));
}

template <class E> inline auto
column_end(const E& expr)
{
    return make_column_iterator<size0<E>()>(std::begin(expr), std::size(expr),
					    size<1>(expr));
}

template <class E> inline auto
column_cend(const E& expr)
{
    return column_end(expr);
}

/************************************************************************
*  various numeric functions						*
************************************************************************/
namespace detail
{
  /**********************************************************************
  *  type aliases							*
  **********************************************************************/
  template <size_t I, size_t J>
  using max = std::integral_constant<size_t, (I > J ? I : J)>;

  template <class ITER, size_t SIZE>
  std::true_type	check_range(range<ITER, SIZE>)			;
  std::false_type	check_range(...)				;

  template <class E>
  using is_range = decltype(check_range(std::declval<E>()));

  /**********************************************************************
  *  struct opnode							*
  **********************************************************************/
  class opnode;

  template <class E>
  using is_opnode = std::is_convertible<E, opnode>;
    
  //! 演算子のノードを表すクラス
  class opnode
  {
    protected:
      template <class E_>
      using argument_t	= std::conditional_t<is_opnode<E_>::value ||
					     is_range<E_>::value,
					     const E_, const E_&>;
  };

  /**********************************************************************
  *  class unary_opnode<OP, E>						*
  **********************************************************************/
  //! 配列式に対する単項演算子を表すクラス
  /*!
    \param OP	各成分に適用される単項演算子の型
    \param E	単項演算子の引数となる式の型
  */
  template <class OP, class E>
  class unary_opnode : public opnode
  {
    public:
      using iterator	= boost::transform_iterator<OP, iterator_t<const E> >;
      using reference	= iterator_reference<iterator>;
      
    public:
		unary_opnode(const E& expr, OP op)
		    :_expr(expr), _op(op)				{}

      constexpr static size_t
		size0()		{ return TU::size0<E>(); }
      iterator	begin()	const	{ return {std::begin(_expr), _op}; }
      iterator	end()	const	{ return {std::end(_expr),   _op}; }
      size_t	size()	const	{ return std::size(_expr); }
      reference	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
      
    private:
      argument_t<E>	_expr;
      const OP		_op;
  };
    
  template <class OP, class E> inline unary_opnode<OP, E>
  make_unary_opnode(const E& expr, OP op)
  {
      return {expr, op};
  }

  /**********************************************************************
  *  class binary_opnode<OP, L, R>					*
  **********************************************************************/
  //! 配列式に対する2項演算子を表すクラス
  /*!
    \param OP	各成分に適用される2項演算子の型
    \param L	2項演算子の第1引数となる式の型
    \param R	2項演算子の第2引数となる式の型
  */
  template <class OP, class L, class R>
  struct binary_opnode : public opnode
  {
    public:
      using iterator	= transform_iterator2<OP, iterator_t<const L>,
						  iterator_t<const R> >;
      using reference	= iterator_reference<iterator>;

    public:
		binary_opnode(const L& l, const R& r, OP op)
		    :_l(l), _r(r), _op(op)
		{
		    assert(std::size(_l) == std::size(_r));
		}

      constexpr static size_t
		size0()
	  	{
		    return max<TU::size0<L>(), TU::size0<R>()>::value;
		}
      iterator	begin()	const	{ return {std::begin(_l), std::begin(_r), _op};}
      iterator	end()	const	{ return {std::end(_l),   std::end(_r),   _op};}
      size_t	size()	const	{ return std::size(_l); }
      reference	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}

    private:
      argument_t<L>	_l;
      argument_t<R>	_r;
      const OP		_op;
  };
    
  template <class OP, class L, class R> inline binary_opnode<OP, L, R>
  make_binary_opnode(const L& l, const R& r, OP op)
  {
      return {l, r, op};
  }
}	// namespace detail

//! 与えられた式の各要素の符号を反転する.
/*!
  \param expr	式
  \return	符号反転演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator -(const E& expr)
{
    return detail::make_unary_opnode(expr, [](const auto& x){ return -x; });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	乗算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator *(const E& expr, element_t<E> c)
{
    return detail::make_unary_opnode(expr, [c](const auto& x){ return x*c; });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param c	乗数
  \param expr	式
  \return	乗算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator *(element_t<E> c, const E& expr)
{
    return detail::make_unary_opnode(expr, [c](const auto& x){ return c*x; });
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	除算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator /(const E& expr, element_t<E> c)
{
    return detail::make_unary_opnode(expr, [c](const auto& x){ return x/c; });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	各要素にcが掛けられた結果の式
*/
template <class E> inline std::enable_if_t<rank<std::decay_t<E> >() != 0, E&>
operator *=(E&& expr, element_t<std::decay_t<E> > c)
{
    constexpr size_t	N = size0<std::decay_t<E> >();
    
    for_each<N>(std::begin(expr), std::size(expr), [c](auto&& x){ x *= c; });
    return expr;
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	各要素がcで割られた結果の式
*/
template <class E> inline std::enable_if_t<rank<std::decay_t<E> >() != 0, E&>
operator /=(E&& expr, element_t<std::decay_t<E> > c)
{
    constexpr size_t	N = size0<std::decay_t<E> >();
    
    for_each<N>(std::begin(expr), std::size(expr), [c](auto&& x){ x /= c; });
    return expr;
}

//! 与えられた2つの式の各要素の和をとる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	加算演算子ノード
*/
template <class L, class R,
	  std::enable_if_t<rank<L>() != 0 && rank<L>() == rank<R>()>* = nullptr>
inline auto
operator +(const L& l, const R& r)
{
    return detail::make_binary_opnode(l, r, [](const auto& x, const auto& y)
					    { return x + y; });
}

//! 与えられた2つの式の各要素の差をとる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	減算演算子ノード
*/
template <class L, class R,
	  std::enable_if_t<rank<L>() != 0 && rank<L>() == rank<R>()>* = nullptr>
inline auto
operator -(const L& l, const R& r)
{
    return detail::make_binary_opnode(l, r, [](const auto& x, const auto& y)
					    { return x - y; });
}

//! 与えられた左辺の式の各要素に右辺の式の各要素を加える.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	各要素が加算された左辺の式
*/
template <class L, class R>
inline std::enable_if_t<rank<std::decay_t<L> >() != 0 &&
			rank<std::decay_t<L> >() == rank<R>(), L&>
operator +=(L&& l, const R& r)
{
    constexpr size_t	N = size0<std::decay_t<L> >();
    
    for_each<N>(std::begin(l), std::size(l), std::begin(r),
		[](auto&& x, const auto& y){ x += y; });
    return l;
}

//! 与えられた左辺の式の各要素から右辺の式の各要素を減じる.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	各要素が減じられた左辺の式
*/
template <class L, class R>
inline std::enable_if_t<rank<std::decay_t<L> >() != 0 &&
			rank<std::decay_t<L> >() == rank<R>(), L&>
operator -=(L&& l, const R& r)
{
    constexpr size_t	N = size0<std::decay_t<L> >();
    
    for_each<N>(std::begin(l), std::size(l), std::begin(r),
		[](auto&& x, const auto& y){ x -= y; });
    return l;
}

/************************************************************************
*  generic algorithms for ranges					*
************************************************************************/
//! 与えられた2次元配列式の転置を返す
/*
  \param expr	2次元配列式
  \return	expr を転置した2次元配列式
 */ 
template <class E, std::enable_if_t<rank<E>() == 2>* = nullptr> inline auto
transpose(const E& expr)
{
    constexpr size_t	SIZE = size0<value_t<E> >();
    return make_range<SIZE>(column_begin(expr), size<1>(expr));
}

//! 与えられた式の各要素の自乗和を求める.
/*!
  \param x	式
  \return	式の各要素の自乗和
*/
template <class E, std::enable_if_t<(rank<E>() != 0)>* = nullptr> inline auto
square(const E& expr)
{
    return square<size0<E>()>(std::begin(expr), std::size(expr));
}

//! 与えられた式の各要素の自乗和の平方根を求める.
/*!
  \param x	式
  \return	式の各要素の自乗和の平方根
*/
template <class T> inline auto
length(const T& x)
{
    return std::sqrt(square(x));
}
    
//! 与えられた二つの式の各要素の差の自乗和を求める.
/*!
  \param x	第1の式
  \param y	第2の式
  \return	xとyの各要素の差の自乗和
*/
template <class L, class R> inline auto
square_distance(const L& x, const R& y)
{
    return square(x - y);
}
    
//! 与えられた二つの式の各要素の差の自乗和の平方根を求める.
/*!
  \param x	第1の式
  \param y	第2の式
  \return	xとyの各要素の差の自乗和の平方根
*/
template <class L, class R> inline auto
distance(const L& x, const R& y)
{
    return std::sqrt(square_distance(x, y));
}

//! 与えられた式のノルムを1に正規化する．
/*!
   \param expr	式 
   \return	正規化された式，すなわち
		\f$
		  \TUvec{e}{}\leftarrow\frac{\TUvec{e}{}}{\TUnorm{\TUvec{e}{}}}
		\f$
*/
template <class E> inline std::enable_if_t<rank<std::decay_t<E> >() != 0, E&>
normalize(E&& expr)
{
    return expr /= length(expr);
}

template <class E, class T>
inline std::enable_if_t<rank<E>() == 1 && std::is_arithmetic<T>::value,
			element_t<E> >
at(const E& expr, T x)
{
    const auto	x0 = std::floor(x);
    const auto	dx = x - x0;
    const auto	i  = size_t(x0);
    return (dx ? (1 - dx) * expr[i] + dx * expr[i+1] : expr[i]);
}

template <class E, class T>
inline std::enable_if_t<rank<E>() == 2 && std::is_arithmetic<T>::value,
			element_t<E> >
at(const E& expr, T x, T y)
{
    const auto	y0 = std::floor(y);
    const auto	dy = y - y0;
    const auto	i  = size_t(y0);
    const auto	a0 = at(expr[i], x);
    return (dy ? (1 - dy) * a0 + dy * at(expr[i+1], x) : a0);
}

}	// namespace TU
#endif	// !__TU_RANGE_H
