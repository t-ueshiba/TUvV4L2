/*!
  \file		range.h
  \author	Toshio UESHIBA
  \brief	2つの反復子によって指定されるレンジの定義と実装
*/
#ifndef TU_RANGE_H
#define TU_RANGE_H

#include <cassert>
#include <initializer_list>
#include "TU/iterator.h"
#include "TU/algorithm.h"

namespace std
{
  //! 式に適用できる反復子の型を返す
  /*!
    E, const E&, E&, E&&型の x に std::begin() または x が所属するnamespaceで
    定義された begin() がADLによって適用できるかチェックし，可能な場合はその型を
    返す．
  */
  template <class E>
  auto	check_begin(E&& x) -> decltype(begin(x))			;
  void	check_begin(...)						;
}

namespace TU
{
/************************************************************************
*  type aliases: iterator_t<E>, value_t<E>, element_t<E>		*
************************************************************************/
//! 式が持つ反復子の型を返す
/*!
  \param E	式またはそれへの参照の型
  \return	E が反復子を持てばその型，持たなければ void
*/
template <class E>
using iterator_t = decltype(std::check_begin(std::declval<E>()));

//! 式が持つ逆反復子の型を返す
/*!
  反復子を持たない式を与えるとコンパイルエラーとなる.
  \param E	式またはそれへの参照の型
  \return	E が持つ逆反復子の型
*/
template <class E>
using reverse_iterator_t = std::reverse_iterator<iterator_t<E> >;

//! 式が持つ反復子が指す型を返す
/*!
  反復子を持たない式を与えるとコンパイルエラーとなる.
  \param E	反復子を持つ式またはそれへの参照の型
  \return	E の反復子が指す型
*/
template <class E>
using value_t	= iterator_value<iterator_t<E> >;

namespace detail
{
  template <class T>
  struct identity
  {
      using type = T;
  };

  template <class E, class=iterator_t<E> >
  struct element_t
  {
      using type = typename element_t<value_t<E> >::type;
  };
  template <class E>
  struct element_t<E, void> : identity<E>				{};
}	// namespace detail
    
//! 式が持つ反復子が指す型を再帰的に辿って到達する型を返す
/*!
  \param E	式またはそれへの参照の型
  \return	E が反復子を持てばそれが指す型を再帰的に辿って到達する型，
		持たなければ E 自身
*/
template <class E>
using element_t	= typename detail::element_t<E>::type;

/************************************************************************
*  rank<E>()								*
************************************************************************/
namespace detail
{
  template <class E>
  auto	check_stdbegin(E&& x) -> decltype(std::begin(x),
					  std::true_type())		;
  auto	check_stdbegin(...)   -> std::false_type			;

  template <class E>
  using has_stdbegin = decltype(check_stdbegin(std::declval<E>()));
}	// namespace detail

//! 式の次元数(軸の個数)を返す
/*!
  \param E	式の型
  \return	式の次元数
*/
template <class E>
constexpr std::enable_if_t<!detail::has_stdbegin<E>::value, size_t>
rank()
{
    return 0;
}
template <class E>
constexpr std::enable_if_t<detail::has_stdbegin<E>::value, size_t>
rank()
{
    return 1 + rank<value_t<E> >();
}
    
/************************************************************************
*  size0<E>()								*
************************************************************************/
namespace detail
{
  template <class E>
  auto		  has_size0(E) -> decltype(E::size0(), std::true_type());
  std::false_type has_size0(...)					;
}	// namespace detail

template <class E>
using has_size0 = decltype(detail::has_size0(std::declval<E>()));
    
//! 式の最上位軸の要素数を返す
/*!
  本関数で返されるのは静的なサイズであり，式が可変個の要素を持つ場合，0が返される
  \param E	式またはそれへの参照の型
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
    return std::decay_t<E>::size0();
}
    
/************************************************************************
*  size<E>()								*
************************************************************************/
namespace detail
{
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

  template <class E> inline auto
  size(const E& expr, std::integral_constant<size_t, 0>)
  {
      return std::size(expr);
  }
  template <size_t I, class E> inline auto
  size(const E& expr, std::integral_constant<size_t, I>)
  {
      return size(*std::begin(expr), std::integral_constant<size_t, I-1>());
  }
}	// namespace detail

//! 与えられた式について，指定された軸の要素数を返す
/*!
  \param I	軸を指定するindex (0 <= I < Dimension)
  \param E	式の型
  \return	軸Iの要素数
 */
template <size_t I, class E, std::enable_if_t<rank<E>() != 0>* = nullptr>
inline auto
size(const E& expr)
{
    return detail::size(expr, std::integral_constant<size_t, I>());
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
  \param SIZE	レンジサイズ(0ならば可変長)
*/
template <class ITER, size_t SIZE=0>
class range
{
  public:
    using value_type	= iterator_value<ITER>;
    
  public:
		range(ITER begin)		:_begin(begin)	{}
		range(ITER begin, size_t)	:_begin(begin)	{}
    
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
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		range(const range<ITER_, SIZE>& r)
		    :_begin(r.begin())
		{
		}

    template <class E_> std::enable_if_t<rank<E_>() != 0, range&>
		operator =(const E_& expr)
		{
		    using	std::begin;
		    using	std::size;
		    
		    assert(size(expr) == SIZE);
		    copy<SIZE>(begin(expr), SIZE, _begin);
		    return *this;
		}

		range(std::initializer_list<value_type> args)
		    :_begin(const_cast<value_type*>(args.begin()))
    		{
		    assert(args.size() == size());
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == SIZE);
		  // initializer_list<T> はalignmentされないので，
		  // SIMD命令が使われぬようcopy<N>()は使用しない．
		    std::copy_n(args.begin(), SIZE, _begin);
		    return *this;
		}

    template <class T_> std::enable_if_t<rank<T_>() == 0, range&>
		operator =(const T_& c)
		{
		    fill<SIZE>(_begin, SIZE, c);
		    return *this;
		}

    constexpr static
    size_t	size0()		{ return SIZE; }
    constexpr static
    size_t	size()		{ return SIZE; }
    auto	begin()	  const	{ return _begin; }
    auto	end()	  const	{ return _begin + SIZE; }
    auto	cbegin()  const	{ return begin(); }
    auto	cend()	  const	{ return end(); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	crend()	  const	{ return rend(); }
    decltype(auto)
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
		}
    
  private:
    const ITER	_begin;
};

//! 可変長レンジ
/*!
  \param ITER	反復子の型
*/
template <class ITER>
class range<ITER, 0>
{
  public:
    using value_type	= iterator_value<ITER>;
    
  public:
		range(ITER begin, size_t size)
		    :_begin(begin), _size(size)			{}
    
		range()						= delete;
		range(const range&)				= default;
    range&	operator =(const range& r)
		{
		    assert(r.size() == size());
		    copy<0>(r._begin, _size, _begin);
		    return *this;
		}
		range(range&&)					= default;
    range&	operator =(range&&)				= default;
    
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		range(const range<ITER_, 0>& r)
		    :_begin(r.begin()), _size(r.size())
		{
		}

    template <class E_> std::enable_if_t<rank<E_>() != 0, range&>
		operator =(const E_& expr)
		{
		    using	std::begin;
		    using	std::size;
		    
		    assert(size(expr) == _size);
		    copy<TU::size0<E_>()>(begin(expr), _size, _begin);
		    return *this;
		}
		
		range(std::initializer_list<value_type> args)
		    :_begin(const_cast<value_type*>(args.begin())),
		     _size(args.size())
    		{
		}
    range&	operator =(std::initializer_list<value_type> args)
		{
		    assert(args.size() == _size);
		  // initializer_list<T> はalignmentされないので，
		  // SIMD命令が使われぬようcopy<N>()は使用しない．
		    std::copy_n(args.begin(), _size, _begin);
		    return *this;
		}
		
    template <class T_> std::enable_if_t<rank<T_>() == 0, range&>
		operator =(const T_& c)
		{
		    fill<0>(_begin, _size, c);
		    return *this;
		}

    constexpr static
    size_t	size0()		{ return 0; }
    size_t	size()	  const	{ return _size; }
    auto	begin()	  const	{ return _begin; }
    auto	end()	  const	{ return _begin + _size; }
    auto	cbegin()  const	{ return begin(); }
    auto	cend()	  const	{ return end(); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	crend()	  const	{ return rend(); }
    decltype(auto)
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(_begin + i);
		}

  private:
    const ITER		_begin;
    const size_t	_size;
};

//! 固定長レンジを生成する
/*!
  \param SIZE	レンジサイズ
  \param iter	レンジの先頭要素を指す反復子
  \return	固定長レンジ
*/
template <size_t SIZE, class ITER> inline range<ITER, SIZE>
make_range(ITER iter)
{
    return {iter};
}
    
//! 可変長レンジを生成する
/*!
  \param iter	レンジの先頭要素を指す反復子
  \param size	レンジサイズ
  \return	可変長レンジ
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
  //for (const auto& val : r)
  //	out << ' ' << val;
    for (auto iter = r.begin(); iter != r.end(); ++iter)
	out << ' ' << *iter;
    return out << std::endl;
}
    
/************************************************************************
*  type alias: iterator_stride<ITER>					*
************************************************************************/
namespace detail
{
  template <class ITER>
  struct iterator_stride
  {
    private:
      template <class ITER_, class BASE_, class VAL_,
		class CAT_,  class REF_,  class DIFF_>
      static BASE_	check_base_type(
			    boost::iterator_adaptor<ITER_, BASE_, VAL_,
						    CAT_, REF_, DIFF_>)	;
      static void	check_base_type(...)				;
      using base_type	= decltype(check_base_type(std::declval<ITER>()));

      template <class ITER_>
      static auto	check_difference(ITER_)
			    -> iterator_difference<ITER_>		;
      static ptrdiff_t	check_difference(...)				;
      using diff_type	= decltype(check_difference(std::declval<ITER>()));
      
    public:
      using type	= typename std::conditional_t<
				       std::is_void<base_type>::value,
				       identity<diff_type>,
				       iterator_stride<base_type> >::type;
  };
  template <class... ITER_>
  struct iterator_stride<zip_iterator<std::tuple<ITER_...> > >
  {
      using type	= std::tuple<typename iterator_stride<ITER_>::type...>;
  };
}	// namespace detail

template <class ITER>
using iterator_stride = typename detail::iterator_stride<ITER>::type;
    
/************************************************************************
*  class range_iterator<ITER, STRIDE, SIZE>				*
************************************************************************/
namespace detail
{
  template <class DIFF, ptrdiff_t STRIDE, size_t SIZE>
  struct stride_and_size
  {
      using stride_t	= ptrdiff_t;
      
      stride_and_size(stride_t, size_t)		{}
      constexpr static auto	stride()	{ return STRIDE; }
      constexpr static auto	size()		{ return SIZE; }
  };
  template <class DIFF, size_t SIZE>
  struct stride_and_size<DIFF, 0, SIZE>
  {
      using stride_t	= DIFF;
      
      stride_and_size(stride_t stride, size_t)
	  :_stride(stride)			{}
      auto			stride() const	{ return _stride; }
      constexpr static auto	size()		{ return SIZE; }

    private:
      stride_t	_stride;
  };
  template <class DIFF, ptrdiff_t STRIDE>
  struct stride_and_size<DIFF, STRIDE, 0>
  {
      using stride_t	= ptrdiff_t;
      
      stride_and_size(stride_t, size_t size)
	  :_size(size)				{}
      constexpr static auto	stride()	{ return STRIDE; }
      auto			size()	 const	{ return _size; }

    private:
      size_t	_size;
  };
  template <class DIFF>
  struct stride_and_size<DIFF, 0, 0>
  {
      using stride_t	= DIFF;
      
      stride_and_size(stride_t stride, size_t size)
	  :_stride(stride), _size(size)		{}
      auto			stride() const	{ return _stride; }
      auto			size()	 const	{ return _size; }

    private:
      stride_t	_stride;
      size_t	_size;
  };
}	// namespace detail

//! 配列を一定間隔に切り分けたレンジを指す反復子
/*!
  \param ITER	配列の要素を指す反復子の型
  \param STRIDE	インクリメントしたときに進める要素数(0ならば可変)
  \param SIZE	レンジサイズ(0ならば可変長)
*/
template <class ITER, ptrdiff_t STRIDE, size_t SIZE>
class range_iterator
    : public boost::iterator_adaptor<range_iterator<ITER, STRIDE, SIZE>,
				     ITER,
				     range<ITER, SIZE>,
				     boost::use_default,
				     range<ITER, SIZE> >,
      public detail::stride_and_size<iterator_stride<ITER>, STRIDE, SIZE>
{
  private:
    using super	= boost::iterator_adaptor<range_iterator,
					  ITER,
					  range<ITER, SIZE>,
					  boost::use_default,
					  range<ITER, SIZE> >;
    using ss	= detail::stride_and_size<iterator_stride<ITER>,
					  STRIDE, SIZE>;
    
  public:
    using		typename super::reference;
    using		typename super::difference_type;
    using stride_t    =	typename ss::stride_t;
    friend class	boost::iterator_core_access;
	  
  public:
		range_iterator(ITER iter=ITER(),
			       iterator_stride<ITER> stride=STRIDE,
			       size_t size=SIZE)
		    :super(iter), ss(stride, size)			{}

    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		range_iterator(
		    const range_iterator<ITER_, STRIDE, SIZE>& iter)
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
		    advance(super::base_reference(), stride());
		}
    void	decrement()
		{
		    advance(super::base_reference(), -stride());
		}
    void	advance(difference_type n)
		{
		    advance(super::base_reference(), n*stride());
		}
    difference_type
		distance_to(const range_iterator& iter) const
		{
		    return (iter.base() - super::base()) / leftmost(stride());
		}

    template <class ITER_, class STRIDE_>
    static void	advance(ITER_& iter, STRIDE_ stride)
		{
		    iter += stride;
		}
    template <class ITER_TUPLE_, class... STRIDE_>
    static void	advance(zip_iterator<ITER_TUPLE_>& iter,
			std::tuple<STRIDE_...> stride)
		{
		    tuple_for_each([](auto&& x, auto y)
				   { range_iterator::advance(x, y); },
				   const_cast<ITER_TUPLE_&>(
				       iter.get_iterator_tuple()),
				   stride);
		}
    template <class ITER_, class... STRIDE_>
    static void	advance(ITER_& iter, std::tuple<STRIDE_...> stride)
		{
		    advance(const_cast<typename ITER_::base_type&>(iter.base()),
			    stride);
		}

    template <class STRIDE_>
    static auto	leftmost(STRIDE_ stride)
		{
		    return stride;
		}
    template <class... STRIDE_>
    static auto	leftmost(std::tuple<STRIDE_...> stride)
		{
		    return leftmost(std::get<0>(stride));
		}
};

//! 反復子が指すレンジが所属する軸のストライドを返す
/*!
  \param iter	レンジを指す反復子
  \return	レンジ軸のストライド
*/
template <class ITER> inline auto
stride(ITER iter)
{
    return std::size(*iter);
}
template <class ITER, ptrdiff_t STRIDE, size_t SIZE> inline auto
stride(const range_iterator<ITER, STRIDE, SIZE>& iter)
{
    return iter.stride();
}
template <class ITER_TUPLE> inline auto
stride(const zip_iterator<ITER_TUPLE>& iter)
{
    return tuple_transform([](const auto& it){ return stride(it); },
			   iter.get_iterator_tuple());
}

//! 固定長レンジを指し，インクリメント時に固定した要素数だけ進める反復子を生成する
/*!
  \param STRIDE	インクリメント時に進める要素数
  \param SIZE	レンジサイズ
  \param iter	レンジの先頭要素を指す反復子
  \return	レンジ反復子
*/
template <ptrdiff_t STRIDE, size_t SIZE, class ITER>
inline range_iterator<ITER, STRIDE, SIZE>
make_range_iterator(ITER iter)
{
    return {iter};
}

//! 固定長レンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param SIZE	レンジサイズ
  \param iter	レンジの先頭要素を指す反復子
  \param stride	インクリメント時に進める要素数
  \return	レンジ反復子
*/
template <size_t SIZE, class ITER>
inline range_iterator<ITER, 0, SIZE>
make_range_iterator(ITER iter, iterator_stride<ITER> stride)
{
    return {iter, stride};
}
    
//! 指定された長さのレンジを指し，インクリメント時に指定した要素数だけ進める反復子を生成する
/*!
  \param iter	レンジの先頭要素を指す反復子
  \param stride	インクリメント時に進める要素数
  \param size	レンジサイズ
  \return	レンジ反復子
*/
template <class ITER> inline range_iterator<ITER, 0, 0>
make_range_iterator(ITER iter, iterator_stride<ITER> stride, size_t size)
{
    return {iter, stride, size};
}
    
/************************************************************************
*  fixed size & fixed stride ranges and associated iterators		*
************************************************************************/
//! 多次元固定長レンジを指し，インクリメント時に固定したブロック数だけ進める反復子を生成する
/*!
  \param STRIDE	インクリメント時に進める最上位軸のブロック数
  \param SIZE	最上位軸のレンジサイズ
  \param SS	2番目以降の軸の{ストライド, レンジサイズ}の並び
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
  \param SIZE		最上位軸のレンジサイズ
  \param SIZES		2番目以降の軸のレンジサイズの並び
  \param stride		最上位軸のストライド
  \param strides	2番目以降の軸のストライドの並び
  \param iter		レンジの先頭要素を指す反復子
  \return		レンジ反復子
*/
template <size_t SIZE, size_t... SIZES, class ITER, class... STRIDES,
	  std::enable_if_t<sizeof...(SIZES) == sizeof...(STRIDES)>* = nullptr>
inline auto
make_range_iterator(ITER iter,
		    iterator_stride<ITER> stride, STRIDES... strides)
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
  \param size		最上位軸のレンジサイズ
  \param ss		2番目以降の軸の{ストライド, レンジサイズ}の並び
  \return		レンジ反復子
*/
template <class ITER, class... SS> inline auto
make_range_iterator(ITER iter,
		    iterator_stride<ITER> stride, size_t size, SS... ss)
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
	  std::enable_if_t<rank<RANGE>() != 0>* = nullptr>
inline auto
slice(RANGE&& r, size_t idx, size_t size, IS... is)
{
    return make_range(detail::make_slice_iterator(
			  std::begin(r) + idx, is...), size);
}

template <size_t SIZE, size_t... SIZES, class RANGE, class... INDICES,
	  std::enable_if_t<rank<RANGE>() != 0 &&
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
make_range(const zip_iterator<ITER_TUPLE>& zip_iter, ARGS... args)
{
    return tuple_transform([args...](auto iter)
			   {
			       return make_range<SIZES...>(iter, args...);
			   }, zip_iter.get_iterator_tuple());
}
    
template <size_t... SIZES, class TUPLE, class... ARGS,
	  std::enable_if_t<is_tuple<TUPLE>::value>* = nullptr>
inline auto
slice(TUPLE&& t, ARGS... args)
{
    return tuple_transform([args...](auto&& x)
			   {
			       return slice<SIZES...>(
				   std::forward<decltype(x)>(x), args...);
			   }, t);
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
		    return {{{super::base()}, _row}, _nrows};
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
column_begin(E&& expr)
{
    using	std::begin;
    using	std::size;
    
    constexpr auto	N = size0<std::remove_reference_t<E> >();
    
    return make_column_iterator<N>(begin(expr), size(expr), 0);
}

template <class E> inline auto
column_cbegin(const E& expr)
{
    return column_begin(expr);
}
    
template <class E> inline auto
column_end(E&& expr)
{
    using	std::begin;
    using	std::size;
    
    constexpr auto	N = size0<std::remove_reference_t<E> >();
    
    return make_column_iterator<N>(begin(expr), size(expr), size<1>(expr));
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
  template <size_t I, size_t... J>
  struct max
  {
      constexpr static size_t	value = (I > max<J...>::value ?
					 I : max<J...>::value);
  };
  template <size_t I>
  struct max<I>
  {
      constexpr static size_t	value = I;
  };

  template <class ITER, size_t SIZE>
  std::true_type	check_range(range<ITER, SIZE>)			;
  std::false_type	check_range(...)				;

  template <class E>
  using is_range = decltype(check_range(std::declval<E>()));

  /**********************************************************************
  *  struct opnode							*
  **********************************************************************/
  //! 演算子のノードを表すクラス
  class opnode
  {
  };

  template <class E>
  using is_opnode = std::is_convertible<E, opnode>;
    
  /**********************************************************************
  *  class unary_opnode<OP, E>						*
  **********************************************************************/
  //! 配列式に対する単項演算子を表すクラス
  /*!
    与えられた配列式が一時オブジェクトの場合は，その内容を本クラスオブジェクト
    内に保持した実体にmoveする．そうでない場合は，それへの参照を本クラスオブ
    ジェクト内に保持する．
    \param OP	各成分に適用される単項演算子の型
    \param E	単項演算子の引数となる式または式への参照の型
  */
  template <class OP, class E>
  class unary_opnode : public opnode
  {
    public:
		unary_opnode(E&& expr, OP op)
		    :_expr(std::forward<E>(expr)), _op(op)	{}

      constexpr static auto
		size0()
		{
		    return TU::size0<E>();
		}
      auto	begin()	const
		{
		    return TU::make_transform_iterator(_op, std::cbegin(_expr));
		}
      auto	end() const
		{
		    return TU::make_transform_iterator(_op, std::cend(_expr));
		}
      auto	size() const
		{
		    return std::size(_expr);
		}
      decltype(auto)
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
      
    private:
      const E	_expr;
      const OP	_op;
  };
    
  template <class OP, class E> inline auto
  make_unary_opnode(E&& expr, OP op)
  {
    // exprの実引数がX&&型(X型の一時オブジェクト)ならば E = X,
    // そうでなければ E = X& または E = const X& となる．
      return unary_opnode<OP, E>(std::forward<E>(expr), op);
  }

  /**********************************************************************
  *  class binary_opnode<OP, L, R>					*
  **********************************************************************/
  //! 配列式に対する2項演算子を表すクラス
  /*!
    与えられた配列式が一時オブジェクトの場合は，その内容を本クラスオブジェクト
    内に保持した実体にmoveする．そうでない場合は，それへの参照を本クラスオブ
    ジェクト内に保持する．
    \param OP	各成分に適用される2項演算子の型
    \param L	2項演算子の第1引数となる式または式への参照の型
    \param R	2項演算子の第2引数となる式または式への参照の型
  */
  template <class OP, class L, class R>
  struct binary_opnode : public opnode
  {
    public:
		binary_opnode(L&& l, R&& r, OP op)
		    :_l(std::forward<L>(l)), _r(std::forward<R>(r)), _op(op)
		{
		    assert(std::size(_l) == std::size(_r));
		}

      constexpr static auto
		size0()
	  	{
		    return max<TU::size0<L>(), TU::size0<R>()>::value;
		}
      auto	begin()	const
		{
		    return TU::make_transform_iterator(
				_op, std::cbegin(_l), std::cbegin(_r));
		}
      auto	end() const
		{
		    return TU::make_transform_iterator(
				_op, std::cend(_l), std::cend(_r));
		}
      auto	size()	const	{ return std::size(_l); }
      decltype(auto)
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}

    private:
      const L	_l;
      const R	_r;
      const OP	_op;
  };
    
  template <class OP, class L, class R> inline auto
  make_binary_opnode(L&& l, R&& r, OP op)
  {
    // l(r)の実引数がX&&型(Y&&型)ならば L = X (R = Y), そうでなければ
    // L = X& (R = Y&) または L = const X& (R = const Y&) となる．
      return binary_opnode<OP, L, R>(std::forward<L>(l),
				     std::forward<R>(r), op);
  }

}	// namespace detail

//! 与えられた式の各要素の符号を反転する.
/*!
  \param expr	式
  \return	符号反転演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator -(E&& expr)
{
    return detail::make_unary_opnode(std::forward<E>(expr),
				     [](auto&& x)
				     { return -std::forward<decltype(x)>(x); });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	乗算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator *(E&& expr, element_t<E> c)
{
    return detail::make_unary_opnode(std::forward<E>(expr),
				     [c](auto&& x)
				     { return std::forward<decltype(x)>(x)*c; });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param c	乗数
  \param expr	式
  \return	乗算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator *(element_t<E> c, E&& expr)
{
    return detail::make_unary_opnode(std::forward<E>(expr),
				     [c](auto&& x)
				     { return c*std::forward<decltype(x)>(x); });
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	除算演算子ノード
*/
template <class E, std::enable_if_t<rank<E>() != 0>* = nullptr> inline auto
operator /(E&& expr, element_t<E> c)
{
    return detail::make_unary_opnode(std::forward<E>(expr),
				     [c](auto&& x)
				     { return std::forward<decltype(x)>(x)/c; });
}

//! 与えられた式の各要素に定数を掛ける.
/*!
  \param expr	式
  \param c	乗数
  \return	各要素にcが掛けられた結果の式
*/
template <class E> inline std::enable_if_t<rank<E>() != 0, E&>
operator *=(E&& expr, element_t<E> c)
{
    constexpr size_t	N = size0<E>();
    
    for_each<N>(std::begin(expr), std::size(expr), [c](auto&& x){ x *= c; });
    return expr;
}

//! 与えられた式の各要素を定数で割る.
/*!
  \param expr	式
  \param c	除数
  \return	各要素がcで割られた結果の式
*/
template <class E> inline std::enable_if_t<rank<E>() != 0, E&>
operator /=(E&& expr, element_t<E> c)
{
    constexpr size_t	N = size0<E>();
    
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
operator +(L&& l, R&& r)
{
    return detail::make_binary_opnode(std::forward<L>(l), std::forward<R>(r),
				      [](auto&& x, auto&& y)
				      { return std::forward<decltype(x)>(x)
					     + std::forward<decltype(y)>(y); });
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
operator -(L&& l, R&& r)
{
    return detail::make_binary_opnode(std::forward<L>(l), std::forward<R>(r),
				      [](auto&& x, auto&& y)
				      { return std::forward<decltype(x)>(x)
					     - std::forward<decltype(y)>(y); });
}

//! 与えられた左辺の式の各要素に右辺の式の各要素を加える.
/*!
  \param l	左辺の式
  \param r	右辺の式
  \return	各要素が加算された左辺の式
*/
template <class L, class R>
inline std::enable_if_t<rank<L>() != 0 && rank<L>() == rank<R>(), L&>
operator +=(L&& l, const R& r)
{
    constexpr size_t	N = detail::max<size0<L>(), size0<R>()>::value;
    
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
inline std::enable_if_t<rank<L>() != 0 && rank<L>() == rank<R>(), L&>
operator -=(L&& l, R&& r)
{
    constexpr size_t	N = detail::max<size0<L>(), size0<R>()>::value;
    
    for_each<N>(std::begin(l), std::size(l), std::begin(r),
		[](auto&& x, const auto& y){ x -= y; });
    return l;
}

/************************************************************************
*  transposition of 2D expressions					*
************************************************************************/
namespace detail
{
  /**********************************************************************
  *  class transpose_opnode<E>						*
  **********************************************************************/
  template <class E>
  class transpose_opnode : public opnode
  {
    public:
		transpose_opnode(E&& expr)
		    :_expr(std::forward<E>(expr))
		{
		}

      constexpr static auto
		size0()
		{
		    return TU::size0<value_t<E> >();
		}
      auto	begin()	const
		{
		    return column_begin(_expr);
		}
      auto	end() const
		{
		    return column_end(_expr);
		}
      auto	size() const
		{
		    return TU::size<1>(_expr);
		}
      decltype(auto)
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
      E		transpose() const 
		{
		    return std::forward<E>(const_cast<transpose_opnode*>(this)
					   ->_expr);
		}

    private:
      E		_expr;
  };

  template <class E>
  std::true_type	check_transposed(const transpose_opnode<E>&)	;
  std::false_type	check_transposed(...)				;
}
    
//! 2次元配列式が他の2次元配列式を転置したものであるか判定する．
template <class E>
using is_transposed = decltype(detail::check_transposed(std::declval<E>()));

//! 与えられたスカラもしくは1次元配列式をそのまま返す
/*
  \param expr	スカラまたは1次元配列式
  \return	expr への定数参照
 */
template <class E> inline std::enable_if_t<(rank<E>() < 2), E&&>
transpose(E&& expr)
{
    return std::forward<E>(expr);
}

//! 与えられた2次元配列式の転置を返す
/*
  \param expr	2次元配列式
  \return	expr を転置した2次元配列式
 */ 
template <class E,
	  std::enable_if_t<rank<E>() == 2 &&
			   !is_transposed<E>::value>* = nullptr> inline auto
transpose(E&& expr)
{
    return detail::transpose_opnode<E>(std::forward<E>(expr));
}

//! 転置された2次元配列式をさらに転置して元に戻す
/*
  \param r	転置された2次元配列式を表すレンジ
  \return	r をさらに転置した2次元配列式
 */ 
template <class E> inline decltype(auto)
transpose(const detail::transpose_opnode<E>& expr)
{
    return expr.transpose();
}

/************************************************************************
*  generic algorithms for ranges					*
************************************************************************/
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
template <class E> inline std::enable_if_t<rank<E>() != 0, E&>
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
#endif	// !TU_RANGE_H
