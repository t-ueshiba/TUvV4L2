/*!
  \file		Array++.h
  \brief	多次元配列クラスの定義と実装
*/
#ifndef __TU_ARRAY_H
#define __TU_ARRAY_H

#include <array>
#include <iomanip>		// for std::ws
#include <memory>		// for std::allocator<T>, std::unique_ptr<T>
#include "TU/range.h"

namespace TU
{
/************************************************************************
*  class external_allocator<T>						*
************************************************************************/
template <class T>
class external_allocator
{
  public:
    using value_type		= T;
    using pointer		= T*;
    using const_pointer		= const T*;
    using reference		= T&;
    using const_reference	= const T&;
    using size_type		= size_t;
    using difference_type	= ptrdiff_t;
    
    template <class T_>
    struct rebind	{ using other = external_allocator<T_>; };

  public:
			external_allocator(pointer p, size_type size)
			    :_p(p), _size(size)				{}

    pointer		allocate(size_type n,
				 typename std::allocator<void>
					     ::const_pointer=nullptr) const
			{
			    if (n > _size)
				throw std::runtime_error("TU::external_allocator<T>::allocate(): too large memory requested!");
			    
			    return _p;
			}
    static void		deallocate(pointer, size_type)			{}
    static void		construct(pointer p, const_reference val)	{}
    static void		destroy(pointer p)				{}
    constexpr size_type	max_size()		const	{ return _size; }
    static pointer	address(reference r)		{ return &r; }
    static const_pointer
			address(const_reference r)	{ return &r; }

  private:
    const pointer	_p;
    const size_type	_size;
};
    
/************************************************************************
*  class BufTraits<T>							*
************************************************************************/
template <class T, class ALLOC>
struct BufTraits
{
    using allocator_type = ALLOC;
    using pointer	 = typename allocator_type::pointer;
    using const_pointer	 = typename allocator_type::const_pointer;
    using iterator	 = pointer;
    using const_iterator = const_pointer;
    
  protected:
    static pointer	null()
			{
			    return nullptr;
			}
    
    template <class IN_, class OUT_>
    static OUT_		copy(IN_ in, IN_ ie, OUT_ out)
			{
			    return std::copy(in, ie, out);
			}

    template <class IN_, class OUT_>
    static OUT_		copy(IN_ in, size_t n, OUT_ out)
			{
			    return std::copy_n(in, n, out);
			}

    template <class T_>
    static void		fill(iterator in, iterator ie, const T_& c)
			{
			    std::fill(in, ie, c);
			}

    template <class T_>
    static void		fill(iterator in, size_t n, const T_& c)
			{
			    std::fill_n(in, n, c);
			}
};

/************************************************************************
*  class Buf<T, ALLOC, SIZE, SIZES...>					*
************************************************************************/
//! 固定長多次元バッファクラス
/*!
  単独で使用することはなく，#array の内部バッファクラスとして使う．
  \param T	要素の型
  \param ALLOC	アロケータの型
  \param SIZE	最初の軸の要素数
  \param SIZES	2番目以降の各軸の要素数
*/
template <class T, class ALLOC, size_t SIZE, size_t... SIZES>
class Buf : public BufTraits<T, ALLOC>
{
  private:
  // このバッファの総容量をコンパイル時に計算
    template <size_t SIZE_, size_t... SIZES_>
    struct prod
    {
	constexpr static size_t	value = SIZE_ * prod<SIZES_...>::value;
    };
    template <size_t SIZE_>
    struct prod<SIZE_>
    {
	constexpr static size_t value = SIZE_;
    };

    constexpr static size_t	Capacity = prod<SIZE, SIZES...>::value;

  // このバッファの各軸のサイズをコンパイル時に計算
    template <size_t I_, size_t SIZE_, size_t... SIZES_>
    struct nth
    {
	constexpr static size_t	value = nth<I_-1, SIZES_...>::value;
    };
    template <size_t SIZE_, size_t... SIZES_>
    struct nth<0, SIZE_, SIZES_...>
    {
	constexpr static size_t	value = SIZE_;
    };

    template <size_t I_>
    using siz			= nth<I_, SIZE, SIZES...>;
    template <size_t I_>
    using axis			= std::integral_constant<size_t, I_>;
    using super			= BufTraits<T, ALLOC>;
    
  public:
    constexpr static size_t	D = 1 + sizeof...(SIZES);

    using sizes_type		= std::array<size_t, D>;
    using value_type		= T;
    using allocator_type	= void;
    using typename super::pointer;
    using typename super::const_pointer;

  public:
  // 標準コンストラクタ/代入演算子およびデストラクタ
		Buf()
		{
		    init(typename std::is_arithmetic<value_type>::type());
		}
		Buf(const Buf&)			= default;
    Buf&	operator =(const Buf&)		= default;
		Buf(Buf&&)			= default;
    Buf&	operator =(Buf&&)		= default;
    
  // 各軸のサイズと最終軸のストライドを指定したコンストラクタとリサイズ関数
    explicit	Buf(const sizes_type& sizes, size_t=0)
		{
		    if (!check_sizes(sizes, axis<D>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::Buf(): mismatched size!");
		    init(typename std::is_arithmetic<value_type>::type());
		}
    void	resize(const sizes_type& sizes, size_t=0)
		{
		    if (!check_sizes(sizes, axis<D>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::resize(): mismatched size!");
		}

    template <size_t I_=0>
    constexpr static auto	size()		{ return siz<I_>::value; }
    template <size_t I_=D-1>
    constexpr static auto	stride()	{ return siz<I_>::value; }
    constexpr static auto	nrow()		{ return siz<0>::value; }
    constexpr static auto	ncol()		{ return siz<1>::value; }

    auto	data()		{ return _a.data(); }
    auto	data()	const	{ return _a.data(); }
    auto	begin()		{ return make_iterator<SIZES...>(_a.begin()); }
    auto	begin()	const	{ return make_iterator<SIZES...>(_a.begin()); }
    auto	end()		{ return make_iterator<SIZES...>(_a.end()); }
    auto	end()	const	{ return make_iterator<SIZES...>(_a.end()); }

    void	fill(const T& c)		{ _a.fill(c); }
    std::istream&
		get(std::istream& in)
		{
		    for (auto& val : _a)
			in >> val;
		    return in;
		}
    
  private:
    void	init(std::true_type)		{ _a.fill(0); }
    void	init(std::false_type)		{}

    template <size_t I_>
    static bool	check_sizes(const sizes_type& sizes, axis<I_>)
		{
		    return (sizes[I_-1] != size<I_-1>() ? false :
			    check_sizes(sizes, axis<I_-1>()));
		}
    static bool	check_sizes(const sizes_type& sizes, axis<0>)
		{
		    return true;
		}
    
    template <class ITER_>
    static auto	make_iterator(ITER_ iter)
		{
		    return iter;
		}
    template <size_t SIZE_, size_t... SIZES_, class ITER_>
    static auto	make_iterator(ITER_ iter)
		{
		    return make_range_iterator<SIZE_, SIZE_>(
			       make_iterator<SIZES_...>(iter));
		}

  private:
    alignas(sizeof(T)) std::array<T, Capacity>	_a;
};

//! 可変長多次元バッファクラス
/*!
  単独で使用することはなく，#array の内部バッファクラスとして使う．
  \param T	要素の型
  \param ALLOC	アロケータの型
  \param SIZES	ダミー(各軸の要素数は動的に決定される)
*/
template <class T, class ALLOC, size_t... SIZES>
class Buf<T, ALLOC, 0, SIZES...> : public BufTraits<T, ALLOC>
{
  private:
    using super			= BufTraits<T, ALLOC>;
    using base_iterator		= typename super::iterator;
    using const_base_iterator	= typename super::const_iterator;
    template <size_t I_>
    using axis			= std::integral_constant<size_t, I_>;

  public:
    constexpr static size_t	D = 1 + sizeof...(SIZES);

    using sizes_type		= std::array<size_t, D>;
    using value_type		= T;
    using typename super::allocator_type;
    using typename super::pointer;
    using typename super::const_pointer;

  public:
  // 標準コンストラクタ/代入演算子およびデストラクタ
		Buf()
		    :_stride(0), _capacity(0), _p(super::null())
		{
		    _sizes.fill(0);
		}
		Buf(const Buf& b)
		    :_sizes(b._sizes), _stride(b._stride),
		     _capacity(b._capacity), _p(alloc(_capacity))
		{
		    copy<0>(b.begin(), size(), begin());
		}
    Buf&	operator =(const Buf& b)
		{
		    if (this != &b)
		    {
			resize(b._sizes, b._stride);
			copy<0>(b.begin(), size(), begin());
		    }
		    return *this;
		}
		Buf(Buf&& b)
		    :_sizes(b._sizes), _stride(b._stride),
		     _capacity(b._capacity), _p(b._p)
		{
		  // b の 破壊時に this->_p がdeleteされることを防ぐ．
		    b._p = super::null();
		}
    Buf&	operator =(Buf&& b)
		{
		    _sizes    = b._sizes;
		    _stride   = b._stride;
		    _capacity = b._capacity;
		    _p        = b._p;
		    
		  // b の 破壊時に this->_p がdeleteされることを防ぐ．
		    b._p = super::null();

		    return *this;
		}
		~Buf()
		{
		    free(_p, _capacity);
		}

  // 各軸のサイズと最終軸のストライドを指定したコンストラクタとリサイズ関数
    explicit	Buf(const sizes_type& sizes, size_t stride=0)
		    :_sizes(sizes),
		     _stride(stride ? stride : _sizes[D-1]),
		     _capacity(capacity(axis<0>())),
		     _p(alloc(_capacity))
		{
		}
    void	resize(const sizes_type& sizes, size_t stride=0)
		{
		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = (stride ? stride : _sizes[D-1]);
		    _capacity = capacity(axis<0>());
		    _p	      = alloc(_capacity);
		}

		Buf(pointer p, const sizes_type& sizes, size_t stride=0)
		    :_sizes(sizes),
		     _stride(stride ? stride : _sizes[D-1]),
		     _capacity(capacity(axis<0>())),
		     _allocator(p, _capacity),
		     _p(alloc(_capacity))
		{
		}

    const auto&	sizes()		const	{ return _sizes; }
    template <size_t I_=0>
    auto	size()		const	{ return _sizes[I_]; }
    template <size_t I_=D-1>
    auto	stride()	const	{ return stride_impl(axis<I_>()); }
    auto	nrow()		const	{ return _sizes[0]; }
    auto	ncol()		const	{ return _sizes[1]; }
    auto	data()			{ return _p; }
    auto	data()		const	{ return _p; }
    auto	begin()
		{
		    return make_iterator<SIZES...>(base_iterator(_p));
		}
    auto	begin() const
		{
		    return make_iterator<SIZES...>(const_base_iterator(_p));
		}
    auto	end()
		{
		    return make_iterator<SIZES...>(base_iterator(
						       _p + _capacity));
		}
    auto	end() const
		{
		    return make_iterator<SIZES...>(const_base_iterator(
						       _p + _capacity));
		}
    void	fill(const T& c)
		{
		    super::fill(_p, _capacity, c);
		}
    std::istream&
		get(std::istream& in)
		{
		    sizes_type	nvalues, sizes;
		    nvalues.fill(0);
		    sizes.fill(0);

		    get(in >> std::ws, nvalues, sizes);

		    return in;
		}
    
  private:
    template <size_t I_>
    auto	stride_impl(axis<I_>)		const	{ return _sizes[I_]; }
    auto	stride_impl(axis<D-1>)		const	{ return _stride; }

    size_t	capacity(axis<D-1>) const
		{
		    return _stride;
		}
    template <size_t I_>
    size_t	capacity(axis<I_>) const
		{
		    return _sizes[I_] * capacity(axis<I_+1>());
		}

    pointer	alloc(size_t siz)
		{
		    const auto	p = _allocator.allocate(siz);
		    for (pointer q = p, qe = q + siz; q != qe; ++q)
			_allocator.construct(q, value_type());
		    return p;
		}
    void	free(pointer p, size_t siz)
		{
		    if (p != super::null())
		    {
			for (pointer q = p, qe = q + siz; q != qe; ++q)
			    _allocator.destroy(q);
			_allocator.deallocate(p, siz);
		    }
		}

    template <class ITER_>
    static auto	make_iterator(ITER_ iter)
		{
		    return iter;
		}
    template <size_t SIZE_, size_t... SIZES_, class ITER_>
    auto	make_iterator(ITER_ iter) const
		{
		    constexpr size_t	I = D - 1 - sizeof...(SIZES_);
		    
		    return make_range_iterator(
			       make_iterator<SIZES_...>(iter),
			       stride<I>(), size<I>());
		}

    base_iterator
		get(std::istream& in, sizes_type& nvalues, sizes_type& sizes)
		{
		    constexpr size_t	BufSiz = (sizeof(value_type) < 2048 ?
						  2048/sizeof(value_type) : 1);
		    std::unique_ptr<value_type[]>
					tmp(new value_type[BufSiz]);
		    base_iterator	iter;
		    size_t		n = 0;
		    
		    for (size_t d = D - 1; n < BufSiz; )
		    {
			char	c;
			
			while (in.get(c))
			    if (!isspace(c) || c == '\n')
				break;
			
			if (in && c != '\n')	// 現在軸の末尾でなければ...
			{
			    in.putback(c);	// 1文字読み戻して
			    in >> tmp[n++];	// 改めて要素をバッファに読み込む

			    d = D - 1;		// 最下位軸に戻して
			    ++nvalues[d];	// 要素数を1だけ増やす
			}
			else			// 現在軸の末尾に到達したなら...
			{
			    if (nvalues[d] > sizes[d])
				sizes[d] = nvalues[d];	// 現在軸の要素数を記録

			    if (d == 0)		// 最上位軸の末尾ならば...
			    {
				resize(sizes);	// 領域を確保して
				iter = base_iterator(_p + _capacity);
				break;		// その末端をiterにセットして返す
			    }
		
			    nvalues[d] = 0;	// 現在軸を先頭に戻し
			    ++nvalues[--d];	// 直上軸に移動して1つ進める
			}
		    }

		    if (n == BufSiz)		// バッファが一杯ならば...
			iter = get(in, nvalues, sizes);	// 再帰してさらに読み込む

		    while (n--)
			*(--iter) = std::move(tmp[n]);	// バッファの内容を移す

		    return iter;		// 読み込まれた先頭位置を返す
		}

  private:
    sizes_type		_sizes;		//!< 各軸の要素数
    size_t		_stride;	//!< 最終軸のストライド
    size_t		_capacity;	//!< バッファ中に収めらる総要素数
    allocator_type	_allocator;	//!< 要素を確保するアロケータ
    pointer		_p;		//!< 先頭要素へのポインタ
};
    
/************************************************************************
*  class array<T, ALLOC, SIZE, SIZES...>				*
************************************************************************/
//! 多次元配列を表すクラス
/*!
  \param T	要素の型
  \param ALLOC	アロケータの型
  \param SIZE	最初の軸の要素数
  \param SIZES	2番目以降の各軸の要素数
*/
template <class T, class ALLOC, size_t SIZE, size_t... SIZES>
class array : public Buf<T, ALLOC, SIZE, SIZES...>
{
  private:
    using super	= Buf<T, ALLOC, SIZE, SIZES...>;
    using super::D;
    
  public:
    using typename super::sizes_type;
    using typename super::pointer;
    using typename super::const_pointer;
    using element_type		 = T;
    using iterator		 = decltype(std::declval<super*>()->begin());
    using const_iterator	 = decltype(std::declval<const super*>()
					    ->begin());
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
					       ::value_type;
    using const_value_type	 = typename std::iterator_traits<const_iterator>
					       ::value_type;
    using reference		 = typename std::iterator_traits<iterator>
					       ::reference;
    using const_reference	 = typename std::iterator_traits<const_iterator>
					       ::reference;
    
  public:
		array()				= default;
		array(const array&)		= default;
    array&	operator =(const array&)	= default;
		array(array&&)			= default;
    array&	operator =(array&&)		= default;
    
    template <class... SIZES_,
	      typename std::enable_if<sizeof...(SIZES_) == D>::type* = nullptr>
    explicit	array(SIZES_... sizes)
		    :super({to_size(sizes)...})
		{
		}
    template <class... SIZES_>
    typename std::enable_if<sizeof...(SIZES_) == D>::type
		resize(SIZES_... sizes)
		{
		    super::resize({to_size(sizes)...});
		}
    
    template <class... SIZES_,
	      typename std::enable_if<sizeof...(SIZES_) == D>::type* = nullptr>
    explicit	array(size_t unit, SIZES_... sizes)
		    :super({to_size(sizes)...}, to_stride(unit, sizes...))
		{
		}
    template <class... SIZES_>
    typename std::enable_if<sizeof...(SIZES_) == D>::type
		resize(size_t unit, SIZES_... sizes)
		{
		    super::resize({to_size(sizes)...},
				  to_stride(unit, sizes...));
		}

    template <class E_,
	      typename std::enable_if<rank<E_>() == D>::type* = nullptr>
		array(const E_& expr)
		    :super(sizes(expr, std::make_index_sequence<D>()))
		{
		    constexpr size_t	S = detail::max<size0(),
							TU::size0<E_>()>::value;
		    copy<S>(std::begin(expr), size(), begin());
		}
    template <class E_>
    typename std::enable_if<rank<E_>() == D, array&>::type
		operator =(const E_& expr)
		{
		    super::resize(sizes(expr, std::make_index_sequence<D>()));
		    constexpr size_t	S = detail::max<size0(),
							TU::size0<E_>()>::value;
		    copy<S>(std::begin(expr), size(), begin());

		    return *this;
		}

		array(std::initializer_list<const_value_type> args)
		    :super(sizes(args))
		{
		    copy<size0()>(args.begin(), size(), begin());
		}
    array&	operator =(std::initializer_list<const_value_type> args)
		{
		    super::resize(sizes(args));
		    copy<size0()>(args.begin(), size(), begin());

		    return *this;
		}

    template <class... SIZES_,
	      typename std::enable_if<sizeof...(SIZES_) == D>::type* = nullptr>
    explicit	array(pointer p, SIZES_... sizes)
		    :super(p, {to_size(sizes)...})
		{
		}

    template <class... SIZES_,
	      typename std::enable_if<sizeof...(SIZES_) == D>::type* = nullptr>
    explicit	array(pointer p, size_t unit, SIZES_... sizes)
		    :super(p, {to_size(sizes)...}, to_stride(unit, sizes...))
		{
		}

    template <class ALLOC_>
    void	write(array<T, ALLOC_, SIZE, SIZES...>& a) const
		{
		    a.resize(sizes(), a.stride());
		    super::copy(begin(), size(), a.begin());
		}

    using	super::size;
    using	super::stride;
    using	super::nrow;
    using	super::ncol;
    using	super::data;
    using	super::begin;
    using	super::end;


    template <class... IS_>
    auto	operator ()(IS_... is)
		{
		    return TU::slice(*this, is...);
		}
    template <class... IS_>
    auto	operator ()(IS_... is) const
		{
		    return TU::slice(*this, is...);
		}
    template <size_t SIZE_, size_t... SIZES_, class... INDICES_,
	      typename std::enable_if<sizeof...(SIZES_) + 1 ==
				      sizeof...(INDICES_)>::type* = nullptr>
    auto	slice(INDICES_... indices)
		{
		    return TU::slice<SIZE_, SIZES_...>(*this, indices...);
		}
    template <size_t SIZE_, size_t... SIZES_, class... INDICES_,
	      typename std::enable_if<sizeof...(SIZES_) + 1 ==
				      sizeof...(INDICES_)>::type* = nullptr>
    auto	slice(INDICES_... indices) const
		{
		    return TU::slice<SIZE_, SIZES_...>(*this, indices...);
		}

    constexpr static
    size_t	size0()		{ return SIZE; }
    auto	cbegin()  const	{ return begin(); }
    auto	cend()	  const	{ return end(); }
    auto	rbegin()	{ return reverse_iterator(end()); }
    auto	rbegin()  const	{ return const_reverse_iterator(end()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	rend()		{ return reverse_iterator(begin()); }
    auto	rend()	  const	{ return const_reverse_iterator(begin()); }
    auto	crend()	  const	{ return rend(); }
    reference	operator [](size_t i)
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    const_reference
		operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    void	fill(const element_type& val)
		{
		    super::fill(val);
		}
    std::istream&
		restore(std::istream& in)
		{
		    restore(in, begin(), size());
		    return in;
		}
    std::ostream&
		save(std::ostream& out) const
		{
		    save(out, begin(), size());
		    return out;
		}
    
  private:
    using	sizes_iterator = typename sizes_type::iterator;
    
    template <class T_>
    static typename std::enable_if<std::is_integral<T_>::value, size_t>::type
		to_size(const T_& arg)
		{
		    return size_t(arg);
		}

    template <class E_, size_t... I_>
    static sizes_type
		sizes(const E_& expr, std::index_sequence<I_...>)
		{
		    return {TU::size<I_>(expr)...};
		}

    template <class T_>
    static typename std::enable_if<rank<T_>() == 0>::type
		set_sizes(sizes_iterator iter, sizes_iterator end, const T_& val)
		{
		    throw std::runtime_error("array<BUF>::set_sizes(): too shallow initializer list!");
		}
    template <class T_>
    static typename std::enable_if<rank<T_>() != 0>::type
		set_sizes(sizes_iterator iter, sizes_iterator end, const T_& r)
		{
		    *iter = r.size();
		    if (++iter != end)
			set_sizes(iter, end, *r.begin());
		}
    static sizes_type
		sizes(std::initializer_list<const_value_type> args)
		{
		    sizes_type	sizs;
		    set_sizes(sizs.begin(), sizs.end(), args);

		    return sizs;
		}
    
    static auto	to_stride(size_t unit, size_t size)
		{
		    constexpr auto	elmsiz = sizeof(element_type);

		    if (unit == 0)
			unit = 1;
		    const auto	n = lcm(elmsiz, unit)/elmsiz;

		    return n*((size + n - 1)/n);
		}
    template <class... SIZES_>
    static auto	to_stride(size_t unit, size_t size, SIZES_... sizes)
		{
		    return to_stride(unit, sizes...);
		}

    static void	restore(std::istream& in, pointer begin, size_t n)
		{
		    in.read(reinterpret_cast<char*>(begin),
			    sizeof(element_type) * n);
		}
    template <class ITER_>
    static void	restore(std::istream& in, ITER_ begin, size_t n)
		{
		    for (size_t i = 0; i != n; ++i, ++begin)
			restore(in, begin->begin(), begin->size());
		}

    static void	save(std::ostream& out, const_pointer begin, size_t n)
		{
		    out.write(reinterpret_cast<const char*>(begin),
			      sizeof(element_type) * n);
		}
    template <class ITER_>
    static void	save(std::ostream& out, ITER_ begin, size_t n)
		{
		    for (size_t i = 0; i != n; ++i, ++begin)
			save(out, begin->begin(), begin->size());
		}
};

//! 多次元配列の指定された軸の要素数を返す
/*!
  軸はテンプレートパラメータ I で指定する
  \param a	多次元配列
  \return	第 I 軸の要素数
 */
template <size_t I, class T, class ALLOC, size_t SIZE, size_t... SIZES>
inline size_t
size(const array<T, ALLOC, SIZE, SIZES...>& a)
{
    return a.template size<I>();
}

//! 多次元配列の指定された軸のストライドを返す
/*!
  軸はテンプレートパラメータ I で指定する
  \param a	多次元配列
  \return	第 I 軸のストライド
 */
template <size_t I, class T, class ALLOC, size_t SIZE, size_t... SIZES>
inline size_t
stride(const array<T, ALLOC, SIZE, SIZES...>& a)
{
    return a.template stride<I>();
}

//! 出力ストリームへ配列を書き出し(ASCII)，さらに改行コードを出力する．
/*!
  \param out	出力ストリーム
  \param a	書き出す配列
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC, size_t SIZE, size_t... SIZES> std::ostream&
operator <<(std::ostream& out, const array<T, ALLOC, SIZE, SIZES...>& a)
{
    for (const auto& val : a)
	out << ' ' << val;
    return out << std::endl;
}
    
//! 入力ストリームから配列を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param a	配列の読み込み先
  \return	inで指定した入力ストリーム
*/
template <class T, class ALLOC, size_t SIZE, size_t... SIZES>
inline std::istream&
operator >>(std::istream& in, array<T, ALLOC, SIZE, SIZES...>& a)
{
    return a.get(in);
}

/************************************************************************
*  type definitions for convenience					*
************************************************************************/
template <class T, size_t N=0, class ALLOC=std::allocator<T> >
using Array = array<T, ALLOC, N>;				//!< 1次元配列

template <class T, size_t R=0, size_t C=0, class ALLOC=std::allocator<T> >
using Array2 = array<T, ALLOC, R, C>;				//!< 2次元配列

template <class T,
	  size_t Z=0, size_t Y=0, size_t X=0, class ALLOC=std::allocator<T> >
using Array3 = array<T, ALLOC, Z, Y, X>;			//!< 3次元配列

/************************************************************************
*  evaluation of opnodes						*
************************************************************************/
namespace detail
{
  template <class E, bool=is_opnode<E>::value>
  struct result_t
  {
      using type = E;		// E が opnode でなければ E そのもの
  };
  template <class E>
  struct result_t<E, true>
  {
    private:
      template <class T_, size_t SIZE_>
      struct array_t
      {
	  using type = array<T_, std::allocator<T_>, SIZE_>;
      };
      template <class T_, size_t SIZE_, size_t... SIZES_>
      struct array_t<array<T_, std::allocator<T_>, SIZES_...>, SIZE_>
      {
	  using type = array<T_, std::allocator<T_>, SIZE_, SIZES_...>;
      };

    public:
      using type = typename array_t<typename result_t<TU::value_t<E> >::type,
				    size0<E>()>::type;
  };
}	// namespace detail

//! 演算子の評価結果の型を返す
/*!
  Eの型が演算子ならばその評価結果である配列の型を，そうでなければEそのものを返す
  \param E	配列式の型
*/
template <class E>
using result_t	= typename detail::result_t<E>::type;

//! 配列式の評価結果を返す
/*!
  \param expr	配列式
  \return	exprが演算子ならばその評価結果である配列を，そうでなければ
		expr自体の参照を返す
*/
template <class E>
inline typename std::conditional<detail::is_opnode<E>::value,
				 result_t<E>, const E&>::type
evaluate(const E& expr)
{
    return expr;
}
    
/************************************************************************
*  products of two ranges						*
************************************************************************/
namespace detail
{
  //! 2つの配列式に対する積演算子を表すクラス
  /*!
    \param OP	積演算子の型
    \param L	積演算子の第1引数となる式の型
    \param R	積演算子の第2引数となる式の型
   */
  template <class OP, class L, class R>
  class product_opnode : public opnode
  {
    private:
      class binder2nd
      {
	private:
	// 右辺が opnode の場合：その評価結果
	// 右辺が opnode でない場合：
	//	右辺が range<ITER, SIZE> に変換可能ならば右辺そのもの
	//	そうでなければそれへの定数参照
	  using cache_t	= typename std::conditional<
			      is_opnode<R>::value || is_range<R>::value,
			      const TU::result_t<R>, const R&>::type;

	public:
		binder2nd(OP op, const R& r)
		    :_r(evaluate(r)), _op(op) 		{}

	  template <class T_>
	  auto	operator ()(const T_& arg)	const	{ return _op(arg, _r); }

	private:
	  cache_t	_r;	// 評価後に固定された第2引数
	  const OP	_op;
      };

    public:
    // transform_iterator への第1テンプレートパラメータを，binder2nd そのもの
    // ではなく，それへの定数参照とすることにより cache のコピーを防ぐ
      using iterator = boost::transform_iterator<const binder2nd&,
						 const_iterator_t<L> >;
      
    public:
		product_opnode(const L& l, const R& r, OP op)
		    :_l(l), _binder(op, r)				{}

      constexpr static size_t
		size0()		{ return TU::size0<L>(); }
      iterator	begin()	const	{ return {std::begin(_l), _binder}; }
      iterator	end()	const	{ return {std::end(_l),   _binder}; }
      size_t	size()	const	{ return std::size(_l); }
      auto	operator [](size_t i) const
		{
		    return *(begin() + i);
		}

    private:
      argument_t<L>	_l;
      const binder2nd	_binder;
  };

  template <class OP, class L, class R> inline product_opnode<OP, L, R>
  make_product_opnode(const L& l, const R& r, OP op)
  {
      return {l, r, op};
  }

  struct bit_xor
  {
      template <class X_, class Y_>
      auto	operator ()(const X_& x, const Y_& y) const
		{
		    return x ^ y;
		}
  };
}	// namespace detail

//! 2つの1次元配列式の内積をとる.
/*!
  演算子ノードではなく，評価結果のスカラー値が返される.
  \param l	左辺の1次元配列式
  \param r	右辺の1次元配列式
  \return	内積の評価結果
*/
template <class L, class R,
	  typename std::enable_if<rank<L>() == 1 &&
				  rank<R>() == 1>::type* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    using value_type = typename std::common_type<value_t<L>, value_t<R> >::type;

    assert(size<0>(l) == size<0>(r));
    constexpr size_t	S = detail::max<size0<L>(),size0<R>()>::value;
    return inner_product<S>(std::begin(l), std::size(l), std::begin(r),
			    value_type(0));
}

//! 2次元配列式と1または2次元配列式の積をとる.
/*!
  \param l	左辺の2次元配列式
  \param r	右辺の1または2次元配列式
  \return	積を表す演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<rank<L>() == 2 &&
				  rank<R>() >= 1 &&
				  rank<R>() <= 2>::type* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    return detail::make_product_opnode(l, r, [](const auto& x, const auto& y)
					     { return x * y; });
}

//! 1次元配列式と2次元配列式の積をとる.
/*!
  \param l	左辺の1次元配列式
  \param r	右辺の2次元配列式
  \return	積を表す演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<rank<L>() == 1 &&
				  rank<R>() == 2>::type* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    constexpr size_t	S = size0<value_t<R> >();
    return detail::make_product_opnode(
      	       make_range<S>(column_begin(r), size<1>(r)), l,
	       [](const auto& x, const auto& y){ return x * y; });
}
    
//! 2つの1次元配列式の外積をとる.
/*!
  \param l	左辺の1次元配列式
  \param r	右辺の1次元配列式
  \return	外積を表す演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<rank<L>() == 1 &&
				  rank<R>() == 1>::type* = nullptr>
inline auto
operator %(const L& l, const R& r)
{
    return detail::make_product_opnode(l, r, [](const auto& x, const auto& y)
					     { return x * y; });
}

//! 2つの1次元配列式のベクトル積をとる.
/*!
  演算子ノードではなく，評価結果の3次元配列が返される.
  \param l	左辺の1次元配列式
  \param r	右辺の1次元配列式
  \return	ベクトル積
*/
template <class L, class R>
inline typename std::enable_if<
	   rank<L>() == 1 && rank<R>() == 1,
	   Array<typename std::common_type<element_t<L>, element_t<R> >::type,
		 3> >::type
operator ^(const L& l, const R& r)
{
#ifdef TU_DEBUG
    std::cout << "operator ^ [" << print_sizes(l) << ']' << std::endl;
#endif
    assert(size<0>(l) == 3 && size<0>(r) == 3);
    
    const auto&	el = evaluate(l);
    const auto&	er = evaluate(r);

    return {el[1] * er[2] - el[2] * er[1],
	    el[2] * er[0] - el[0] * er[2],
	    el[0] * er[1] - el[1] * er[0]};
}

//! 2次元配列式の各行と1次元配列式のベクトル積をとる.
/*!
  \param l	左辺の2次元配列式
  \param r	右辺の1次元配列式
  \return	ベクトル積を表す演算子ノード
*/
template <class L, class R,
	  typename std::enable_if<rank<L>() == 2 &&
				  rank<R>() == 1>::type* = nullptr>
inline auto
operator ^(const L& l, const R& r)
{
    return detail::make_product_opnode(l, r, detail::bit_xor());
}

}	// namespace TU
#endif	// !__TU_ARRAY_H
