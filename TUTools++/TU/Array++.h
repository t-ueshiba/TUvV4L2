/*!
  \file		Array++.h
  \author	Toshio UESHIBA
  \brief	多次元配列クラスの定義と実装
*/
#ifndef TU_ARRAY_H
#define TU_ARRAY_H

#include <array>
#include <iomanip>		// for std::ws
#include <memory>		// for std::allocator<T>, std::unique_ptr<T>
#include "TU/range.h"

namespace TU
{
/************************************************************************
*  class BufTraits<T>							*
************************************************************************/
template <class T, class ALLOC>
class BufTraits : public std::allocator_traits<ALLOC>
{
  private:
    using super			= std::allocator_traits<ALLOC>;

  public:
    using iterator		= typename super::pointer;
    using const_iterator	= typename super::const_pointer;
    
  protected:
    using			typename super::pointer;

    static auto null()		{ return nullptr; }
    static auto ptr(pointer p)	{ return p; }
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
    constexpr static size_t	Dimension = 1 + sizeof...(SIZES);

    using sizes_type		= std::array<size_t, Dimension>;
    using			typename super::value_type;
    using			typename super::pointer;
    using			typename super::const_pointer;

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
		    if (!check_sizes(sizes, axis<Dimension>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::Buf(): mismatched size!");
		    init(typename std::is_arithmetic<value_type>::type());
		}
    bool	resize(const sizes_type& sizes, size_t=0)
		{
		    if (!check_sizes(sizes, axis<Dimension>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::resize(): mismatched size!");
		    return false;
		}

    template <size_t I_=0>
    constexpr static auto	size()		{ return siz<I_>::value; }
    template <size_t I_=Dimension-1>
    constexpr static ptrdiff_t	stride()	{ return siz<I_>::value; }
    constexpr static auto	nrow()		{ return siz<0>::value; }
    constexpr static auto	ncol()		{ return siz<1>::value; }
    constexpr static auto	capacity()	{ return Capacity; }

    auto	data()		{ return _a.data(); }
    auto	data()	const	{ return _a.data(); }
    auto	begin()		{ return make_iterator<SIZES...>(_a.begin()); }
    auto	begin()	const	{ return make_iterator<SIZES...>(_a.begin()); }
    auto	end()		{ return make_iterator<SIZES...>(_a.end()); }
    auto	end()	const	{ return make_iterator<SIZES...>(_a.end()); }

    std::istream&
		get(std::istream& in)
		{
		    for (auto& val : _a)
			in >> val;
		    return in;
		}

    friend bool	operator ==(const Buf& a, const Buf& b)
		{
		    return a._a == b._a;
		}
    friend bool	operator !=(const Buf& a, const Buf& b)
		{
		    return a._a != b._a;
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
  //alignas(sizeof(T)) std::array<T, Capacity>	_a;
    std::array<T, Capacity>	_a;
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
    constexpr static size_t	Dimension = 1 + sizeof...(SIZES);

    using sizes_type		= std::array<size_t, Dimension>;
    using			typename super::value_type;
    using			typename super::pointer;
    using			typename super::const_pointer;
    using			typename super::allocator_type;

  public:
  // 標準コンストラクタ/代入演算子およびデストラクタ
		Buf()
		    :_stride(0), _capacity(0), _ext(false), _p(super::null())
		{
		    _sizes.fill(0);
		}
		Buf(const Buf& b)
		    :_sizes(b._sizes), _stride(b._stride),
		     _capacity(b._capacity), _ext(false), _p(alloc(_capacity))
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
		     _capacity(b._capacity), _ext(b._ext), _p(b._p)
		{
		  // b の 破壊時に this->_p がdeleteされることを防ぐ．
		    b._p = super::null();
		}
    Buf&	operator =(Buf&& b)
		{
		    _sizes    = b._sizes;
		    _stride   = b._stride;
		    _capacity = b._capacity;
		    _ext      = b._ext;
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
    explicit	Buf(const sizes_type& sizes, ptrdiff_t stride=0)
		    :_sizes(sizes),
		     _stride(stride ? stride : _sizes[Dimension-1]),
		     _capacity(capacity(axis<0>())),
		     _ext(false),
		     _p(alloc(_capacity))
		{
		}
    bool	resize(const sizes_type& sizes, ptrdiff_t stride=0)
		{
		    if (stride == 0)
			stride = sizes[Dimension-1];
		    
		    if ((stride == _stride) && (sizes == _sizes))
			return false;

		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = stride;
		    _capacity = capacity(axis<0>());
		    _ext      = false;
		    _p	      = alloc(_capacity);

		    return true;
		}

  // 外部記憶領域および各軸のサイズと最終軸のストライドを指定したコンストラクタと
  // リサイズ関数
    explicit	Buf(pointer p, const sizes_type& sizes, ptrdiff_t stride=0)
		    :_sizes(sizes),
		     _stride(stride ? stride : _sizes[Dimension-1]),
		     _capacity(capacity(axis<0>())),
		     _ext(true),
		     _p(p)
		{
		}
    void	resize(pointer p, const sizes_type& sizes, ptrdiff_t stride=0)
		{
		    if (stride == 0)
			stride = sizes[Dimension-1];
		    
		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = stride;
		    _capacity = capacity(axis<0>());
		    _ext      = true;
		    _p	      = p;
		}

    const auto&	sizes()		const	{ return _sizes; }
    template <size_t I_=0>
    auto	size()		const	{ return _sizes[I_]; }
    template <size_t I_=Dimension-1>
    ptrdiff_t	stride()	const	{ return stride_impl(axis<I_>()); }
    auto	capacity()	const	{ return _capacity; }
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
    ptrdiff_t	stride_impl(axis<I_>)		const	{ return _sizes[I_]; }
    ptrdiff_t	stride_impl(axis<Dimension-1>)	const	{ return _stride; }

    size_t	capacity(axis<Dimension-1>) const
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
		    const auto	p = super::allocate(_allocator, siz);
		    for (pointer q = p, qe = q + siz; q != qe; ++q)
			super::construct(_allocator,
					 super::ptr(q), value_type());
		    return p;
		}
    void	free(pointer p, size_t siz)
		{
		    if (!_ext && p != super::null())
		    {
			for (pointer q = p, qe = q + siz; q != qe; ++q)
			    super::destroy(_allocator, super::ptr(q));
			super::deallocate(_allocator, p, siz);
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
		    constexpr size_t	I = Dimension - 1 - sizeof...(SIZES_);
		    
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
		    
		    for (size_t d = Dimension - 1; n < BufSiz; )
		    {
			char	c;
			
			while (in.get(c))
			    if (!isspace(c) || c == '\n')
				break;
			
			if (in && c != '\n')	// 現在軸の末尾でなければ...
			{
			    in.putback(c);	// 1文字読み戻して
			    in >> tmp[n++];	// 改めて要素をバッファに読み込む

			    d = Dimension - 1;	// 最下位軸に戻して
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
    ptrdiff_t		_stride;	//!< 最終軸のストライド
    size_t		_capacity;	//!< バッファ中に収めらる総要素数
    allocator_type	_allocator;	//!< 要素を確保するアロケータ
    bool		_ext;		//!< _p が外部記憶領域なら true
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
    using super::Dimension;
    
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
    using value_type		 = iterator_value<iterator>;
    using const_value_type	 = iterator_value<const_iterator>;
    using reference		 = iterator_reference<iterator>;
    using const_reference	 = iterator_reference<const_iterator>;
    
  public:
		array()				= default;
		array(const array&)		= default;
    array&	operator =(const array&)	= default;
		array(array&&)			= default;
    array&	operator =(array&&)		= default;


    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == Dimension &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(SIZES_... sizes)
		    :super({to_size(sizes)...})
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == Dimension &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value, bool>
		resize(SIZES_... sizes)
		{
		    return super::resize({to_size(sizes)...});
		}
    
    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == Dimension &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(size_t unit, SIZES_... sizes)
		    :super({to_size(sizes)...}, to_stride(unit, sizes...))
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == Dimension &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value, bool>
		resize(size_t unit, SIZES_... sizes)
		{
		    return super::resize({to_size(sizes)...},
					 to_stride(unit, sizes...));
		}

    template <class E_,
	      std::enable_if_t<rank<E_>() == Dimension + rank<T>()>* = nullptr>
		array(const E_& expr)
		    :super(sizes(expr, std::make_index_sequence<Dimension>()))
		{
		    constexpr size_t	S = detail::max<size0(),
							TU::size0<E_>()>::value;
		    copy<S>(std::begin(expr), size(), begin());
		}
    template <class E_>
    std::enable_if_t<rank<E_>() == Dimension + rank<T>(), array&>
		operator =(const E_& expr)
		{
		    super::resize(sizes(expr,
					std::make_index_sequence<Dimension>()));
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
	      std::enable_if_t<
		  sizeof...(SIZES_) == Dimension &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(pointer p, SIZES_... sizes)
		    :super(p, {to_size(sizes)...})
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == Dimension &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value>
		resize(pointer p, SIZES_... sizes)
		{
		    super::resize(p, {to_size(sizes)...});
		}
	    
    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == Dimension &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(pointer p, size_t unit, SIZES_... sizes)
		    :super(p, {to_size(sizes)...}, to_stride(unit, sizes...))
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == Dimension &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value>
		resize(pointer p, size_t unit, SIZES_... sizes)
		{
		    super::resize(p, {to_size(sizes)...},
				  to_stride(unit, sizes...));
		}
	    
    template <class T_> std::enable_if_t<rank<T_>() == 0, array&>
		operator =(const T_& c)
		{
		    TU::fill<size0()>(begin(), size(), c);

		    return *this;
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
	      std::enable_if_t<sizeof...(SIZES_) + 1 ==
			       sizeof...(INDICES_)>* = nullptr>
    auto	slice(INDICES_... indices)
		{
		    return TU::slice<SIZE_, SIZES_...>(*this, indices...);
		}
    template <size_t SIZE_, size_t... SIZES_, class... INDICES_,
	      std::enable_if_t<sizeof...(SIZES_) + 1 ==
			       sizeof...(INDICES_)>* = nullptr>
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
    std::ostream&
		put(std::ostream& out) const
		{
		    for (const auto& val : *this)
			out << ' ' << val;
		    return out;
		}
    
  private:
    using	sizes_iterator = typename sizes_type::iterator;
    
    template <class T_>
    static std::enable_if_t<std::is_integral<T_>::value, size_t>
		to_size(const T_& arg)
		{
		    return size_t(arg);
		}

    template <class E_, size_t... I_>
    static sizes_type
		sizes(const E_& expr, std::index_sequence<I_...>)
		{
		    return {{TU::size<I_>(expr)...}};
		}

    template <class T_>
    static std::enable_if_t<rank<T_>() == 0>
		set_sizes(sizes_iterator iter, sizes_iterator end, const T_& val)
		{
		    throw std::runtime_error("array<BUF>::set_sizes(): too shallow initializer list!");
		}
    template <class T_>
    static std::enable_if_t<rank<T_>() != 0>
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
    
    static ptrdiff_t
		to_stride(size_t unit, size_t size)
		{
		    constexpr auto	elmsiz = sizeof(element_type);

		    if (unit == 0)
			unit = 1;
		    const auto	n = lcm(elmsiz, unit)/elmsiz;

		    return n*((size + n - 1)/n);
		}
    template <class... SIZES_>
    static ptrdiff_t
		to_stride(size_t unit, size_t size, SIZES_... sizes)
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

//! 出力ストリームへ配列を書き出し(ASCII)，さらに改行コードを出力する．
/*!
  \param out	出力ストリーム
  \param a	書き出す配列
  \return	outで指定した出力ストリーム
*/
template <class T, class ALLOC, size_t SIZE, size_t... SIZES>
inline std::ostream&
operator <<(std::ostream& out, const array<T, ALLOC, SIZE, SIZES...>& a)
{
    return a.put(out) << std::endl;
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

template <class T, class ALLOC, size_t SIZE, size_t... SIZES> inline auto
serialize(const array<T, ALLOC, SIZE, SIZES...>& a)
{
    return make_range(a.data(), a.capacity());
}

/************************************************************************
*  type alias: replace_element<S, T>					*
************************************************************************/
namespace detail
{
// array<S, std::allocator<S>, SIZE, SIZES...> のSをTに置換する
  template <class S, size_t SIZE, size_t... SIZES, class T>
  struct replace_element<array<S, std::allocator<S>, SIZE, SIZES...>, T>
  {
    private:
      using U	 = typename replace_element<S, T>::type;

    public:
      using type = array<U, std::allocator<U>, SIZE, SIZES...>;
  };
}	// namespace detail
    
/************************************************************************
*  substance_t<E, PRED>							*
************************************************************************/
namespace detail
{
  template <class E, template <class> class PRED, bool=PRED<E>::value>
  struct substance_t
  {
      using type = E;		// PRED<E>::value == false ならば E そのもの
  };
  template <class E, template <class> class PRED>
  struct substance_t<E, PRED, true>
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

      using E1	 = typename substance_t<TU::value_t<E>, PRED>::type;
      
    public:
      using type = typename array_t<E1,
				    (size0<E1>() == 0 ? 0 : size0<E>())>::type;
  };
  template <class... E, template <class> class PRED>
  struct substance_t<std::tuple<E...>, PRED, false>
  {
      using type = std::tuple<typename substance_t<E, PRED>::type...>;
  };
}	// namespace detail

//! 反復子が指す型. ただし，それがrangeならば，そのrangeが表現する配列型
template <class ITER>
using iterator_substance
	  = typename detail::substance_t<decayed_iterator_value<ITER>,
					 detail::is_range>::type;

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

}	// namespace TU
#endif	// !TU_ARRAY_H
