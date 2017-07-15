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

    constexpr static size_t	Alignment = 0;

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
    using super			= BufTraits<T, ALLOC>;
    using base_iterator		= typename super::iterator;
    using const_base_iterator	= typename super::const_iterator;

  // このバッファの総容量をコンパイル時に計算
    template <size_t SIZE_, size_t... SIZES_>
    struct cap
    {
	constexpr static size_t	value = SIZE_ * cap<SIZES_...>::value;
    };
    template <size_t SIZE_>
    struct cap<SIZE_>
    {
      private:
	constexpr static auto	n = lcm(sizeof(T),
					super::Alignment ? super::Alignment : 1)
				  / sizeof(T);

      public:
	constexpr static size_t value = n*((SIZE_ + n - 1)/n);
    };

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

  public:
    constexpr static size_t	rank()	{ return 1 + sizeof...(SIZES); }

    using sizes_type		= std::array<size_t, rank()>;
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
   		    if (!check_sizes(sizes, axis<rank()>()))
   			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::Buf(): mismatched size!");
   		    init(typename std::is_arithmetic<value_type>::type());
   		}
    bool	resize(const sizes_type& sizes, size_t=0)
   		{
   		    if (!check_sizes(sizes, axis<rank()>()))
   			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::resize(): mismatched size!");
   		    return false;
   		}

    template <size_t I_=0>
    constexpr static auto	size()		{ return siz<I_>::value; }
    constexpr static ptrdiff_t	stride()
		   		{
				    return cap<size<rank()-1>()>::value;
				}
    constexpr static auto	nrow()		{ return siz<0>::value; }
    constexpr static auto	ncol()		{ return siz<1>::value; }
    constexpr static auto	capacity()
				{
				    return cap<SIZE, SIZES...>::value;
				}
    auto	data()		{ return _a.data(); }
    auto	data()	const	{ return _a.data(); }
    auto	begin()
		{
		    return make_iterator<SIZES...>(base_iterator(data()));
		}
    auto	begin() const
		{
		    return make_iterator<SIZES...>(const_base_iterator(data()));
		}
    auto	end()
		{
		    return make_iterator<SIZES...>(base_iterator(
						       data() + capacity()));
		}
    auto	end() const
		{
		    return make_iterator<SIZES...>(const_base_iterator(
						       data() + capacity()));
		}

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
		    constexpr ptrdiff_t
			STRIDE = (sizeof...(SIZES_) ? SIZE_ : stride());

		    return make_range_iterator<STRIDE, SIZE_>(
			       make_iterator<SIZES_...>(iter));
		}

  private:
#if !defined(__GNUG__)	// g++ には alignas(0) がエラーになるバグがある
    alignas(super::Alignment)
#endif
    std::array<T, capacity()>	_a;
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
    constexpr static size_t	rank()	{ return 1 + sizeof...(SIZES); }

    using sizes_type		= std::array<size_t, rank()>;
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
		Buf(Buf&& b) noexcept
		    :_sizes(b._sizes), _stride(b._stride),
		     _capacity(b._capacity), _ext(b._ext), _p(b._p)
		{
		  // b の 破壊時に this->_p がdeleteされることを防ぐ．
		    b._p = super::null();
		}
    Buf&	operator =(Buf&& b) noexcept
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
    explicit	Buf(const sizes_type& sizes, size_t alignment)
		    :_sizes(sizes),
		     _stride(to_stride(alignment, _sizes[rank()-1])),
		     _capacity(capacity_of(axis<0>())),
		     _ext(false),
		     _p(alloc(_capacity))
		{
		}
    bool	resize(const sizes_type& sizes, size_t alignment)
		{
		    const auto	stride = to_stride(alignment, sizes[rank()-1]);
		    
		    if ((stride == _stride) && (sizes == _sizes))
			return false;

		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = stride;
		    _capacity = capacity_of(axis<0>());
		    _ext      = false;
		    _p	      = alloc(_capacity);

		    return true;
		}

  // 外部記憶領域および各軸のサイズと最終軸のストライドを指定したコンストラクタと
  // リサイズ関数
    explicit	Buf(pointer p, const sizes_type& sizes, size_t alignment)
		    :_sizes(sizes),
		     _stride(to_stride(alignment, _sizes[rank()-1])),
		     _capacity(capacity_of(axis<0>())),
		     _ext(true),
		     _p(p)
		{
		}
    void	resize(pointer p, const sizes_type& sizes, size_t alignment)
		{
		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = to_stride(alignment, _sizes[rank()-1]);
		    _capacity = capacity_of(axis<0>());
		    _ext      = true;
		    _p	      = p;
		}

    const auto&	sizes()		const	{ return _sizes; }
    template <size_t I_=0>
    auto	size()		const	{ return _sizes[I_]; }
    auto	stride()	const	{ return _stride; }
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
    static ptrdiff_t
		to_stride(size_t alignment, size_t size)
		{
		    if (rank() == 1)
			return size;
		    
		    constexpr auto	elmsiz = sizeof(T);

		    if (alignment == 0)
			alignment = (super::Alignment ? super::Alignment : 1);
		    const auto	n = lcm(elmsiz, alignment)/elmsiz;

		    return n*((size + n - 1)/n);
		}

    ptrdiff_t	stride_of(axis<rank()-1>)	const	{ return _stride; }
    template <size_t I_>
    ptrdiff_t	stride_of(axis<I_>)		const	{ return _sizes[I_]; }

    size_t	capacity_of(axis<rank()-1>) const
		{
		    return _stride;
		}
    template <size_t I_>
    size_t	capacity_of(axis<I_>) const
		{
		    return _sizes[I_] * capacity_of(axis<I_+1>());
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
		    constexpr size_t	I = rank() - 1 - sizeof...(SIZES_);

		    return make_range_iterator(
			       make_iterator<SIZES_...>(iter),
			       stride_of(axis<I>()), size<I>());
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
		    
		    for (size_t d = rank() - 1; n < BufSiz; )
		    {
			char	c;
			
			while (in.get(c))
			    if (!isspace(c) || c == '\n')
				break;
			
			if (in && c != '\n')	// 現在軸の末尾でなければ...
			{
			    in.putback(c);	// 1文字読み戻して
			    in >> tmp[n++];	// 改めて要素をバッファに読み込む

			    d = rank() - 1;	// 最下位軸に戻して
			    ++nvalues[d];	// 要素数を1だけ増やす
			}
			else			// 現在軸の末尾に到達したなら...
			{
			    if (nvalues[d] > sizes[d])
				sizes[d] = nvalues[d];	// 現在軸の要素数を記録

			    if (d == 0)		// 最上位軸の末尾ならば...
			    {			// 領域を確保して
				resize(sizes, super::Alignment);
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

    using	super::rank;
    using	super::size;
    using	super::stride;
    using	super::nrow;
    using	super::ncol;
    using	super::data;
    using	super::begin;
    using	super::end;

    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == rank() &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(SIZES_... sizes)
		    :super({to_size(sizes)...}, super::Alignment)
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == rank() &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value, bool>
		resize(SIZES_... sizes)
		{
		    return super::resize({to_size(sizes)...}, super::Alignment);
		}
    
    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == rank() &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(size_t alignment, SIZES_... sizes)
		    :super({to_size(sizes)...}, alignment)
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == rank() &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value, bool>
		resize(size_t alignment, SIZES_... sizes)
		{
		    return super::resize({to_size(sizes)...}, alignment);
		}

    template <class E_,
	      std::enable_if_t<TU::rank<E_>() == rank() + TU::rank<T>()>*
	      = nullptr>
		array(const E_& expr)
		    :super(sizes(expr, std::make_index_sequence<rank()>()),
			   super::Alignment)
		{
		    constexpr auto	S = detail::max<size0(),
							TU::size0<E_>()>::value;
		    copy<S>(std::cbegin(expr), size(), begin());
		}
    template <class E_>
    std::enable_if_t<TU::rank<E_>() == rank() + TU::rank<T>(), array&>
		operator =(const E_& expr)
		{
		    super::resize(sizes(expr,
					std::make_index_sequence<rank()>()),
				  super::Alignment);
		    constexpr auto	S = detail::max<size0(),
							TU::size0<E_>()>::value;
		    copy<S>(std::cbegin(expr), size(), begin());

		    return *this;
		}

		array(std::initializer_list<const_value_type> args)
		    :super(sizes(args), super::Alignment)
		{
		    copy<size0()>(args.begin(), size(), begin());
		}
    array&	operator =(std::initializer_list<const_value_type> args)
		{
		    super::resize(sizes(args), super::Alignment);
		    copy<size0()>(args.begin(), size(), begin());

		    return *this;
		}

    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == rank() &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(pointer p, SIZES_... sizes)
		    :super(p, {to_size(sizes)...}, super::Alignment)
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == rank() &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value>
		resize(pointer p, SIZES_... sizes)
		{
		    super::resize(p, {to_size(sizes)...}, super::Alignment);
		}
	    
    template <class... SIZES_,
	      std::enable_if_t<
		  sizeof...(SIZES_) == rank() &&
		  all<std::is_integral, std::tuple<SIZES_...> >::value>*
	      = nullptr>
    explicit	array(pointer p, size_t alignment, SIZES_... sizes)
		    :super(p, {to_size(sizes)...}, alignment)
		{
		}
    template <class... SIZES_>
    std::enable_if_t<sizeof...(SIZES_) == rank() &&
		     all<std::is_integral, std::tuple<SIZES_...> >::value>
		resize(pointer p, size_t alignment, SIZES_... sizes)
		{
		    super::resize(p, {to_size(sizes)...}, alignment);
		}
	    
    template <class T_> std::enable_if_t<TU::rank<T_>() == 0, array&>
		operator =(const T_& c)
		{
		    TU::fill<size0()>(begin(), size(), c);

		    return *this;
		}

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
    static std::enable_if_t<TU::rank<T_>() == 0>
		set_sizes(sizes_iterator iter, sizes_iterator end, const T_& val)
		{
		    throw std::runtime_error("array<BUF>::set_sizes(): too shallow initializer list!");
		}
    template <class T_>
    static std::enable_if_t<TU::rank<T_>() != 0>
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
*  type alias: replace_element<S, T>					*
************************************************************************/
namespace detail
{
// array<S, ALLOC<S>, SIZE, SIZES...> のSをTに置換する
  template <class S, template <class> class ALLOC,
	    size_t SIZE, size_t... SIZES, class T>
  struct replace_element<array<S, ALLOC<S>, SIZE, SIZES...>, T>
  {
    private:
      using U	 = typename replace_element<S, T>::type;

    public:
      using type = array<U, ALLOC<U>, SIZE, SIZES...>;
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
      template <class T_, class ALLOC_, size_t SIZE_, size_t... SIZES_>
      struct array_t<array<T_, ALLOC_, SIZES_...>, SIZE_>
      {
	  using type = array<T_, ALLOC_, SIZE_, SIZES_...>;
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

  template <class T>
  using	is_range_or_opnode	= std::integral_constant<bool,
							 is_range<T>::value ||
							 is_opnode<T>::value>;
}	// namespace detail

//! 反復子が指す型. ただし，それがrangeまたはopnodeならば，それが表現する配列型
template <class ITER>
using iterator_substance
	  = typename detail::substance_t<decayed_iterator_value<ITER>,
					 detail::is_range_or_opnode>::type;

/************************************************************************
*  evaluation of opnodes						*
************************************************************************/
//! 演算子の評価結果の型を返す
/*!
  Eが演算子に変換可能ならばその評価結果である配列の型を，そうでなければ
  Eへの定数参照を返す
  \param E	配列式の型
*/
template <class E>
using result_t	= typename detail::substance_t<const E&,
					       detail::is_opnode>::type;
    
//! 式の評価結果を返す
/*!
  \param expr	式
  \return	exprが演算子ならばその評価結果である配列を，そうでなければ
		expr自体の参照を返す
*/
template <class E> inline result_t<E>
evaluate(const E& expr)
{
    return expr;
}

template <class E>
inline std::enable_if_t<detail::is_opnode<E>::value, std::ostream&>
operator <<(std::ostream& out, const E& expr)
{
    return out << evaluate(expr);
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
	public:
	  binder2nd(OP op, R&& r) :_r(std::forward<R>(r)), _op(op)	{}

	  template <class T_>
	  auto	operator ()(T_&& arg) const
		{
		    return _op(std::forward<T_>(arg), _r);
		}

	private:
	// R が opnode に変換可能（参照型も可）であれば，その評価結果の型
	// opnode に変換できなければ R そのもの
	  using	cache_t = typename substance_t<R, is_opnode>::type;
	  
	  const cache_t	_r;	// 第2引数を評価してキャッシュに保存
	  const OP	_op;
      };


    public:
		product_opnode(L&& l, R&& r, OP op)
		    :_l(std::forward<L>(l)), _binder(op, std::forward<R>(r))
		{
		}

      constexpr static auto
		size0()
		{
		    return TU::size0<L>();
		}
      auto	begin()	const
		{
		  // transform_iterator への第1テンプレートパラメータを，
		  // binder2nd そのものではなく，それへの定数参照とする
		  // ことにより，キャッシュのコピーを防ぐ
		    return make_transform_iterator1(std::cbegin(_l),
						    std::cref(_binder));
		}
      auto	end() const
		{
		    return make_transform_iterator1(std::cend(_l),
						    std::cref(_binder));
		}
      auto	size() const
		{
		    return std::size(_l);
		}
      decltype(auto)
		operator [](size_t i) const
		{
		    return *(begin() + i);
		}

    private:
      const L		_l;
      const binder2nd	_binder;
  };

  template <class OP, class L, class R> inline auto
  make_product_opnode(L&& l, R&& r, OP op)
  {
      return product_opnode<OP, L, R>(std::forward<L>(l),
				      std::forward<R>(r), op);
  }

  struct bit_xor
  {
      template <class X_, class Y_>
      auto	operator ()(X_&& x, Y_&& y) const
		{
		    return std::forward<X_>(x) ^ std::forward<Y_>(y);
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
	  std::enable_if_t<rank<L>() == 1 && rank<R>() == 1>* = nullptr>
inline auto
operator *(const L& l, const R& r)
{
    using value_type = std::common_type_t<value_t<L>, value_t<R> >;

    assert(size<0>(l) == size<0>(r));
    constexpr size_t	S = detail::max<size0<L>(), size0<R>()>::value;
    return inner_product<S>(std::begin(l), std::size(l), std::begin(r),
			    value_type(0));
}

//! 2次元配列式と1または2次元配列式の積をとる.
/*!
  \param l	左辺の2次元配列式
  \param r	右辺の1または2次元配列式
  \return	積を表す演算子ノード
*/
template <class L, class R, std::enable_if_t<rank<L>() == 2 &&
					     rank<R>() >= 1 &&
					     rank<R>() <= 2>* = nullptr>
inline auto
operator *(L&& l, R&& r)
{
    return detail::make_product_opnode(
		std::forward<L>(l), std::forward<R>(r),
		[](auto&& x, auto&& y)
		{ return std::forward<decltype(x)>(x)
		       * std::forward<decltype(y)>(y); });
}

//! 1次元配列式と2次元配列式の積をとる.
/*!
  右辺の2次元配列式 r がrow major order. l の各要素を係数とした r の各行の
  線型結合を計算する．
  \param l	左辺の1次元配列式
  \param r	右辺の2次元配列式
  \return	積を表す演算子ノード
*/
template <class L, class R,
	  std::enable_if_t<rank<L>() == 1 && rank<R>() == 2>* = nullptr>
inline auto
operator *(L&& l, R&& r)
{
    constexpr auto
		S   = detail::max<detail::max<size0<L>(), size0<R>()>::value,
				  1>::value - 1;
    auto	a   = std::cbegin(l);
    auto	b   = std::cbegin(r);
    auto	val = evaluate(*a * *b);
    for_each<S>(++a, std::size(l) - 1, ++b,
		[&val](const auto& x, const auto& y){ val += x * y; });
    return val;
}

//! 1次元配列式と2次元配列式の積をとる.
/*!
  右辺の2次元配列式 r がcolumn major order. l と r の各列の内積を計算する．
  \param l	左辺の1次元配列式
  \param r	右辺の2次元配列式
  \return	積を表す演算子ノード
*/
template <class L, class ROW, size_t NROWS, size_t NCOLS,
	  std::enable_if_t<rank<L>() == 1>* = nullptr>
inline auto
operator *(L&& l, const range<column_iterator<ROW, NROWS>, NCOLS>& r)
{
    return detail::make_product_opnode(
		transpose(r), std::forward<L>(l),
		[](auto&& x, auto&& y)
		{ return std::forward<decltype(x)>(x)
		       * std::forward<decltype(y)>(y); });
}
    
//! 2つの1次元配列式の外積をとる.
/*!
  \param l	左辺の1次元配列式
  \param r	右辺の1次元配列式
  \return	外積を表す演算子ノード
*/
template <class L, class R,
	  std::enable_if_t<rank<L>() == 1 && rank<R>() == 1>* = nullptr>
inline auto
operator %(L&& l, R&& r)
{
    return detail::make_product_opnode(
		std::forward<L>(l), std::forward<R>(r),
		[](auto&& x, auto&& y)
		{ return std::forward<decltype(x)>(x)
		       * std::forward<decltype(y)>(y); });
}

//! 2つの1次元配列式のベクトル積をとる.
/*!
  演算子ノードではなく，評価結果の3次元配列が返される.
  \param l	左辺の1次元配列式
  \param r	右辺の1次元配列式
  \return	ベクトル積
*/
template <class L, class R>
inline std::enable_if_t<rank<L>() == 1 && rank<R>() == 1,
			Array<std::common_type_t<element_t<L>, element_t<R> >,
			      3> >
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
template <class L, class R, std::enable_if_t<rank<L>() == 2 &&
					     rank<R>() == 1>* = nullptr>
inline auto
operator ^(L&& l, R&& r)
{
    return detail::make_product_opnode(std::forward<L>(l), std::forward<R>(r),
				       detail::bit_xor());
}

}	// namespace TU
#endif	// !TU_ARRAY_H
