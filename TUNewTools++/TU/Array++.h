/*
 *  $Id$
 */
#include <array>
#include "TU/range.h"
#include "TU/utility.h"		// for std::index_sequence<Ints...>

namespace TU
{
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
    static OUT_		copy(IN_ ib, IN_ ie, OUT_ out)
			{
			    return std::copy(ib, ie, out);
			}

    template <class T_>
    static void		fill(iterator ib, iterator ie, const T_& c)
			{
			    std::fill(ib, ie, c);
			}

    static void		init(iterator ib, iterator ie)
			{
			    std::fill(ib, ie, 0);
			}
};

/************************************************************************
*  class Buf<T, ALLOC, SIZES...>					*
************************************************************************/
//! 固定長バッファクラス
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
    
  protected:
    constexpr static size_t	D = 1 + sizeof...(SIZES);
    
  public:
    using allocator_type	= void;
    using value_type		= T;
    using pointer		= typename super::pointer;
    using const_pointer		= typename super::const_pointer;

  public:
  // 標準コンストラクタ/代入演算子およびデストラクタ
		Buf()				= default;
		Buf(const Buf&)			= default;
    Buf&	operator =(const Buf&)		= default;
		Buf(Buf&&)			= default;
    Buf&	operator =(Buf&&)		= default;
    
  // 各軸のサイズと最終軸のストライドを指定したコンストラクタとリサイズ関数
    explicit	Buf(const std::array<size_t, D>& sizes, size_t=0)
		{
		    if (!check_sizes(sizes, axis<D>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::Buf(): mismatched size!");
		}
    void	resize(const std::array<size_t, D>& sizes, size_t=0)
		{
		    if (!check_sizes(sizes, axis<D>()))
			throw std::logic_error("Buf<T, ALLOC, SIZE, SIZES...>::resize(): mismatched size!");
		}

    template <size_t I_>
    constexpr static auto	size(axis<I_>)	{ return siz<I_>::value; }
    template <size_t I_>
    constexpr static auto	stride(axis<I_>){ return siz<I_>::value; }
    constexpr static auto	size()		{ return siz<0>::value; }
    constexpr static auto	nrow()		{ return siz<0>::value; }
    constexpr static auto	ncol()		{ return siz<1>::value; }

    auto	begin()		{ return make_iterator<SIZES...>(_a.begin()); }
    auto	begin()	const	{ return make_iterator<SIZES...>(_a.begin()); }
    auto	end()		{ return make_iterator<SIZES...>(_a.end()); }
    auto	end()	const	{ return make_iterator<SIZES...>(_a.end()); }

  private:
    template <size_t I_>
    static bool	check_sizes(const std::array<size_t, D>& sizes, axis<I_>)
		{
		    return (sizes[I_-1] != size(axis<I_-1>()) ? false :
			    check_sizes(sizes, axis<I_-1>()));
		}
    static bool	check_sizes(const std::array<size_t, D>& sizes, axis<0>)
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

//! 可変長バッファクラス
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

  protected:
    constexpr static size_t	D = 1 + sizeof...(SIZES);

  public:
    using allocator_type	= typename super::allocator_type;
    using value_type		= T;
    using pointer		= typename super::pointer;
    using const_pointer		= typename super::const_pointer;

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
		    super::copy(b.begin(), b.end(), begin());
		}
    Buf&	operator =(const Buf& b)
		{
		    if (this != &b)
		    {
			resize(b._sizes, b._stride);
			super::copy(b.begin(), b.end(), begin());
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
		}
		~Buf()
		{
		    free(_p, _capacity);
		}

  // 各軸のサイズと最終軸のストライドを指定したコンストラクタとリサイズ関数
    explicit	Buf(const std::array<size_t, D>& sizes, size_t stride=0)
		    :_sizes(sizes),
		     _stride(stride ? stride : _sizes[D-1]),
		     _capacity(capacity(axis<0>())),
		     _p(alloc(_capacity))
		{
		}
    void	resize(const std::array<size_t, D>& sizes, size_t stride=0)
		{
		    free(_p, _capacity);
		    _sizes    = sizes;
		    _stride   = (stride ? stride : _sizes[D-1]);
		    _capacity = capacity(axis<0>());
		    _p	      = alloc(_capacity);
		}

    template <size_t I_>
    auto	size(axis<I_>)	  const	{ return _sizes[I_]; }
    template <size_t I_>
    auto	stride(axis<I_>)  const	{ return _sizes[I_]; }
    auto	stride(axis<D-1>) const	{ return _stride; }
    auto	size()		  const	{ return _sizes[0]; }
    auto	nrow()		  const	{ return _sizes[0]; }
    auto	ncol()		  const	{ return _sizes[1]; }
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
		    return make_iterator<SIZES...>(base_iterator(_p + _capacity));
		}
    auto	end() const
		{
		    return make_iterator<SIZES...>(const_base_iterator(
						       _p + _capacity));
		}

  private:
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
		    constexpr size_t	I = D - sizeof...(SIZES_);
		    
		    return make_range_iterator(
			       make_iterator<SIZES_...>(iter),
			       size(axis<I>()), stride(axis<I>()));
		}

  private:
    allocator_type		_allocator;	//!< 要素を確保するアロケータ
    std::array<size_t, D>	_sizes;		//!< 各軸の要素数
    size_t			_stride;	//!< 最終軸のストライド
    size_t			_capacity;	//!< バッファ中に収めらる総要素数
    pointer			_p;		//!< 先頭要素へのポインタ
};
    
/************************************************************************
*  class array<BUF>							*
************************************************************************/
template <class BUF>	class array;
    
template <size_t I, class BUF> inline size_t
size(const array<BUF>& a)
{
    return a.size(std::integral_constant<size_t, I>());
}

template <size_t I, class BUF> inline size_t
stride(const array<BUF>& a)
{
    return a.stride(std::integral_constant<size_t, I>());
}

template <class BUF>
class array : public BUF
{
  private:
    using super			= BUF;
    using buf_iterator		= typename super::iterator;
    using const_buf_iterator	= typename super::const_iterator;
    template <size_t I_>
    using axis			= std::integral_constant<size_t, I_>;
    
    constexpr static size_t	D = super::D;
    
  public:
    using element_type		 = typename super::value_type;
    using pointer		 = typename super::pointer;
    using const_pointer		 = typename super::const_pointer;
    using iterator		 = decltype(std::declval<super*>()->begin());
    using const_iterator	 = decltype(std::declval<const super*>()
					    ->begin());
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
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
		    :super({cvt_to_size(sizes)...}, stride)
		{
		}
    template <class... SIZES_,
	      typename std::enable_if<sizeof...(SIZES_) == D>::type* = nullptr>
    explicit	array(size_t stride, SIZES_... sizes)
		    :super({cvt_to_size(sizes)...}, stride)
		{
		}
    template <class... SIZES_>
    typename std::enable_if<sizeof...(SIZES_) == D>::type
		resize(SIZES_... sizes)
		{
		    super::resize({cvt_to_size(sizes)...});
		}
    
    template <class... SIZES_>
    typename std::enable_if<sizeof...(SIZES_) == D>::type
		resize(size_t stride, SIZES_... sizes)
		{
		    super::resize({cvt_to_size(sizes)...}, stride);
		}

    template <class E_,
	      typename std::enable_if<is_range<E_>::value>::type* = nullptr>
		array(const E_& expr)
		    :super(sizes(expr, std::make_index_sequence<D>()))
		{
		    std::copy(std::begin(expr), std::end(expr), begin());
		}
    template <class E_>
    typename std::enable_if<is_range<E_>::value, array&>::type
		operator =(const E_& expr)
		{
		    super::resize(sizes(expr, std::make_index_sequence<D>()));
		    std::copy(std::begin(expr), std::end(expr), begin());

		    return *this;
		}

    using	super::size;
    using	super::stride;
    using	super::nrow;
    using	super::ncol;
    using	super::begin;
    using	super::end;
    
    auto	cbegin()  const	{ return begin(); }
    auto	cend()	  const	{ return end(); }
    auto	rbegin()	{ return std::make_reverse_iterator(end()); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	rend()		{ return std::make_reverse_iterator(begin()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crend()	  const	{ return rend(); }
    auto	operator [](size_t i)
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    const auto&	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    void	fill(const element_type& val)
		{
		    super::fill(begin(), end(), val);
		}
    
  private:
    template <class T_>
    static typename std::enable_if<std::is_integral<T_>::value, size_t>::type
		cvt_to_size(const T_& arg)
		{
		    return size_t(arg);
		}

    template <class E_, size_t... I_>
    static std::array<size_t, D>
		sizes(const E_& expr, std::index_sequence<I_...>)
		{
		    return {TU::size<I_>(expr)...};
		}
};

template <class T, size_t N=0, class ALLOC=std::allocator<T> >
using Array = array<Buf<T, ALLOC, N> >;

template <class T, size_t R=0, size_t C=0, class ALLOC=std::allocator<T> >
using Array2 = array<Buf<T, ALLOC, R, C> >;

template <class T,
	  size_t Z=0, size_t Y=0, size_t X=0, class ALLOC=std::allocator<T> >
using Array3 = array<Buf<T, ALLOC, Z, Y, X> >;

}	// namespace TU

