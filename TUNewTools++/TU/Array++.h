/*
 *  $Id$
 */
#include "TU/utility.h"
#include "TU/iterator.h"
#include <array>

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
*  class Buf<T, N, ALLOC>						*
************************************************************************/
//! 固定長バッファクラス
/*!
  単独で使用することはなく，#new_array の内部バッファクラスとして使う．
  \param T	要素の型
  \param N	バッファ中の要素数
*/
template <class T, size_t N, class ALLOC>
class Buf : public BufTraits<T, ALLOC>
{
  private:
    using super		 = BufTraits<T, ALLOC>;
    
  public:
    using allocator_type = void;
    using value_type	 = T;
    using pointer	 = typename super::pointer;
    using const_pointer	 = typename super::const_pointer;
    using iterator	 = typename super::iterator;
    using const_iterator = typename super::const_iterator;
    
  public:
    explicit		Buf(size_t siz=N)	{ resize(siz); }
			Buf(const Buf&)		= default;
    Buf&		operator =(const Buf&)	= default;
			Buf(Buf&&)		= default;
    Buf&		operator =(Buf&&)	= default;

    constexpr static size_t
			size()			{ return N; }
    iterator		begin()			{ return _a.begin(); }
    const_iterator	begin()		const	{ return _a.begin(); }
    iterator		end()			{ return _a.end(); }
    const_iterator	end()		const	{ return _a.end(); }
    
  //! バッファの要素数を変更する．
  /*!
    実際にはバッファの要素数を変更することはできないので，与えられた要素数が
    このバッファの要素数に一致する場合のみ，通常どおりにこの関数から制御が返る．
    \param siz	新しい要素数
  */
    static void		resize(size_t siz)	{ assert(siz == N); }

  private:
    alignas(sizeof(T))	std::array<T, N>	_a;
};

//! 可変長バッファクラス
/*!
  単独で使用することはなく，#new_array の内部バッファクラスとして使う．
  \param T	要素の型
  \param ALLOC	アロケータの型
*/
template <class T, class ALLOC>
class Buf<T, 0, ALLOC> : public BufTraits<T, ALLOC>
{
  private:
    using super		 = BufTraits<T, ALLOC>;
    
  public:
    using allocator_type = typename super::allocator_type;
    using value_type	 = T;
    using pointer	 = typename super::pointer;
    using const_pointer	 = typename super::const_pointer;
    using iterator	 = typename super::iterator;
    using const_iterator = typename super::const_iterator;
    
  public:
    explicit		Buf(size_t siz=0)
			    :_size(siz), _p(alloc(_size))	{}
			Buf(const Buf& b)
			    :_size(b._size), _p(alloc(_size))
			{
			    super::copy(b._p, b._p + b._size, _p);
			}
    Buf&		operator =(const Buf& b)
			{
			    if (this != &b)
			    {
				resize(b._size);
				super::copy(b._p, b._p + b._size, _p);
			    }
			    return *this;
			}
			Buf(Buf&& b)
			    :_size(b._size), _p(b._p)
			{
			  // b の 破壊時に this->_p がdeleteされることを防ぐ．
			    b._size = 0;
			    b._p    = super::null();
			}
    Buf&		operator =(Buf&& b)
			{
			    free(_p, _size);
			    _size = b._size;
			    _p	  = b._p;
			    
			  // b の 破壊時に this->_p がdeleteされることを防ぐ．
			    b._size = 0;
			    b._p    = super::null();

			    return *this;
			}
			~Buf()			{ free(_p, _size); }

    size_t		size()		const	{ return _size; }
    iterator		begin()			{ return _p; }
    const_iterator	begin()		const	{ return _p; }
    iterator		end()			{ return _p + _size; }
    const_iterator	end()		const	{ return _p + _size; }
    
  //! バッファの要素数を変更する．
  /*!
    ただし，他のオブジェクトと記憶領域を共有しているバッファの要素数を
    変更することはできない．
    \param siz			新しい要素数
    \throw std::logic_error	記憶領域を他のオブジェクトと共有している場合
				に送出
  */
    void		resize(size_t siz)
			{
			    free(_p, _size);
			    _size = siz;
			    _p	  = alloc(_size);
			}
	
  private:
    pointer		alloc(size_t siz)
			{
			    pointer	p = _allocator.allocate(siz);
			    for (pointer q = p, qe = q + siz; q != qe; ++q)
				_allocator.construct(q, value_type());
			    return p;
			}
    void		free(pointer p, size_t siz)
			{
			    if (p)
			    {
				for (pointer q = p, qe = q + siz; q != qe; ++q)
				    _allocator.destroy(q);
				_allocator.deallocate(p, siz);
			    }
			}
    
  private:
    allocator_type	_allocator;		//!< アロケータ
    size_t		_size;			//!< 要素数
    pointer		_p;			//!< 記憶領域の先頭ポインタ
};

/************************************************************************
*  class new_array<D, BUF>						*
************************************************************************/
template <size_t D, class BUF>	class new_array;

template <size_t I, size_t D, class BUF> inline size_t
size(const new_array<D, BUF>& a)
{
    return a.size(std::integral_constant<size_t, I>());
}

template <size_t I, size_t D, class BUF> inline size_t
stride(const new_array<D, BUF>& a)
{
    return a.stride(std::integral_constant<size_t, I>());
}

template <size_t D, class BUF>
class new_array
{
  private:
    using buf_type		= BUF;
    using buf_iterator		= typename buf_type::iterator;
    using const_buf_iterator	= typename buf_type::const_iterator;
    using _0			= std::integral_constant<size_t, 0>;
    using _1			= std::integral_constant<size_t, 1>;
    using _D1			= std::integral_constant<size_t, D-1>;

    template <class ITER_>
    static auto	make_iterator_dummy(ITER_ iter, _D1)
		{
		    return make_range_iterator(iter, 0, 0);
		}
    template <class ITER_, size_t I_>
    static auto	make_iterator_dummy(ITER_ iter,
				    std::integral_constant<size_t, I_>)
		{
		    return make_range_iterator(
			       make_iterator_dummy(
				   iter,
				   std::integral_constant<size_t, I_+1>()),
			       0, 0);
		}
    
  public:
    using element_type		 = typename buf_type::value_type;
    using pointer		 = typename buf_type::pointer;
    using const_pointer		 = typename buf_type::const_pointer;
#ifndef __INTEL_COMPILER
    using iterator		 = decltype(
				       make_iterator_dummy(
					   std::declval<buf_iterator>(), _1()));
    using const_iterator	 = decltype(
				       make_iterator_dummy(
					   std::declval<const_buf_iterator>(),
					   _1()));
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
					       ::value_type;
    using reference		 = typename std::iterator_traits<iterator>
					       ::reference;
    using const_reference	 = typename std::iterator_traits<const_iterator>
					       ::reference;
#endif
  public:
    template <class... ARGS_,
	      typename std::enable_if<sizeof...(ARGS_) == D>::type* = nullptr>
    explicit	new_array(ARGS_... args)
		    :_sizes{cvt_to_size(args)...},
		     _stride(_sizes[D-1]),
		     _buf(capacity())
		{
		}
    template <class... ARGS_,
	      typename std::enable_if<sizeof...(ARGS_) == D>::type* = nullptr>
    explicit	new_array(size_t stride, ARGS_... args)
		    :_sizes{cvt_to_size(args)...},
		     _stride(stride),
		     _buf(capacity())
		{
		}
    template <class... ARGS_,
	      typename std::enable_if<sizeof...(ARGS_) == D>::type* = nullptr>
		new_array(pointer p, ARGS_... args)
		    :_sizes{cvt_to_size(args)...},
		     _stride(_sizes[D-1]),
		     _buf(p, capacity())
		{
		}
    template <class... ARGS_,
	      typename std::enable_if<sizeof...(ARGS_) == D>::type* = nullptr>
		new_array(pointer p, size_t stride, ARGS_... args)
		    :_sizes{cvt_to_size(args)...},
		     _stride(stride),
		     _buf(p, capacity())
		{
		}

		new_array()
		    :_stride(0), _buf()
		{
		    _sizes.fill(0);
		}
		new_array(const new_array&)			= default;
    new_array&	operator =(const new_array&)			= default;
		new_array(new_array&&)				= default;
    new_array&	operator =(new_array&&)				= default;
    
    template <class E_,
	      typename std::enable_if<is_range<E_>::value>::type* = nullptr>
		new_array(const E_& expr)
		    :_sizes(sizes(expr, std::make_index_sequence<D>())),
		     _stride(_sizes[D-1]),
		     _buf(capacity())
		{
		    std::copy(std::begin(expr), std::end(expr), begin());
		}
    template <class E_>
    typename std::enable_if<is_range<E_>::value, new_array&>::type
		operator =(const E_& expr)
		{
		    _sizes  = sizes(expr, std::make_index_sequence<D>());
		    _stride = _sizes[D-1];
		    _buf.resize(capacity());
		    std::copy(std::begin(expr), std::end(expr), begin());

		    return *this;
		}
    
    auto	size()	  const	{ return _sizes[0]; }
    auto	nrow()	  const	{ return _sizes[0]; }
    auto	ncol()	  const	{ return _sizes[1]; }
    auto	stride()  const	{ return _stride; }
	
    template <size_t I_>
    auto	size(std::integral_constant<size_t, I_>) const
		{
		    return _sizes[I_];
		}
    auto	stride(_D1) const
		{
		    return _stride;
		}
    template <size_t I_>
    auto	stride(std::integral_constant<size_t, I_>) const
		{
		    return _sizes[I_];
		}

    auto	begin()		{ return make_iterator(_buf.begin(), _1()); }
    auto	begin()	  const	{ return make_iterator(_buf.begin(), _1()); }
    auto	cbegin()  const	{ return begin(); }
    auto	end()		{ return make_iterator(_buf.end(), _1()); }
    auto	end()	  const	{ return make_iterator(_buf.end(), _1()); }
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

    template <class... ARGS_>
    typename std::enable_if<sizeof...(ARGS_) == D>::type
		resize(ARGS_... args)
		{
		    _sizes  = {cvt_to_size(args)...};
		    _stride = _sizes[D-1];
		    _buf.resize(capacity());
		}
    
    template <class... ARGS_>
    typename std::enable_if<sizeof...(ARGS_) == D>::type
		resize(size_t stride, ARGS_... args)
		{
		    _sizes  = {cvt_to_size(args)...};
		    _stride = stride;
		    _buf.resize(capacity());
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
    
    size_t	capacity() const
		{
		    return capacity(_0());
		}
    size_t	capacity(_D1) const
		{
		    return _stride;
		    
		}
    template <size_t I_>
    size_t	capacity(std::integral_constant<size_t, I_>) const
		{
		    return _sizes[I_]
			 * capacity(std::integral_constant<size_t, I_+1>());
		}

    template <class ITER_>
    auto	make_iterator(ITER_ iter, _D1) const
		{
		    return make_range_iterator(iter, _sizes[D-1], _stride);
		}
    template <class ITER_, size_t I_>
    auto	make_iterator(ITER_ iter,
			      std::integral_constant<size_t, I_>) const
		{
		    return make_range_iterator(
			       make_iterator(
				   iter,
				   std::integral_constant<size_t, I_+1>()),
			       _sizes[I_], _sizes[I_]);
		}
    
  private:
    std::array<size_t, D>	_sizes;
    size_t			_stride;
    buf_type			_buf;
};

template <class BUF>
class new_array<1, BUF>
{
  private:
    using buf_type		= BUF;

  public:
    using element_type		 = typename buf_type::value_type;
    using pointer		 = typename buf_type::pointer;
    using const_pointer		 = typename buf_type::const_pointer;
    using iterator		 = typename buf_type::iterator;
    using const_iterator	 = typename buf_type::const_iterator;
    using reverse_iterator	 = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;
    using value_type		 = typename std::iterator_traits<iterator>
					       ::value_type;
    using reference		 = typename std::iterator_traits<iterator>
					       ::reference;
    using const_reference	 = typename std::iterator_traits<const_iterator>
					       ::reference;

  public:
    explicit	new_array(size_t siz)		 :_buf(siz)	{}
		new_array(pointer p, size_t siz) :_buf(p, siz)	{}

		new_array()					= default;
		new_array(const new_array&)			= default;
    new_array&	operator =(const new_array&)			= default;
		new_array(new_array&&)				= default;
    new_array&	operator =(new_array&&)				= default;
    
    template <class E_,
	      typename std::enable_if<is_range<E_>::value>::type* = nullptr>
		new_array(const E_& expr)
		    :_buf(std::size(expr))
		{
		    std::copy(std::begin(expr), std::end(expr), begin());
		}
    template <class E_>
    typename std::enable_if<is_range<E_>::value, new_array&>::type
		operator =(const E_& expr)
		{
		    _buf.resize(std::size(expr));
		    std::copy(std::begin(expr), std::end(expr), begin());

		    return *this;
		}
    
    auto	size()	  const	{ return _buf.size(); }
    auto	size(std::integral_constant<size_t, 0>) const
		{
		    return size();
		}
    auto	stride(std::integral_constant<size_t, 0>) const
		{
		    return size();
		}

    auto	begin()		{ return _buf.begin(); }
    auto	begin()	  const	{ return _buf.begin(); }
    auto	cbegin()  const	{ return begin(); }
    auto	end()		{ return _buf.end(); }
    auto	end()	  const	{ return _buf.end(); }
    auto	cend()	  const	{ return end(); }
    auto	rbegin()	{ return std::make_reverse_iterator(end()); }
    auto	rbegin()  const	{ return std::make_reverse_iterator(end()); }
    auto	crbegin() const	{ return rbegin(); }
    auto	rend()		{ return std::make_reverse_iterator(begin()); }
    auto	rend()	  const	{ return std::make_reverse_iterator(begin()); }
    auto	crend()	  const	{ return rend(); }
    reference	operator [](size_t i)
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    const auto&	operator [](size_t i) const
		{
		    assert(i < size());
		    return *(begin() + i);
		}
    
  private:
    buf_type	_buf;
};

template <class T, class ALLOC=std::allocator<T> >
using Array = new_array<1, Buf<T, 0, ALLOC> >;

template <class T, class ALLOC=std::allocator<T> >
using Array2 = new_array<2, Buf<T, 0, ALLOC> >;

template <class T, class ALLOC=std::allocator<T> >
using Array3 = new_array<3, Buf<T, 0, ALLOC> >;

}	// namespace TU

