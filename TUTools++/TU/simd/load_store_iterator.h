/*!
  \file		load_store_iterator.h
  \author	Toshio UESHIBA
  \brief	メモリとSIMDベクトル間のデータ転送を行う反復子の定義
*/
#if !defined(TU_SIMD_LOAD_STORE_ITERATOR_H)
#define TU_SIMD_LOAD_STORE_ITERATOR_H

#include "TU/iterator.h"
#include "TU/simd/load_store.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  iterator_wrapper<ITER>						*
************************************************************************/
//! 反復子をラップして名前空間 TU::simd に取り込むためのクラス
/*!
  本クラスの目的は，以下の2つである．
    (1)	iterator_value<ITER> がSIMDベクトル型となるあらゆる反復子を
	その機能を保持したまま名前空間 TU::simd のメンバとすることにより，
	ADL を発動して simd/algorithm.h に定義された関数を起動できる
	ようにすること．
    (2)	ITER が const T* 型または T* 型のとき，それぞれ load_iterator<T, true>
	store_iterator<T, true> を生成できるようにすること．本クラスでラップ
	されたポインタが指すアドレスは sizeof(vec<T>) にalignされているものと
	みなされる．
  \param ITER	ラップする反復子の型
*/ 
template <class ITER>
class iterator_wrapper
    : public boost::iterator_adaptor<iterator_wrapper<ITER>, ITER>
{
  private:
    using	super = boost::iterator_adaptor<iterator_wrapper, ITER>;

  public:
		iterator_wrapper(ITER iter)	:super(iter)	{}
    template <class ITER_,
	      std::enable_if_t<std::is_convertible<ITER_, ITER>::value>*
	      = nullptr>
		iterator_wrapper(ITER_ iter)	:super(iter)	{}
};

//! 反復子をラップして名前空間 TU::simd に取り込む
/*!
  \param iter	ラップする反復子
*/
template <class ITER> inline iterator_wrapper<ITER>
wrap_iterator(ITER iter)
{
    return {iter};
}
    
//! ラップされた反復子からもとの反復子を取り出す
/*!
  \param iter	ラップされた反復子
  \return	もとの反復子
*/
template <class ITER> inline ITER
make_accessor(iterator_wrapper<ITER> iter)
{
    return iter.base();
}

namespace detail
{
  template <class ITER>
  struct vsize		// iterator_value<ITER> が vec<T> 型
  {
      constexpr static size_t	value = iterator_value<ITER>::size;
  };
  template <class T>
  struct vsize<T*>
  {
      constexpr static size_t	value = vec<std::remove_cv_t<T> >::size;
  };
}	// namespace detail

//! ある要素型を指定された個数だけカバーするために必要なSIMDベクトルの個数を調べる
/*!
  T を value_type<ITER> がSIMDベクトル型となる場合はその要素型，ITER がポインタの
  場合はそれが指す要素の型としたとき，指定された個数のT型要素をカバーするために
  必要な vec<T> 型SIMDベクトルの最小個数を返す．
  \param n	要素の個数
  \return	nをカバーするために必要なSIMDベクトルの個数
*/
template <class ITER> constexpr inline size_t
make_terminator(size_t n)
{
    return (n ? (n - 1)/detail::vsize<ITER>::value + 1 : 0);
}
    
/************************************************************************
*  class load_iterator<T, ALIGNED>					*
************************************************************************/
//! 反復子が指すアドレスからSIMDベクトルを読み込む反復子
/*!
  \param T		SIMDベクトルの成分の型
  \param ALIGNED	読み込み元のアドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class T, bool ALIGNED=false>
class load_iterator
    : public boost::iterator_adaptor<load_iterator<T, ALIGNED>,
				     const T*,
				     vec<T>,
				     boost::use_default,
				     vec<T> >
{
  private:
    using element_type	= T;
    using super		= boost::iterator_adaptor<load_iterator,
						  const T*,
						  vec<T>,
						  boost::use_default,
						  vec<T> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::value_type;
    using	typename super::reference;
    
  public:
    load_iterator(const T* p=nullptr)	:super(p)	{}
    load_iterator(const value_type* p)
	:super(reinterpret_cast<const T*>(p))		{}
	       
  private:
    reference		dereference() const
			{
			    return load<ALIGNED>(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference()
				+= n * difference_type(value_type::size);
			}
    void		increment()
			{
			    super::base_reference()
				+= difference_type(value_type::size);
			}
    void		decrement()
			{
			    super::base_reference()
				-= difference_type(value_type::size);
			}
    difference_type	distance_to(load_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / difference_type(value_type::size);
			}
};

//! 定数ポインタからSIMDベクトルを読み込む反復子を生成する
/*!
  定数ポインタはalignされている必要はない．
  \param p	読み込み元を指す定数ポインタ
  \return	SIMDベクトルを書き込む反復子
*/
template <bool ALIGNED=false, class T> inline load_iterator<T, false>
make_accessor(const T* p)
{
    return {p};
}

//! ラップされた定数ポインタからSIMDベクトルを読み込む反復子を生成する
/*!
  ラップされたポインタは sizeof(vec<T>) にalignされていなければならない．
  \param p	ラップされた定数ポインタ
  \return	SIMDベクトルを読み込む反復子
*/
template <class T> inline load_iterator<T, true>
make_accessor(iterator_wrapper<const T*> p)
{
    return {p.base()};
}
    
/************************************************************************
*  class store_iterator<T, ALIGNED>					*
************************************************************************/
namespace detail
{
  template <class T, bool ALIGNED=false>
  class store_proxy
  {
    public:
      using element_type = T;
      using value_type	 = decltype(load<ALIGNED>(std::declval<const T*>()));
      
	
    public:
      store_proxy(T* p)		:_p(p)			{}

			operator value_type() const
			{
			    return load<ALIGNED>(_p);
			}
      store_proxy&	operator =(value_type val)
			{
			    store<ALIGNED>(_p, val);
			    return *this;
			}
      store_proxy&	operator +=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) + val);
			}
      store_proxy&	operator -=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) - val);
			}
      store_proxy&	operator *=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) * val);
			}
      store_proxy&	operator /=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) / val);
			}
      store_proxy&	operator %=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) % val);
			}
      store_proxy&	operator &=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) & val);
			}
      store_proxy&	operator |=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) | val);
			}
      store_proxy&	operator ^=(value_type val)
			{
			    return operator =(load<ALIGNED>(_p) ^ val);
			}

    private:
      T* 	_p;
  };
}	// namespace detail

//! 反復子が指す書き込み先にSIMDベクトルを書き込む反復子
/*!
  \param T		SIMDベクトルの成分の型
  \param ALIGNED	書き込み先アドレスがalignmentされていればtrue,
			そうでなければfalse
*/
template <class T, bool ALIGNED=false>
class store_iterator
    : public boost::iterator_adaptor<
		store_iterator<T, ALIGNED>,
		T*,
		typename detail::store_proxy<T, ALIGNED>::value_type,
  // boost::use_default とすると libc++ で std::fill() に適用できない
		iterator_category<T*>,
		detail::store_proxy<T, ALIGNED> >
{
  private:
    using super	= boost::iterator_adaptor<
		      store_iterator,
		      T*,
		      typename detail::store_proxy<T, ALIGNED>::value_type,
		      iterator_category<T*>,
		      detail::store_proxy<T, ALIGNED> >;
    friend	class boost::iterator_core_access;

  public:
    using	typename super::difference_type;
    using	typename super::value_type;
    using	typename super::reference;
    
  public:
    store_iterator(T* p=nullptr)	:super(p)	{}
    store_iterator(value_type* p)
	:super(reinterpret_cast<T*>(p))			{}

    value_type		operator ()() const
			{
			    return load<ALIGNED>(super::base());
			}
    
  private:
    reference		dereference() const
			{
			    return reference(super::base());
			}
    void		advance(difference_type n)
			{
			    super::base_reference()
				+= n * difference_type(value_type::size);
			}
    void		increment()
			{
			    super::base_reference()
				+= difference_type(value_type::size);
			}
    void		decrement()
			{
			    super::base_reference()
				-= difference_type(value_type::size);
			}
    difference_type	distance_to(store_iterator iter) const
			{
			    return (iter.base() - super::base())
				 / diffence_type(value_type::size);
			}
};

//! ポインタからSIMDベクトルを書き込む反復子を生成する
/*!
  ポインタはalignされている必要はない．
  \param p	書き込み先を指すポインタ
  \return	SIMDベクトルを書き込む反復子
*/
template <class T> inline store_iterator<T, false>
make_accessor(T* p)
{
    return {p};
}

//! ラップされたポインタからSIMDベクトルを書き込む反復子を生成する
/*!
  ラップされたポインタは sizeof(vec<T>) にalignされていなければならない．
  \param p	ラップされたポインタ
  \return	SIMDベクトルを書き込む反復子
*/
template <class T> inline store_iterator<T, true>
make_accessor(iterator_wrapper<T*> p)
{
    return {p.base()};
}
    
//! zip_iterator中の各反復子からSIMDベクトルを読み書きする反復子を生成し，それを再度zip_iteratorにまとめる
/*!
  \param zip_iter	SIMDベクトルを読み込み元/書き込み先を指す反復子を束ねた
			zip_iterator
  \return		SIMDベクトルを読み書きする反復子を束ねたzip_iterator
*/
template <class ITER_TUPLE> inline auto
make_accessor(zip_iterator<ITER_TUPLE> zip_iter)
{
    return make_zip_iterator(tuple_transform([](auto iter)
					     { return make_accessor(iter); },
					     zip_iter.get_iterator_tuple()));
}

}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_LOAD_STORE_ITERATOR_H
