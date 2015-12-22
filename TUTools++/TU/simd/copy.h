/*
 *  $Id$
 */
#if !defined(__TU_SIMD_COPY_H)
#define	__TU_SIMD_COPY_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"
#if defined(TU_SIMD_DEBUG)
#  include <typeinfo>
#endif

namespace TU
{
namespace simd
{
namespace detail
{
  template <class T, bool MASK, class ITER_TUPLE, class OUT>
  class copier
  {
    public:
      using head_iterator = tuple_head<ITER_TUPLE>;
      
    private:
      struct generic_cvtdown
      {
	  template <class T_=T, class ITER_>
	  typename std::enable_if<
	      (vec<T_>::size == std::iterator_traits<ITER_>::value_type::size),
	      vec<T_> >::type
			operator ()(ITER_& iter) const
			{
			    auto	x = *iter;
			    ++iter;
			    return cvt<T_, MASK>(x);
			}
	  template <class T_=T, class ITER_>
	  typename std::enable_if<
	      (vec<T_>::size > std::iterator_traits<ITER_>::value_type::size),
	      vec<T_> >::type
			operator ()(ITER_& iter) const
			{
			    using S = typename std::iterator_traits<ITER_>
						  ::value_type::element_type;
			    using A = cvt_above_type<T_, S, MASK>;
	  
			    auto	x = operator ()<A>(iter);
			    auto	y = operator ()<A>(iter);
			    return cvt<T_, MASK>(x, y);
			}
	  template <class S_>
	  typename std::enable_if<vec<T>::size == vec<S_>::size, vec<T> >::type
			operator ()(vec<S_> x) const
			{
			    return cvt<T, MASK>(x);
			}
      };

      template <size_t N_, bool HI_>
      struct generic_cvtup
      {
	  template <class ITER_>
	  typename std::enable_if<
	      std::iterator_traits<ITER_>::value_type::size == N_,
	      typename std::iterator_traits<ITER_>::value_type>::type
	  		operator ()(ITER_& iter) const
			{
			    auto	x = *iter;
			    ++iter;
			    return x;
			}
	  template <class ITER_>
	  typename std::enable_if<
	      std::iterator_traits<ITER_>::value_type::size != N_, ITER_&>::type
			operator ()(ITER_& iter) const
			{
			    return iter;
			}
	  template <class S_>
	  typename std::enable_if<vec<cvt_upper_type<T, S_, MASK> >::size == N_,
				  vec<cvt_upper_type<T, S_, MASK> > >::type
			operator ()(vec<S_> x) const
			{
			    return cvt<cvt_upper_type<T, S_, MASK>,
				       MASK, HI_>(x);
			}
      };

      template <class ITER_, class DUMMY_=void>
      struct max_size
      {
	  static constexpr size_t
	      value = std::iterator_traits<typename std::remove_reference<
					       ITER_>::type>::value_type::size;
      };
      template <class T_>
      struct max_size<vec<T_> >
      {
	  static constexpr size_t	value = vec<T_>::size;
      };
      template <class DUMMY_>
      struct max_size<boost::tuples::null_type, DUMMY_>
      {
	  static constexpr size_t	value = 0;
      };
      template <class HEAD_, class TAIL_>
      struct max_size<boost::tuples::cons<HEAD_, TAIL_> >
      {
	  static constexpr size_t	head_max = max_size<HEAD_>::value;
	  static constexpr size_t	tail_max = max_size<TAIL_>::value;
	  static constexpr size_t	value	 = (head_max > tail_max ?
						    head_max : tail_max);
      };
      template <class... T_>
      struct max_size<boost::tuple<T_...> >
	  : max_size<typename boost::tuple<T_...>::inherited>
      {
      };
      
    private:
      template <class TUPLE_>
      typename std::enable_if<(vec<T>::size == max_size<TUPLE_>::value)>::type
		cvtup(TUPLE_& x)
		{
#if defined(TU_SIMD_DEBUG)
		    constexpr size_t	N = max_size<TUPLE_>::value;
		    std::cout << '*' << N << ": " << typeid(TUPLE_).name()
			      << std::endl;
#endif
		    *_out = boost::tuples::cons_transform(x, generic_cvtdown());
		    ++_out;
		}
      template <class TUPLE_>
      typename std::enable_if<(vec<T>::size < max_size<TUPLE_>::value)>::type
		cvtup(TUPLE_& x)
		{
		    constexpr size_t	N = max_size<TUPLE_>::value;
#if defined(TU_SIMD_DEBUG)
		    std::cout << ' ' << N << ": " << typeid(TUPLE_).name()
			      << std::endl;
#endif
		    auto	y = boost::tuples::cons_transform(
					x, generic_cvtup<N/2, false>());
		    cvtup(y);
		    auto	z = boost::tuples::cons_transform(
					x, generic_cvtup<N/2, true>());
		    cvtup(z);
		}

    public:
		copier(const ITER_TUPLE& t, head_iterator end, OUT out)
		    :_t(t), _end(end), _out(out)			{}

      OUT	operator ()()
		{
		    while (boost::get<0>(_t) != _end)
		    {
			constexpr size_t	N = max_size<ITER_TUPLE>::value;
#if defined(TU_SIMD_DEBUG)
			std::cout << '#' << N << ": "
				  << typeid(decltype(_t)).name()
				  << std::endl;
#endif
			auto	x = boost::tuples::cons_transform(
					_t, generic_cvtup<N, false>());
			cvtup(x);
		    }
		    return _out;
		}

    private:
      ITER_TUPLE		_t;
      const head_iterator	_end;
      OUT			_out;
  };
}	// namespace detail

template <class T, bool MASK=false, class OUT, class IN, class... IN_EXTRA>
inline OUT
copy(OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::copier<T, MASK, boost::tuple<IN, IN_EXTRA...>, OUT>
	cp(boost::make_tuple(in, in_extra...), end, out);
    return cp();
}
    
template <class T, bool MASK=false, class ITER_TUPLE, class OUT> inline OUT
copy(const ITER_TUPLE& ib, const ITER_TUPLE& ie, OUT out)
{
    detail::copier<T, MASK, ITER_TUPLE, OUT>	cp(ib, boost::get<0>(ie), out);
    return cp();
}
    
template <class T, class ITER_TUPLE, class OUT> inline OUT
copy(const fast_zip_iterator<ITER_TUPLE>& ib,
     const fast_zip_iterator<ITER_TUPLE>& ie, OUT out)
{
    return copy(ib.get_iterator_tuple(), ie.get_iterator_tuple(), out);
}
    
}	// namespace simd
}	// namespace TU

#endif	// !__TU_SIMD_COPY_H
