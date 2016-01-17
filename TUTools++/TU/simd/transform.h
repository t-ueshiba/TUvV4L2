/*
 *  $Id$
 */
#if !defined(__TU_SIMD_TRANSFORM_H)
#define	__TU_SIMD_TRANSFORM_H

#include "TU/iterator.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
namespace detail
{
  template <class T, bool MASK, class ITER_TUPLE, class OUT, class FUNC>
  class transformer
  {
    public:
      using head_iterator = tuple_head<ITER_TUPLE>;
      using result_type =
		typename std::conditional<
		    std::is_void<
			typename std::iterator_traits<OUT>::value_type>::value,
		    vec<T>,
		    typename std::iterator_traits<OUT>::value_type>::type;
      
    private:
      using R = typename result_type::element_type;

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
			    return cvt<T_, false, MASK>(x);
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
			    return cvt<T, false, MASK>(x);
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
				       HI_, MASK>(x);
			}
      };

      struct generic_cvtadj
      {
	  template <class S_>
	  auto		operator ()(vec<S_> x) const
			{
			    return cvtadj<R, false, MASK>(x);
			}
	  template <class S_>
	  auto		operator ()(vec<S_> x, vec<S_> y) const
			{
			    return cvtadj<R, MASK>(x, y);
			}
      };
      
      template <class ITER_, class DUMMY_=void>
      struct max_size
      {
	  static constexpr size_t
		value = std::iterator_traits<ITER_>::value_type::size;
      };
      template <class ITER_>
      struct max_size<ITER_&> : max_size<ITER_>
      {
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
      template <class TUPLE_,
		typename std::enable_if<
		    (vec<T>::size == max_size<TUPLE_>::value)>::type* = nullptr>
      auto	cvtup_down(TUPLE_&& x)
		{
		    return cvtadj<R, false, MASK>(
			       _func(boost::tuples::cons_transform(
					 generic_cvtdown(), x)));
		}
      template <class TUPLE_,
		typename std::enable_if<
		    (vec<T>::size < max_size<TUPLE_>::value)>::type* = nullptr>
      auto	cvtup_down(TUPLE_&& x)
		{
		    constexpr size_t	N = max_size<TUPLE_>::value;

		    const auto	y = cvtup_down(boost::tuples::cons_transform(
						   generic_cvtup<N/2, false>(),
						   x));
		    const auto	z = cvtup_down(boost::tuples::cons_transform(
						   generic_cvtup<N/2, true >(),
						   x));
		    return cvtadj<R, MASK>(y, z);
		}

      template <class TUPLE_>
      typename std::enable_if<(vec<R>::size == max_size<TUPLE_>::value)>::type
		cvtup(TUPLE_&& x)
		{
		    *_out = cvtup_down(x);
		    ++_out;
		}
      template <class TUPLE_>
      typename std::enable_if<(vec<T>::size < max_size<TUPLE_>::value &&
			       vec<R>::size < max_size<TUPLE_>::value)>::type
		cvtup(TUPLE_&& x)
		{
		    constexpr size_t	N = max_size<TUPLE_>::value;

		    cvtup(boost::tuples::cons_transform(
			      generic_cvtup<N/2, false>(), x));
		    cvtup(boost::tuples::cons_transform(
			      generic_cvtup<N/2, true >(), x));
		}

    public:
		transformer(const ITER_TUPLE& t,
			    head_iterator end, OUT out, FUNC func)
		    :_t(t), _end(end), _out(out), _func(func)		{}

      OUT	operator ()()
		{
		    while (boost::get<0>(_t) != _end)
		    {
			constexpr size_t	N = max_size<ITER_TUPLE>::value;

			cvtup(boost::tuples::cons_transform(
				  generic_cvtup<N, false>(), _t));
		    }
		    return _out;
		}

    private:
      ITER_TUPLE		_t;
      const head_iterator	_end;
      OUT			_out;
      FUNC			_func;
  };
}	// namespace detail

template <class T, bool MASK=false,
	  class FUNC, class OUT, class IN, class... IN_EXTRA> inline OUT
transform(FUNC func, OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, MASK, boost::tuple<IN, IN_EXTRA...>, OUT, FUNC>
	trns(boost::make_tuple(in, in_extra...), end, out, func);
    return trns();
}
    
template <class T, bool MASK=false, class ITER_TUPLE, class OUT, class FUNC>
inline OUT
transform(const ITER_TUPLE& ib, const ITER_TUPLE& ie, OUT out, FUNC func)
{
    detail::transformer<T, MASK, ITER_TUPLE, OUT, FUNC>
	trns(ib, boost::get<0>(ie), out, func);
    return trns();
}
    
template <class T, class ITER_TUPLE, class OUT, class FUNC> inline OUT
transform(const fast_zip_iterator<ITER_TUPLE>& ib,
	  const fast_zip_iterator<ITER_TUPLE>& ie, OUT out, FUNC func)
{
    return transform(ib.get_iterator_tuple(),
		     ie.get_iterator_tuple(), out, func);
}
    
template <class T, bool MASK=false, class OUT, class IN, class... IN_EXTRA>
inline OUT
copy(OUT out, IN in, IN end, IN_EXTRA... in_extra)
{
    detail::transformer<T, MASK, boost::tuple<IN, IN_EXTRA...>, OUT, identity>
	trns(boost::make_tuple(in, in_extra...), end, out, identity());
    return trns();
}
    
template <class T, bool MASK=false, class ITER_TUPLE, class OUT> inline OUT
copy(const ITER_TUPLE& ib, const ITER_TUPLE& ie, OUT out)
{
    detail::transformer<T, MASK, ITER_TUPLE, OUT, identity>
	trns(ib, boost::get<0>(ie), out, identity());
    return trns();
}
    
template <class T, class ITER_TUPLE, class OUT> inline OUT
copy(const fast_zip_iterator<ITER_TUPLE>& ib,
     const fast_zip_iterator<ITER_TUPLE>& ie, OUT out)
{
    return copy(ib.get_iterator_tuple(), ie.get_iterator_tuple(), out);
}
    
}	// namespace simd
}	// namespace TU

#endif	// !__TU_SIMD_TRANSFORM_H
