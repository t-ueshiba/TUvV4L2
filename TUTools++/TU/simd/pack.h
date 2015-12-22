/*
 *  $Id$
 */
#if !defined(__TU_SIMD_PACK_H)
#define __TU_SIMD_PACK_H

#include "TU/pair.h"
#include "TU/simd/cvt.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  class pack<T, N>							*
************************************************************************/
template <class T, size_t N=1>
using pack = pair_tree<vec<T>, N>;

template <class PACK>
using pack_vec = typename pair_traits<PACK>::leftmost_type;

template <class PACK>
using pack_element = typename pack_vec<PACK>::element_type;

template <class T, class PACK>
using pack_target = pack<T, (pair_traits<PACK>::size*
			     pack_vec<PACK>::size)/vec<T>::size>;

template <class PACK> inline std::ostream&
operator <<(std::ostream& out, const std::pair<PACK, PACK>& x)
{
    return out << '[' << x.first << ' ' << x.second << ']';
}

/************************************************************************
*  Converting packs							*
************************************************************************/
namespace detail
{
  template <class T, bool MASK>
  struct converter
  {
    private:
      template <class T_, class S_>
      typename std::enable_if<(vec<T_>::size == vec<S_>::size), vec<T_> >::type
      cvtadj(vec<S_> x) const
      {
	  return cvt<T_, MASK>(x);
      }

      template <class T_, class S_>
      typename std::enable_if<(2*vec<T_>::size == vec<S_>::size),
			      std::pair<vec<T_>, vec<T_> > >::type
      cvtadj(vec<S_> x) const
      {
	  return std::make_pair(cvt<T_, MASK, false>(x),
				cvt<T_, MASK, true >(x));
      }

      template <class T_, class S_>
      typename std::enable_if<(vec<T_>::size == 2*vec<S_>::size),
			      vec<T_> >::type
      cvtadj(const std::pair<vec<S_>, vec<S_> >& x) const
      {
	  return cvt<T_, MASK>(x.first, x.second);
      }

      template <class T_, class PACK_> pack_target<T_, std::pair<PACK_, PACK_> >
      cvtadj(const std::pair<PACK_, PACK_>& x) const
      {
	  return std::make_pair(cvtadj<T_>(x.first), cvtadj<T_>(x.second));
      }

    public:
      template <class PACK_>
      typename std::enable_if<std::is_same<T, pack_element<PACK_> >::value,
			      PACK_>::type
      operator ()(const PACK_& x) const
      {
	  return x;
      }
    
      template <class PACK_>
      typename std::enable_if<!std::is_same<T, pack_element<PACK_> >::value,
			      pack_target<T, PACK_> >::type
      operator ()(const PACK_& x) const
      {
	  using	S = pack_element<PACK_>;
	  using A = typename std::conditional<
			(vec<T>::size < vec<S>::size),
		        cvt_upper_type<T, S, MASK>,
			cvt_lower_type<T, S, MASK> >::type;
    
	  return (*this)(cvtadj<A>(x));
      }
  };

  template <size_t I>
  struct generic_get
  {
      template <class T>
      auto	operator ()(T&& x)	const -> decltype(std::get<I>(x))
		{
		    return std::get<I>(x);
		}
  };

  template <size_t I, class HEAD, class TAIL> inline auto
  get(const boost::tuples::cons<HEAD, TAIL>& x)
      -> decltype(boost::tuples::cons_transform(x, generic_get<I>()))
  {
      return boost::tuples::cons_transform(x, generic_get<I>());
  }

  template <class T>
  struct rearranged
  {
      typedef T	type;
  };
  template <class T, class TAIL>
  struct rearranged<boost::tuples::cons<std::pair<T, T>, TAIL> >
  {
    private:
      typedef boost::tuples::cons<std::pair<T, T>, TAIL>	argument_type;
      typedef typename rearranged<
	  decltype(detail::get<0>(std::declval<argument_type>()))>::type
								element_type;

    public:
      typedef std::pair<element_type, element_type>		type;
  };

  template <class T>
  inline typename std::enable_if<!is_pair<tuple_head<T> >::value, T>::type
  rearrange(const T& x)
  {
      return x;
  }

  template <class T, class TAIL>
  inline typename detail::rearranged<boost::tuples::cons<std::pair<T, T>,
							 TAIL> >::type
  rearrange(const boost::tuples::cons<std::pair<T, T>, TAIL>& x)
  {
      return std::make_pair(rearrange(detail::get<0>(x)),
			    rearrange(detail::get<1>(x)));
  }
}	// namesapce detail

template <class T, bool MASK=false, class PACK> inline pack_target<T, PACK>
cvt_pack(const PACK& x)
{
    return detail::converter<T, MASK>()(x);
}

template <class T, bool MASK=false, class HEAD, class TAIL> inline auto
cvt_pack(const boost::tuples::cons<HEAD, TAIL>& x)
    -> decltype(detail::rearrange(boost::tuples::cons_transform(
  				      x, detail::converter<T, MASK>())))
  //-> decltype(boost::tuples::cons_transform(x, detail::converter<T, MASK>()))
{
    return detail::rearrange(boost::tuples::cons_transform(
  				 x, detail::converter<T, MASK>()));
  //return boost::tuples::cons_transform(x, detail::converter<T, MASK>());
}

}	// namespace simd
}	// namespace TU

#endif	// !__TU_SIMD_PACK_H
