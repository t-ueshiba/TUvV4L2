/*
 *  $Id$
 */
#if !defined(TU_SIMD_PACK_H)
#define TU_SIMD_PACK_H

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
using pack	   = pair_tree<vec<T>, N>;

template <class PACK>
using pack_vec	   = typename pair_traits<PACK>::leftmost_type;

template <class PACK>
using pack_element = typename pack_vec<PACK>::element_type;

template <class T, class PACK>
using pack_target  = pack<T, (pair_traits<PACK>::size*
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
  /**********************************************************************
  *  class converter<T, MASK>						*
  **********************************************************************/
  template <class T, bool MASK>
  struct converter
  {
    private:
      template <class T_, class S_>
      std::enable_if_t<(vec<T_>::size == vec<S_>::size), vec<T_> >
      cvtadj(vec<S_> x) const
      {
	  return cvt<T_, false, MASK>(x);
      }

      template <class T_, class S_>
      std::enable_if_t<(2*vec<T_>::size == vec<S_>::size),
		       std::pair<vec<T_>, vec<T_> > >
      cvtadj(vec<S_> x) const
      {
	  return std::make_pair(cvt<T_, false, MASK>(x),
				cvt<T_, true , MASK>(x));
      }

      template <class T_, class S_>
      std::enable_if_t<(vec<T_>::size == 2*vec<S_>::size), vec<T_> >
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
      std::enable_if_t<std::is_same<T, pack_element<PACK_> >::value, PACK_>
      operator ()(const PACK_& x) const
      {
	  return x;
      }
    
      template <class PACK_>
      std::enable_if_t<!std::is_same<T, pack_element<PACK_> >::value,
		       pack_target<T, PACK_> >
      operator ()(const PACK_& x) const
      {
	  using	S = pack_element<PACK_>;
	  using A = std::conditional_t<(vec<T>::size < vec<S>::size),
				       cvt_upper_type<T, S, MASK>,
				       cvt_lower_type<T, S, MASK> >;
    
	  return (*this)(cvtadj<A>(x));
      }
  };

  template <size_t I, class... T> inline auto
  get(const std::tuple<T...>& t)
  {
      return tuple_transform([](auto x){ return std::get<I>(x); }, t);
  }

  template <class T>
  inline std::enable_if_t<!is_pair<std::tuple_element<0, T> >::value, T>
  rearrange(const T& x)
  {
      return x;
  }

  template <class T, class... TAIL> inline auto
  rearrange(const std::tuple<std::pair<T, T>, TAIL...>& x)
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

template <class T, bool MASK=false, class... S> inline auto
cvt_pack(const std::tuple<S...>& x)
{
    return detail::rearrange(tuple_transform(detail::converter<T, MASK>(), x));
}

}	// namespace simd
}	// namespace TU

#endif	// !TU_SIMD_PACK_H
