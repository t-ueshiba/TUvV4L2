/*
 *  $Id$
 */
/*!
  \file		utility.h
  \brief	プログラミングに便利な種々のクラスの定義と実装
*/
#ifndef __TU_UTILITY_H
#define __TU_UTILITY_H

#include <cstddef>	// for size_t
#include <utility>

namespace std
{
#if __cplusplus <= 201103L
/************************************************************************
*  struct index_sequence<size_t...>					*
************************************************************************/
template <size_t...> struct index_sequence			{};

namespace detail
{
  template <size_t N, size_t... IDX>
  struct make_index_sequence : make_index_sequence<N - 1, N - 1, IDX...>
  {
  };
  template <size_t... IDX>
  struct make_index_sequence<0, IDX...>
  {
      typedef index_sequence<IDX...>	type;
  };
}	// namespace detail
    
template <size_t N>
using make_index_sequence = typename detail::make_index_sequence<N>::type;
#endif
}	// namespace std
#endif	// !__TU_UTILITY_H
