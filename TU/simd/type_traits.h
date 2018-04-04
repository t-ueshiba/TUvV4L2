/*!
  \file		type_traits.h
  \author	Toshio UESHIBA
  \brief	SIMDベクトルの成分となる型の特性
*/
#if !defined(TU_SIMD_TYPE_TRAITS_H)
#define TU_SIMD_TYPE_TRAITS_H

#include <cstdint>

namespace TU
{
namespace simd
{
/************************************************************************
*  type traits								*
************************************************************************/
template <class T>	struct type_traits_base;

template <>
struct type_traits_base<int8_t>
{
    using lower_type	= void;
    using upper_type	= int16_t;
    using mask_type	= uint8_t;
};
    
template <>
struct type_traits_base<int16_t>
{
    using lower_type	= int8_t;
    using upper_type	= int32_t;
    using mask_type	= uint16_t;
};
    
template <>
struct type_traits_base<int32_t>
{
    using lower_type	= int16_t;
    using upper_type	= int64_t;
    using mask_type	= uint32_t;
};
    
template <>
struct type_traits_base<int64_t>
{
    using lower_type	= int32_t;
    using upper_type	= void;
    using mask_type	= uint64_t;
};
    
template <>
struct type_traits_base<uint8_t>
{
    using lower_type	= void;
    using upper_type	= uint16_t;
    using mask_type	= uint8_t;
};
    
template <>
struct type_traits_base<uint16_t>
{
    using lower_type	= uint8_t;
    using upper_type	= uint32_t;
    using mask_type	= uint16_t;
};
    
template <>
struct type_traits_base<uint32_t>
{
    using lower_type	= uint16_t;
    using upper_type	= uint64_t;
    using mask_type	= uint32_t;
};
    
template <>
struct type_traits_base<uint64_t>
{
    using lower_type	= uint32_t;
    using upper_type	= void;
    using mask_type	= uint64_t;
};

template <>
struct type_traits_base<float>
{
    using lower_type	= void;
    using upper_type	= double;
};

template <>
struct type_traits_base<double>
{
    using lower_type	= float;
    using upper_type	= void;
};

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/x86/type_traits.h"
#elif defined(NEON)
#  include "TU/simd/arm/type_traits.h"
#endif

namespace TU
{
namespace simd
{
namespace detail
{
  template <class T> struct identity	{ using type = T; };
}	// namespace detail
    
template <class T>
using signed_type		= typename std::conditional_t<
						std::is_integral<T>::value,
						std::make_signed<T>,
						detail::identity<T> >::type;
template <class T>
using lower_type		= typename type_traits<T>::lower_type;
template <class T>
using upper_type		= typename type_traits<T>::upper_type;
template <class T>
using mask_type			= typename type_traits<T>::mask_type;
template <class T>
using complementary_type	= typename type_traits<T>::complementary_type;
template <class T>
using complementary_mask_type	= typename
				      type_traits<T>::complementary_mask_type;
template <class T>
using base_type			= typename type_traits<T>::base_type;
template <class T>
using integral_type		= std::conditional_t<
				      std::is_integral<T>::value,
				      T, complementary_type<T> >;
}	// namespace simd
}	// namespace TU
	
#endif	// !TU_SIMD_TYPE_TRAITS_H
