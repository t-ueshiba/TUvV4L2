/*
 *  $Id$
 */
#if !defined(__TU_SIMD_TYPE_TRAITS_H)
#define __TU_SIMD_TYPE_TRAITS_H

#include <sys/types.h>

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
    typedef int8_t		signed_type;
    typedef u_int8_t		unsigned_type;
    typedef void		lower_type;
    typedef int16_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<int16_t>
{
    typedef int16_t		signed_type;
    typedef u_int16_t		unsigned_type;
    typedef int8_t		lower_type;
    typedef int32_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<int32_t>
{
    typedef int32_t		signed_type;
    typedef u_int32_t		unsigned_type;
    typedef int16_t		lower_type;
    typedef int64_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<int64_t>
{
    typedef int64_t		signed_type;
    typedef u_int64_t		unsigned_type;
    typedef int32_t		lower_type;
    typedef void		upper_type;
    typedef double		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<u_int8_t>
{
    typedef int8_t		signed_type;
    typedef u_int8_t		unsigned_type;
    typedef void		lower_type;
    typedef u_int16_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<u_int16_t>
{
    typedef int16_t		signed_type;
    typedef u_int16_t		unsigned_type;
    typedef u_int8_t		lower_type;
    typedef u_int32_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<u_int32_t>
{
    typedef int32_t		signed_type;
    typedef u_int32_t		unsigned_type;
    typedef u_int16_t		lower_type;
    typedef u_int64_t		upper_type;
    typedef float		complementary_type;
    typedef complementary_type	complementary_mask_type;
};
    
template <>
struct type_traits_base<u_int64_t>
{
    typedef int64_t		signed_type;
    typedef u_int64_t		unsigned_type;
    typedef u_int32_t		lower_type;
    typedef void		upper_type;
    typedef double		complementary_type;
    typedef complementary_type	complementary_mask_type;
};

template <>
struct type_traits_base<float>
{
    typedef int32_t		signed_type;
    typedef u_int32_t		unsigned_type;
    typedef void		lower_type;
    typedef double		upper_type;
    typedef int32_t		complementary_type;
    typedef complementary_type	complementary_mask_type;
};

template <>
struct type_traits_base<double>
{
    typedef int32_t		signed_type;
    typedef u_int32_t		unsigned_type;
    typedef float		lower_type;
    typedef void		upper_type;
    typedef int32_t		complementary_type;
    typedef complementary_type	complementary_mask_type;
};

}	// namespace simd
}	// namespace TU

#if defined(MMX)
#  include "TU/simd/intel/type_traits.h"
#elif defined(NEON)
#  include "TU/simd/arm/type_traits.h"
#endif

namespace TU
{
namespace simd
{
template <class T>
using signed_type		= typename type_traits<T>::singed_type;
template <class T>
using unsigned_type		= typename type_traits<T>::unsinged_type;
template <class T>
using lower_type		= typename type_traits<T>::lower_type;
template <class T>
using upper_type		= typename type_traits<T>::upper_type;
template <class T>
using complementary_type	= typename type_traits<T>::complementary_type;
template <class T>
using complementary_mask_type	= typename
				      type_traits<T>::complementary_mask_type;
template <class T>
using base_type			= typename type_traits<T>::base_type;
    
}	// namespace simd
}	// namespace TU
	
#endif	// !__TU_SIMD_TYPE_TRAITS_H
