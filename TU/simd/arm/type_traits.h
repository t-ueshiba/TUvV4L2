/*
 *  $id$
 */
#if !defined(TU_SIMD_ARM_TYPE_TRAITS_H)
#define TU_SIMD_ARM_TYPE_TRAITS_H

#include <arm_neon.h>

namespace TU
{
namespace simd
{
/************************************************************************
*  type traits								*
************************************************************************/
template <class T>	struct type_traits;

template <>
struct type_traits<int8_t> : type_traits_base<int8_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= int8x8_t;
    using base_type			= int8x16_t;
};
    
template <>
struct type_traits<int16_t> : type_traits_base<int16_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= int16x4_t;
    using base_type			= int16x8_t;
};
    
template <>
struct type_traits<int32_t> : type_traits_base<int32_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= int32x2_t;
    using base_type			= int32x4_t;
};
    
template <>
struct type_traits<int64_t> : type_traits_base<int64_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= int64x1_t;
    using base_type			= int64x2_t;
};
    
template <>
struct type_traits<uint8_t> : type_traits_base<uint8_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= uint8x8_t;
    using base_type			= uint8x16_t;
};
    
template <>
struct type_traits<uint16_t> : type_traits_base<uint16_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= uint16x4_t;
    using base_type			= uint16x8_t;
};
    
template <>
struct type_traits<uint32_t> : type_traits_base<uint32_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= uint32x2_t;
    using base_type			= uint32x4_t;
};
    
template <>
struct type_traits<uint64_t> : type_traits_base<uint64_t>
{
    using complementary_type		= float;
    using complementary_mask_type	= complementary_type;
    using half_type			= uint64x1_t;
    using base_type			= uint64x2_t;
};

template <>
struct type_traits<float> : type_traits_base<float>
{
    using mask_type			= uint32_t;
    using complementary_type		= int32_t;
    using complementary_mask_type	= mask_type;
    using half_type			= float32x2_t;
    using base_type			= float32x4_t;
};

template <class T> using half_type = typename type_traits<T>::half_type;
    
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ARM_TYPE_TRAITS_H
