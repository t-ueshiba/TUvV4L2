/*
 *  $Id$
 */
#if !defined(TU_SIMD_ARM_ALLOCATOR_H)
#define TU_SIMD_ARM_ALLOCATOR_H

#include <memory>

namespace TU
{
namespace simd
{
template <class T> using allocator = std::allocator<T>;
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_ARM_ALLOCATOR_H
