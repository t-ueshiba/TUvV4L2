/*
 *  $Id$
 */
#if !defined(TU_SIMD_X86_MISC_H)
#define TU_SIMD_X86_MISC_H

namespace TU
{
namespace simd
{
inline void	empty()					{ _mm_empty(); }
  
}	// namespace simd
}	// namespace TU
#endif	// !TU_SIMD_X86_MISC_H
