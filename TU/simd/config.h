/*!
  \file		config.h
  \author	Toshio UESHIBA
  \brief	SIMD命令のクラスを指定するマクロ
*/
#if !defined(TU_SIMD_CONFIG_H)
#define TU_SIMD_CONFIG_H

#if defined(AVX2)		// Core-i7 Haswell (2013)
#  define AVX
#endif
#if defined(AVX)		// Core-i7 Sandy-Bridge (2011)
#  define SSE4
#endif
#if defined(SSE4)		// Core2 with Penryn core(45nm)
#  define SSSE3
#endif
#if defined(SSSE3)		// Core2 (Jun. 2006)
#  define SSE3
#endif
#if defined(SSE3)		// Pentium-4 with Prescott core (Feb. 2004)
#  define SSE2
#endif
#if defined(SSE2)		// Pentium-4 (Nov. 2000)
#  define SSE
#endif
#if defined(SSE)		// Pentium-3 (Feb. 1999)
#  define MMX
#endif

#if defined(MMX) || defined(NEON)
#  define SIMD
#endif

#endif	// !TU_SIMD_CONFIG_H
