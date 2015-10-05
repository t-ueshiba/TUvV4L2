/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_ARCH_H)
#define __TU_SIMD_ARM_ARCH_H

/************************************************************************
*  Macros for constructing mnemonics of intrinsics			*
************************************************************************/
#define SIMD_SUFFIX(type)	SIMD_SUFFIX_##type

#define SIMD_SUFFIX_int8_t	_s8
#define SIMD_SUFFIX_int16_t	_s16
#define SIMD_SUFFIX_int32_t	_s32
#define SIMD_SUFFIX_int64_t	_s64
#define SIMD_SUFFIX_u_int8_t	_u8
#define SIMD_SUFFIX_u_int16_t	_u16
#define SIMD_SUFFIX_u_int32_t	_u32
#define SIMD_SUFFIX_u_int64_t	_u64
#define SIMD_SUFFIX_float	_f32
#define SIMD_SUFFIX_q		q
#define SIMD_SUFFIX_l		l
#define SIMD_SUFFIX_w		w
#define SIMD_SUFFIX_void

#define SIMD_CAT(op, to, from)		v##op##to##from
#define SIMD_MNEMONIC(op, to, from)	SIMD_CAT(op, to, from)

#define SIMD_FUNC(signature, op, args, to, from)			\
    inline signature							\
    {									\
	return SIMD_MNEMONIC(op, SIMD_SUFFIX(to), SIMD_SUFFIX(from))	\
	    args;							\
    }

#define SIMD_SPECIALIZED_FUNC(signature, op, args, to, from)		\
    template <> SIMD_FUNC(signature, op, args, to, from)

#define SIMD_UNARY_FUNC(func, op, type)					\
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x), op, (x), q, type)

#define SIMD_BINARY_FUNC(func, op, type)				\
    SIMD_SPECIALIZED_FUNC(vec<type> func(vec<type> x, vec<type> y),	\
			  op, (x, y), q, type)

#endif	// !__TU_SIMD_ARM_ARCH_H
