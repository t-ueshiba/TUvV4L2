/*
 *  $Id$
 */
#if !defined(__TU_SIMD_ARM_VEC_H)
#define __TU_SIMD_ARM_VEC_H

#include "TU/simd/arm/arch.h"

namespace TU
{
namespace simd
{
/************************************************************************
*  Constructors of vec<T>						*
************************************************************************/
template <class T> inline
vec<T>::vec(element_type a0,  element_type a1)
    :_base{a0, a1}					{}

template <class T> inline
vec<T>::vec(element_type a0,  element_type a1,
	    element_type a2,  element_type a3)
    :_base{a0, a1, a2, a3}				{}

template <class T> inline
vec<T>::vec(element_type a0,  element_type a1,
	    element_type a2,  element_type a3,
	    element_type a4,  element_type a5,
	    element_type a6,  element_type a7)
    :_base{a0, a1, a2, a3, a4, a5, a6, a7}		{}

template <class T> inline
vec<T>::vec(element_type a0,  element_type a1,
	    element_type a2,  element_type a3,
	    element_type a4,  element_type a5,
	    element_type a6,  element_type a7,
	    element_type a8,  element_type a9,
	    element_type a10, element_type a11,
	    element_type a12, element_type a13,
	    element_type a14, element_type a15)
    :_base{a0, a1, a2,  a3,  a4,  a5,  a6,  a7,
	a8, a9, a10, a11, a12, a13, a14, a15}		{}

#define SIMD_CONSTRUCTOR_1(type)					\
    template <> inline vec<type>::vec(type a)				\
	:_base(SIMD_MNEMONIC(dupq_n, , SIMD_SUFFIX(type))(a))	{}	\
    template <> inline vec<type>&					\
    vec<type>::operator =(type a)					\
    {									\
	_base = SIMD_MNEMONIC(dupq_n, , SIMD_SUFFIX(type))(a);		\
	return *this;							\
    }

SIMD_CONSTRUCTOR_1(int8_t)
SIMD_CONSTRUCTOR_1(int16_t)
SIMD_CONSTRUCTOR_1(int32_t)
SIMD_CONSTRUCTOR_1(int64_t)
SIMD_CONSTRUCTOR_1(u_int8_t)
SIMD_CONSTRUCTOR_1(u_int16_t)
SIMD_CONSTRUCTOR_1(u_int32_t)
SIMD_CONSTRUCTOR_1(u_int64_t)
SIMD_CONSTRUCTOR_1(float)

#undef SIMD_CONSTRUCTOR_1

}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_ARM_VEC_H
