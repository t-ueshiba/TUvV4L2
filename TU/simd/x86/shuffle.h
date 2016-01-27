/*
 *  $Id$
 */
#if !defined(__TU_SIMD_X86_SHUFFLE_H)
#define __TU_SIMD_X86_SHUFFLE_H

namespace TU
{
namespace simd
{
/************************************************************************
*  Shuffle operators							*
************************************************************************/
//! 8つの成分を持つ整数ベクトルの下位4成分をシャッフルする．
/*!
  上位4成分は変化しない．
  \param I0	最下位に来る成分のindex (0 <= I0 < 4)
  \param I1	下から2番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から3番目に来る成分のindex (0 <= I2 < 4)
  \param I3	下から4番目に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> vec<T>
shuffle_low(vec<T> x)							;

//! 8つの成分を持つ整数ベクトルの上位4成分をシャッフルする．
/*!
  下位4成分は変化しない．
  \param I0	下から5番目に来る成分のindex (0 <= I0 < 4)
  \param I1	下から6番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から7番目に来る成分のindex (0 <= I2 < 4)
  \param I3	最上位に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> vec<T>
shuffle_high(vec<T> x)							;

//! 4つの成分を持つ整数ベクトルの成分をシャッフルする．
/*!
  \param I0	最下位に来る成分のindex (0 <= I0 < 4)
  \param I1	下から2番目に来る成分のindex (0 <= I1 < 4)
  \param I2	下から3番目に来る成分のindex (0 <= I2 < 4)
  \param I3	最上位に来る成分のindex (0 <= I3 < 4)
  \param x	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t I3, size_t I2, size_t I1, size_t I0, class T> vec<T>
shuffle(vec<T> x)							;

#define SIMD_SHUFFLE_LOW_HIGH_I4(type)					\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    SIMD_FUNC(vec<type> shuffle_low(vec<type> x),			\
	      shufflelo, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
	      void, type, SIMD_SIGNED)					\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    SIMD_FUNC(vec<type> shuffle_high(vec<type> x),			\
	      shufflehi, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
	      void, type, SIMD_SIGNED)
#define SIMD_SHUFFLE_I4(type)						\
    template <size_t I3, size_t I2, size_t I1, size_t I0>		\
    SIMD_FUNC(vec<type> shuffle(vec<type> x),				\
	      shuffle, (x, _MM_SHUFFLE(I3, I2, I1, I0)),		\
	      void, type, SIMD_SIGNED)

#if defined(SSE2)
  SIMD_SHUFFLE_I4(int32_t)
  SIMD_SHUFFLE_I4(uint32_t)
  SIMD_SHUFFLE_LOW_HIGH_I4(int16_t)
  SIMD_SHUFFLE_LOW_HIGH_I4(uint16_t)
#elif defined(SSE)
  SIMD_SHUFFLE_I4(int16_t)
  SIMD_SHUFFLE_I4(uint16_t)
#endif
  
#undef SIMD_SHUFFLE_LOW_HIGH_I4
#undef SIMD_SHUFFLE_I4

//! 4つの成分を持つ2つの浮動小数点数ベクトルの成分をシャッフルする．
/*!
  下位2成分はxから，上位2成分はyからそれぞれ選択される．
  \param Xl	最下位に来るベクトルxの成分のindex (0 <= Xl < 4)
  \param Xh	下から2番目に来るベクトルxの成分のindex (0 <= Xh < 4)
  \param Yl	下から3番目に来るベクトルyの成分のindex (0 <= Yl < 4)
  \param Yh	最上位に来るベクトルyの成分のindex (0 <= Yh < 4)
  \param x	シャッフルされるベクトル
  \param y	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t Yh, size_t Yl, size_t Xh, size_t Xl, class T> vec<T>
shuffle(vec<T> x, vec<T> y)						;

//! 2つの成分を持つ2つの浮動小数点数ベクトルの成分をシャッフルする．
/*!
  下位成分はxから，上位成分はyからそれぞれ選択される．
  \param X	下位に来るベクトルxの成分のindex (0 <= I0 < 2)
  \param Y	上位に来るベクトルyの成分のindex (0 <= I3 < 2)
  \param x	シャッフルされるベクトル
  \param y	シャッフルされるベクトル
  \return	シャッフルされたベクトル
*/
template <size_t Y, size_t X, class T> vec<T>
shuffle(vec<T> x, vec<T> y)						;

#define _MM_SHUFFLE4(i3, i2, i1, i0)					\
    (((i3) << 3) | ((i2) << 2) | ((i1) << 1) | (i0))
#define SIMD_SHUFFLE_F4(type)						\
    template <size_t Yh, size_t Yl, size_t Xh, size_t Xl>		\
    SIMD_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
	      shuffle, (x, y, _MM_SHUFFLE(Yh, Yl, Xh, Xl)),		\
	      void, type, SIMD_SUFFIX)
#define SIMD_SHUFFLE_D4(type)						\
    template <size_t Yh, size_t Yl, size_t Xh, size_t Xl>		\
    SIMD_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
	      shuffle, (x, y, _MM_SHUFFLE4(Yh, Yl, Xh, Xl)),		\
	      void, type, SIMD_SUFFIX)
#define SIMD_SHUFFLE_D2(type)						\
    template <size_t Y, size_t X>					\
    SIMD_FUNC(vec<type> shuffle(vec<type> x, vec<type> y),		\
	      shuffle, (x, y, _MM_SHUFFLE2(Y, X)),			\
	      void, type, SIMD_SUFFIX)

#if defined(SSE)
  SIMD_SHUFFLE_F4(float)
#  if defined(AVX)
  SIMD_SHUFFLE_D4(double)
#  elif defined(SSE2)
  SIMD_SHUFFLE_D2(double)
#  endif
#endif

#undef SIMD_SHUFFLE_D2
#undef SIMD_SHUFFLE_F4
  
}	// namespace simd
}	// namespace TU
#endif	// !__TU_SIMD_X86_SHUFFLE_H
