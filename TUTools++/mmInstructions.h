/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2002-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *  
 *  $Id: mmInstructions.h,v 1.15 2009-03-17 00:42:11 ueshiba Exp $
 */
#if !defined(__mmInstructions_h) && defined(__INTEL_COMPILER)
#define __mmInstructions_h

#include "TU/types.h"

#if defined(SSE4)
#  define SSSE3
#endif
#if defined(SSSE3)
#  define SSE3
#endif
#if defined(SSE3)
#  define SSE2
#endif
#if defined(SSE2)
#  define SSE
#endif
#if defined(SSE)
#  define MMX
#endif

#if defined(SSE4)		// Core2 with Penryn core(45nm)
#  include <smmintrin.h>
#  include <nmmintrin.h>
#elif defined(SSSE3)		// Core2 (Jun. 2006)
#  include <tmmintrin.h>
#elif defined(SSE3)		// Pentium-4 with Prescott core (Feb. 2004)
#  include <pmmintrin.h>
#elif defined(SSE2)		// Pentium-4 (Nov. 2000)
#  include <emmintrin.h>
#elif defined(SSE)		// Pentium-3 (Feb. 1999)
#  include <xmmintrin.h>
#elif defined(MMX)		// MMX Pentium
#  include <mmintrin.h>
#endif

#if defined(MMX)
namespace TU
{
/************************************************************************
*  型定義								*
************************************************************************/
#if defined(SSE2)
  typedef __m128i	mmBase;
#else
  typedef __m64		mmBase;
#endif
  template <class T>
  struct mmInt
  {
      typedef T		ElmType;
      enum		{ElmSiz = sizeof(ElmType),
			 NElms  = sizeof(mmBase)/sizeof(ElmType)};

      mmInt(mmBase m)	:_val(m)	{}
      operator const mmBase&()	const	{return _val;}
      operator mmBase&()		{return _val;}

      static u_int	floor(u_int n)	{return NElms*(n/NElms);}
      static u_int	ceil(u_int n)	{return (n == 0 ? 0 :
						 NElms*((n - 1)/NElms + 1));}
      
    private:
      mmBase		_val;
  };
    
  typedef mmInt<s_char>		mmInt8;
  typedef mmInt<u_char>		mmUInt8;
  typedef mmInt<short>		mmInt16;
  typedef mmInt<u_short>	mmUInt16;
  typedef mmInt<int>		mmInt32;
  typedef mmInt<u_int>		mmUInt32;
  typedef mmInt<int64>		mmInt64;
  typedef mmInt<u_int64>	mmUInt64;

  static const int		mmNBytes  = mmInt8::NElms,
				mmNWords  = mmInt16::NElms,
				mmNDWords = mmInt32::NElms,
				mmNQWords = mmInt64::NElms;
#if defined(SSE)
  struct mmFlt
  {
      typedef float	ElmType;
      enum		{ElmSiz = sizeof(ElmType),
			 NElms  = sizeof(__m128)/sizeof(ElmType)};

      mmFlt(__m128 m)	:_val(m)	{}
      operator const __m128&()	const	{return _val;}
      operator __m128&()		{return _val;}

      static u_int	floor(u_int n)	{return NElms*(n/NElms);}
      static u_int	ceil(u_int n)	{return (n == 0 ? 0 :
						 NElms*((n - 1)/NElms + 1));}

    private:
      __m128		_val;
  };
#endif
#if defined(SSE2)
  struct mmDbl
  {
      typedef double	ElmType;
      enum		{ElmSiz = sizeof(ElmType),
			 NElms  = sizeof(__m128)/sizeof(ElmType)};

      mmDbl(__m128d m)	:_val(m)	{}
      operator const __m128d&()	const	{return _val;}
      operator __m128d&()		{return _val;}

      static u_int	floor(u_int n)	{return NElms*(n/NElms);}
      static u_int	ceil(u_int n)	{return (n == 0 ? 0 :
						 NElms*((n - 1)/NElms + 1));}

    private:
      __m128d		_val;
  };
#endif

/************************************************************************
*  制御命令								*
************************************************************************/
  static inline void	mmEmpty()	{_mm_empty();}
    
/************************************************************************
*  Load/Store								*
************************************************************************/
  template <class T> static void  mmStoreRMost(typename T::ElmType* p, T x);
#if defined(SSE2)
  static inline mmInt8
  mmLoad(const s_char* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmUInt8
  mmLoad(const u_char* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmInt16
  mmLoad(const short* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmUInt16
  mmLoad(const u_short* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmInt32
  mmLoad(const int* p)			{return _mm_load_si128((mmBase*)p);}
  static inline mmUInt32
  mmLoad(const u_int* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmInt64
  mmLoad(const int64* p)		{return _mm_load_si128((mmBase*)p);}
  static inline mmUInt64
  mmLoad(const u_int64* p)		{return _mm_load_si128((mmBase*)p);}
#  if defined(SSE3)
  static inline mmInt8
  mmLoadU(const s_char* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmUInt8
  mmLoadU(const u_char* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmInt16
  mmLoadU(const short* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmUInt16
  mmLoadU(const u_short* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmInt32
  mmLoadU(const int* p)			{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmUInt32
  mmLoadU(const u_int* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmInt64
  mmLoadU(const int64* p)		{return _mm_lddqu_si128((mmBase*)p);}
  static inline mmUInt64
  mmLoadU(const u_int64* p)		{return _mm_lddqu_si128((mmBase*)p);}
#  else
  static inline mmInt8
  mmLoadU(const s_char* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmUInt8
  mmLoadU(const u_char* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmInt16
  mmLoadU(const short* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmUInt16
  mmLoadU(const u_short* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmInt32
  mmLoadU(const int* p)			{return _mm_loadu_si128((mmBase*)p);}
  static inline mmUInt32
  mmLoadU(const u_int* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmInt64
  mmLoadU(const int64* p)		{return _mm_loadu_si128((mmBase*)p);}
  static inline mmUInt64
  mmLoadU(const u_int64* p)		{return _mm_loadu_si128((mmBase*)p);}
#  endif
  template <class T> static inline void
  mmStore(typename T::ElmType* p, T x)	{_mm_store_si128((mmBase*)p, x);}
  template <class T> static inline void
  mmStoreU(typename T::ElmType* p, T x)	{_mm_storeu_si128((mmBase*)p, x);}
#else
  static inline mmInt8
  mmLoad(const s_char* p)		{return *((mmBase*)p);}
  static inline mmUInt8
  mmLoad(const u_char* p)		{return *((mmBase*)p);}
  static inline mmInt16
  mmLoad(const short* p)		{return *((mmBase*)p);}
  static inline mmUInt16
  mmLoad(const u_short* p)		{return *((mmBase*)p);}
  static inline mmInt32
  mmLoad(const int* p)			{return *((mmBase*)p);}
  static inline mmUInt32
  mmLoad(const u_int* p)		{return *((mmBase*)p);}
  static inline mmInt64
  mmLoad(const int64* p)		{return *((mmBase*)p);}
  static inline mmUInt64
  mmLoad(const u_int64* p)		{return *((mmBase*)p);}
  static inline mmInt8
  mmLoadU(const s_char* p)		{return *((mmBase*)p);}
  static inline mmUInt8
  mmLoadU(const u_char* p)		{return *((mmBase*)p);}
  static inline mmInt16
  mmLoadU(const short* p)		{return *((mmBase*)p);}
  static inline mmUInt16
  mmLoadU(const u_short* p)		{return *((mmBase*)p);}
  static inline mmInt32
  mmLoadU(const int* p)			{return *((mmBase*)p);}
  static inline mmUInt32
  mmLoadU(const u_int* p)		{return *((mmBase*)p);}
  static inline mmInt64
  mmLoadU(const int64* p)		{return *((mmBase*)p);}
  static inline mmUInt64
  mmLoadU(const u_int64* p)		{return *((mmBase*)p);}
  template <class T> static inline void
  mmStore(typename T::ElmType* p, T x)	{*((mmBase*)p) = x;}
  template <class T> static inline void
  mmStoreU(typename T::ElmType* p, T x)	{*((mmBase*)p) = x;}
#endif
#if defined(SSE)
  static inline mmFlt
  mmLoad(const float* p)		{return _mm_load_ps(p);}
  static inline mmFlt
  mmLoadU(const float* p)		{return _mm_loadu_ps(p);}
  static inline mmFlt
  mmLoadRMost(const float* p)		{return _mm_load_ss(p);}
  template <> inline void
  mmStore(float* p, mmFlt x)		{_mm_store_ps(p, x);}
  template <> inline void
  mmStoreU(float* p, mmFlt x)		{_mm_storeu_ps(p, x);}
  template <> inline void
  mmStoreRMost(float* p, mmFlt x)	{_mm_store_ss(p, x);}
#endif
#if defined(SSE2)
  static inline mmDbl
  mmLoad(const double* p)		{return _mm_load_pd(p);}
  static inline mmDbl
  mmLoadU(const double* p)		{return _mm_loadu_pd(p);}
  static inline mmDbl
  mmLoadRMost(const double* p)		{return _mm_load_sd(p);}
  template <> inline void
  mmStore(double* p, mmDbl x)		{_mm_store_pd(p, x);}
  template <> inline void
  mmStoreU(double* p, mmDbl x)		{_mm_storeu_pd(p, x);}
  template <> inline void
  mmStoreRMost(double* p, mmDbl x)	{_mm_store_sd(p, x);}
#endif

/************************************************************************
*  全要素に0をセット							*
************************************************************************/
#if defined(SSE2)
  template <class T> static inline T
  mmZero()				{return _mm_setzero_si128();}
#else
  template <class T> static inline T
  mmZero()				{return _mm_setzero_si64();}
#endif
#if defined(SSE)
  template <> inline mmFlt
  mmZero()				{return _mm_setzero_ps();}
#endif
#if defined(SSE2)
  template <> inline mmDbl
  mmZero()				{return _mm_setzero_pd();}
#endif

/************************************************************************
*  各要素を個別にセット							*
************************************************************************/
#if defined(SSE2)
  template <class T> static inline T
  mmSet(char b15, char b14, char b13,
	char b12, char b11, char b10,	
	char b9,  char b8,  char b7,
	char b6,  char b5,  char b4,
	char b3,  char b2,  char b1,
	char b0)			{return _mm_set_epi8(b15,b14,b13,b12,
							     b11,b10,b9, b8,
							     b7, b6, b5, b4,
							     b3, b2, b1, b0);}
  template <class T> static inline T
  mmSet(short w7, short w6, short w5,
	short w4, short w3, short w2,
	short w1, short w0)		{return _mm_set_epi16(w7, w6, w5, w4,
							      w3, w2, w1, w0);}
  template <class T> static inline T
  mmSet(int w, int z, int y, int x)	{return _mm_set_epi32(w, z, y, x);}
#else
  template <class T> static inline T
  mmSet(char b7, char b6, char b5,
	char b4, char b3, char b2,
	char b1, char b0)		{return _mm_set_pi8(b7, b6, b5, b4,
							    b3, b2, b1, b0);}
  template <class T> static inline T
  mmSet(short w, short z,
	short y, short x)		{return _mm_set_pi16(w, z, y, x);}
  template <class T> static inline T
  mmSet(int y, int x)			{return _mm_set_pi32(y, x);}
#endif
#if defined(SSE)
  static inline mmFlt
  mmSet(float w, float z,
	float y, float x)		{return _mm_set_ps(w, z, y, x);}
#endif
#if defined(SSE2)
  static inline mmDbl
  mmSet(double y, double x)		{return _mm_set_pd(y, x);}
#endif
	      
/************************************************************************
*  全要素に同一の値をセット						*
************************************************************************/
  template <class T> static T	mmSetAll(typename T::ElmType x);
#if defined(SSE2)
  template <> inline mmInt8
  mmSetAll<mmInt8>(s_char x)		{return _mm_set1_epi8(x);}
  template <> inline mmUInt8
  mmSetAll<mmUInt8>(u_char x)		{return _mm_set1_epi8(x);}
  template <> inline mmInt16
  mmSetAll<mmInt16>(short x)		{return _mm_set1_epi16(x);}
  template <> inline mmUInt16
  mmSetAll<mmUInt16>(u_short x)		{return _mm_set1_epi16(x);}
  template <> inline mmInt32
  mmSetAll<mmInt32>(int x)		{return _mm_set1_epi32(x);}
  template <> inline mmUInt32
  mmSetAll<mmUInt32>(u_int x)		{return _mm_set1_epi32(x);}
#else
  template <> inline mmInt8
  mmSetAll<mmInt8>(s_char x)		{return _mm_set1_pi8(x);}
  template <> inline mmUInt8
  mmSetAll<mmUInt8>(u_char x)		{return _mm_set1_pi8(x);}
  template <> inline mmInt16
  mmSetAll<mmInt16>(short x)		{return _mm_set1_pi16(x);}
  template <> inline mmUInt16
  mmSetAll<mmUInt16>(u_short x)		{return _mm_set1_pi16(x);}
  template <> inline mmInt32
  mmSetAll<mmInt32>(int x)		{return _mm_set1_pi32(x);}
  template <> inline mmUInt32
  mmSetAll<mmUInt32>(u_int x)		{return _mm_set1_pi32(x);}
#endif
#if defined(SSE)
  template <> inline mmFlt
  mmSetAll<mmFlt>(float x)		{return _mm_set1_ps(x);}
#endif
#if defined(SSE2)
  template <> inline mmDbl
  mmSetAll<mmDbl>(double x)		{return _mm_set1_pd(x);}
#endif

/************************************************************************
*  全要素にN番目の要素をセット						*
************************************************************************/
  template <class T, u_int N> static T	mmSetAll(T x);
#if defined(SSE)
  template <u_int N> inline mmFlt
  mmSetAll(mmFlt x)			{return _mm_shuffle_ps(x, x,
						  _MM_SHUFFLE(N, N, N, N));}
#endif
#if defined(SSE2)
  template <u_int N> inline mmInt32
  mmSetAll(mmInt32 x)			{return _mm_shuffle_epi32(x,
						  _MM_SHUFFLE(N, N, N, N));}
  template <u_int N> inline mmUInt32
  mmSetAll(mmUInt32 x)			{return _mm_shuffle_epi32(x,
						  _MM_SHUFFLE(N, N, N, N));}
  template <u_int N> inline mmDbl
  mmSetAll(mmDbl x)			{return _mm_shuffle_pd(x, x,
						  _MM_SHUFFLE2(N, N));}
#endif

/************************************************************************
*  右端に指定された値を，それ以外の要素に0をセット			*
************************************************************************/
  template <class T> static T	mmSetRMost(typename T::ElmType x);
#if defined(SSE2)
  template <> inline mmInt8
  mmSetRMost<mmInt8>(s_char x)		{return _mm_set_epi8(0, 0, 0, 0,
							     0, 0, 0, 0,
							     0, 0, 0, 0,
							     0, 0, 0, x);}
  template <> inline mmUInt8
  mmSetRMost<mmUInt8>(u_char x)
					{return _mm_set_epi8(0, 0, 0, 0,
							     0, 0, 0, 0,
							     0, 0, 0, 0,
							     0, 0, 0, x);}
  template <> inline mmInt16
  mmSetRMost<mmInt16>(short x)		{return _mm_set_epi16(0, 0, 0, 0,
							      0, 0, 0, x);}
  template <> inline mmUInt16
  mmSetRMost<mmUInt16>(u_short x)	{return _mm_set_epi16(0, 0, 0, 0,
							      0, 0, 0, x);}
  template <> inline mmInt32
  mmSetRMost<mmInt32>(int x)		{return _mm_cvtsi32_si128(x);}
  template <> inline mmUInt32
  mmSetRMost<mmUInt32>(u_int x)		{return _mm_cvtsi32_si128(x);}
#else
  template <> inline mmInt8
  mmSetRMost<mmInt8>(s_char x)		{return _mm_set_pi8(0, 0, 0, 0,
							    0, 0, 0, x);}
  template <> inline mmUInt8
  mmSetRMost<mmUInt8>(u_char x)		{return _mm_set_pi8(0, 0, 0, 0,
							    0, 0, 0, x);}
  template <> inline mmInt16
  mmSetRMost<mmInt16>(short x)		{return _mm_set_pi16(0, 0, 0, x);}
  template <> inline mmUInt16
  mmSetRMost<mmUInt16>(u_short x)	{return _mm_set_pi16(0, 0, 0, x);}
  template <> inline mmInt32
  mmSetRMost<mmInt32>(int x)		{return _mm_set_pi32(0, x);}
  template <> inline mmUInt32
  mmSetRMost<mmUInt32>(u_int x)		{return _mm_set_pi32(0, x);}
#endif
#if defined(SSE)
  template <> inline mmFlt
  mmSetRMost<mmFlt>(float x)		{return _mm_set_ss(x);}
#endif
#if defined(SSE2)
  template <> inline mmDbl
  mmSetRMost<mmDbl>(double x)		{return _mm_set_sd(x);}
#endif

/************************************************************************
*  要素のシフト								*
************************************************************************/
#if defined(SSE2)
  template <u_int N, class T> static inline mmInt<T>
  mmShiftElmL(mmInt<T> x)		{return _mm_slli_si128(
						  x, N*mmInt<T>::ElmSiz);}
  template <u_int N, class T> static inline mmInt<T>
  mmShiftElmR(mmInt<T> x)		{return _mm_srli_si128(
						  x, N*mmInt<T>::ElmSiz);}
  template <u_int N> static inline mmFlt
  mmShiftElmL(mmFlt x)			{return _mm_castsi128_ps(
						  _mm_slli_si128(
						    _mm_castps_si128(x),
						    N*mmFlt::ElmSiz));}
  template <u_int N> static inline mmFlt
  mmShiftElmR(mmFlt x)			{return _mm_castsi128_ps(
						  _mm_srli_si128(
						    _mm_castps_si128(x),
						    N*mmFlt::ElmSiz));}
  template <u_int N> static inline mmDbl
  mmShiftElmL(mmDbl x)			{return _mm_castsi128_pd(
						  _mm_slli_si128(
						    _mm_castpd_si128(x),
						    N*mmDbl::ElmSiz));}
  template <u_int N> static inline mmDbl
  mmShiftElmR(mmDbl x, u_int N=1)	{return _mm_castsi128_pd(
						  _mm_srli_si128(
						    _mm_castpd_si128(x),
						    N*mmDbl::ElmSiz));}
#else
  template <u_int N, class T> static inline mmInt<T>
  mmShiftElmL(mmInt<T> x)		{return _mm_slli_si64(
						  x, 8*N*mmInt<T>::ElmSiz);}
  template <u_int N, class T> static inline mmInt<T>
  mmShiftElmR(mmInt<T> x)		{return _mm_srli_si64(
						  x, 8*N*mmInt<T>::ElmSiz);}
#endif

/************************************************************************
*  左端の要素が右端に来るまでシフト					*
************************************************************************/
  template <class T> static inline T
  mmShiftLMostToRMost(T x)		{return mmShiftElmR<T::NElms - 1>(x);}

/************************************************************************
*  右端の要素が左端に来るまでシフト					*
************************************************************************/
  template <class T> static inline T
  mmShiftRMostToLMost(T x)		{return mmShiftElmL<T::NElms - 1>(x);}

/************************************************************************
*  回転と逆転								*
************************************************************************/
#if defined(SSE)
  static inline mmFlt
  mmRotateElmL(mmFlt x)		{return _mm_shuffle_ps(
					  x, x, _MM_SHUFFLE(2, 1, 0, 3));}
  static inline mmFlt
  mmRotateElmR(mmFlt x)		{return _mm_shuffle_ps(
					  x, x, _MM_SHUFFLE(0, 3, 2, 1));}
  static inline mmFlt
  mmReverseElm(mmFlt x)		{return _mm_shuffle_ps(
					  x, x, _MM_SHUFFLE(0, 1, 2, 3));}
#endif
#if defined(SSE2)
  static inline mmInt32
  mmRotateElmL(mmInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(2, 1, 0, 3));}
  static inline mmInt32
  mmRotateElmR(mmInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(0, 3, 2, 1));}
  static inline mmInt32
  mmReverseElm(mmInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(0, 1, 2, 3));}
  static inline mmUInt32
  mmRotateElmL(mmUInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(2, 1, 0, 3));}
  static inline mmUInt32
  mmRotateElmR(mmUInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(0, 3, 2, 1));}
  static inline mmUInt32
  mmReverseElm(mmUInt32 x)	{return _mm_shuffle_epi32(
					  x, _MM_SHUFFLE(0, 1, 2, 3));}
  static inline mmDbl
  mmRotateElmL(mmDbl x)		{return _mm_shuffle_pd(x, x,
						       _MM_SHUFFLE2(0, 1));}
  static inline mmDbl
  mmRotateElmR(mmDbl x)		{return mmRotateElmL(x);}
  static inline mmDbl
  mmReverseElm(mmDbl x)		{return mmRotateElmL(x);}
#endif
  
/************************************************************************
*  xの右端要素をyの右端要素に置き換え					*
************************************************************************/
#if defined(SSE)
  static inline mmFlt
  mmReplaceRMost(mmFlt x, mmFlt y)	{return _mm_move_ss(x, y);}
#endif
#if defined(SSE2)
  static inline mmDbl
  mmReplaceRMost(mmDbl x, mmDbl y)	{return _mm_move_sd(x, y);}
#endif
    
/************************************************************************
*  下半分／上半分について要素を2つ複製					*
************************************************************************/
#if defined(SSE2)
  static inline mmInt8
  mmDupL(mmInt8 x)			{return _mm_unpacklo_epi8(x, x);}
  static inline mmInt8
  mmDupH(mmInt8 x)			{return _mm_unpackhi_epi8(x, x);}
  static inline mmUInt8
  mmDupL(mmUInt8 x)			{return _mm_unpacklo_epi8(x, x);}
  static inline mmUInt8
  mmDupH(mmUInt8 x)			{return _mm_unpackhi_epi8(x, x);}
  static inline mmInt16
  mmDupL(mmInt16 x)			{return _mm_unpacklo_epi16(x, x);}
  static inline mmInt16
  mmDupH(mmInt16 x)			{return _mm_unpackhi_epi16(x, x);}
  static inline mmUInt16
  mmDupL(mmUInt16 x)			{return _mm_unpacklo_epi16(x, x);}
  static inline mmUInt16
  mmDupH(mmUInt16 x)			{return _mm_unpackhi_epi16(x, x);}
  static inline mmInt32
  mmDupL(mmInt32 x)			{return _mm_unpacklo_epi32(x, x);}
  static inline mmInt32
  mmDupH(mmInt32 x)			{return _mm_unpackhi_epi32(x, x);}
  static inline mmUInt32
  mmDupL(mmUInt32 x)			{return _mm_unpacklo_epi32(x, x);}
  static inline mmUInt32
  mmDupH(mmUInt32 x)			{return _mm_unpackhi_epi32(x, x);}
#else
  static inline mmInt8
  mmDupL(mmInt8 x)			{return _mm_unpacklo_pi8(x, x);}
  static inline mmInt8
  mmDupH(mmInt8 x)			{return _mm_unpackhi_pi8(x, x);}
  static inline mmUInt8
  mmDupL(mmUInt8 x)			{return _mm_unpacklo_pi8(x, x);}
  static inline mmUInt8
  mmDupH(mmUInt8 x)			{return _mm_unpackhi_pi8(x, x);}
  static inline mmInt16
  mmDupL(mmInt16 x)			{return _mm_unpacklo_pi16(x, x);}
  static inline mmInt16
  mmDupH(mmInt16 x)			{return _mm_unpackhi_pi16(x, x);}
  static inline mmUInt16
  mmDupL(mmUInt16 x)			{return _mm_unpacklo_pi16(x, x);}
  static inline mmUInt16
  mmDupH(mmUInt16 x)			{return _mm_unpackhi_pi16(x, x);}
  static inline mmInt32
  mmDupL(mmInt32 x)			{return _mm_unpacklo_pi32(x, x);}
  static inline mmInt32
  mmDupH(mmInt32 x)			{return _mm_unpackhi_pi32(x, x);}
  static inline mmUInt32
  mmDupL(mmUInt32 x)			{return _mm_unpacklo_pi32(x, x);}
  static inline mmUInt32
  mmDupH(mmUInt32 x)			{return _mm_unpackhi_pi32(x, x);}
#endif
#if defined(SSE)
  static inline mmFlt
  mmDupL(mmFlt x)			{return _mm_unpacklo_ps(x, x);}
  static inline mmFlt
  mmDupH(mmFlt x)			{return _mm_unpackhi_ps(x, x);}
#endif
#if defined(SSE2)
  static inline mmDbl
  mmDupL(mmDbl x)			{return _mm_unpacklo_pd(x, x);}
  static inline mmDbl
  mmDupH(mmDbl x)			{return _mm_unpackhi_pd(x, x);}
#endif
    
/************************************************************************
*  1/4ずつのそれぞれについて要素を4つ複製				*
************************************************************************/
#if defined(SSE2)
  static inline mmInt8
  mmQuad0(mmInt8 x)			{x = _mm_unpacklo_epi8(x, x);
					 return _mm_unpacklo_epi16(x, x);}
  static inline mmInt8
  mmQuad1(mmInt8 x)			{x = _mm_unpacklo_epi8(x, x);
					 return _mm_unpackhi_epi16(x, x);}
  static inline mmInt8
  mmQuad2(mmInt8 x)			{x = _mm_unpackhi_epi8(x, x);
					 return _mm_unpacklo_epi16(x, x);}
  static inline mmInt8
  mmQuad3(mmInt8 x)			{x = _mm_unpackhi_epi8(x, x);
					 return _mm_unpackhi_epi16(x, x);}

  static inline mmUInt8
  mmQuad0(mmUInt8 x)			{x = _mm_unpacklo_epi8(x, x);
					 return _mm_unpacklo_epi16(x, x);}
  static inline mmUInt8
  mmQuad1(mmUInt8 x)			{x = _mm_unpacklo_epi8(x, x);
					 return _mm_unpackhi_epi16(x, x);}
  static inline mmUInt8
  mmQuad2(mmUInt8 x)			{x = _mm_unpackhi_epi8(x, x);
					 return _mm_unpacklo_epi16(x, x);}
  static inline mmUInt8
  mmQuad3(mmUInt8 x)			{x = _mm_unpackhi_epi8(x, x);
					 return _mm_unpackhi_epi16(x, x);}

  static inline mmInt16
  mmQuad0(mmInt16 x)			{x = _mm_unpacklo_epi16(x, x);
					 return _mm_unpacklo_epi32(x, x);}
  static inline mmInt16
  mmQuad1(mmInt16 x)			{x = _mm_unpacklo_epi16(x, x);
					 return _mm_unpackhi_epi32(x, x);}
  static inline mmInt16
  mmQuad2(mmInt16 x)			{x = _mm_unpackhi_epi16(x, x);
					 return _mm_unpacklo_epi32(x, x);}
  static inline mmInt16
  mmQuad3(mmInt16 x)			{x = _mm_unpackhi_epi16(x, x);
					 return _mm_unpackhi_epi32(x, x);}

  static inline mmUInt16
  mmQuad0(mmUInt16 x)			{x = _mm_unpacklo_epi16(x, x);
					 return _mm_unpacklo_epi32(x, x);}
  static inline mmUInt16
  mmQuad1(mmUInt16 x)			{x = _mm_unpacklo_epi16(x, x);
					 return _mm_unpackhi_epi32(x, x);}
  static inline mmUInt16
  mmQuad2(mmUInt16 x)			{x = _mm_unpackhi_epi16(x, x);
					 return _mm_unpacklo_epi32(x, x);}
  static inline mmUInt16
  mmQuad3(mmUInt16 x)			{x = _mm_unpackhi_epi16(x, x);
					 return _mm_unpackhi_epi32(x, x);}

  static inline mmInt32
  mmQuad0(mmInt32 x)			{return mmSetAll<0>(x);}
  static inline mmInt32
  mmQuad1(mmInt32 x)			{return mmSetAll<1>(x);}
  static inline mmInt32
  mmQuad2(mmInt32 x)			{return mmSetAll<2>(x);}
  static inline mmInt32
  mmQuad3(mmInt32 x)			{return mmSetAll<3>(x);}

  static inline mmUInt32
  mmQuad0(mmUInt32 x)			{return mmSetAll<0>(x);}
  static inline mmUInt32
  mmQuad1(mmUInt32 x)			{return mmSetAll<1>(x);}
  static inline mmUInt32
  mmQuad2(mmUInt32 x)			{return mmSetAll<2>(x);}
  static inline mmUInt32
  mmQuad3(mmUInt32 x)			{return mmSetAll<3>(x);}
#else
  static inline mmInt8
  mmQuad0(mmInt8 x)			{x = _mm_unpacklo_pi8(x, x);
					 return _mm_unpacklo_pi16(x, x);}
  static inline mmInt8
  mmQuad1(mmInt8 x)			{x = _mm_unpacklo_pi8(x, x);
					 return _mm_unpackhi_pi16(x, x);}
  static inline mmInt8
  mmQuad2(mmInt8 x)			{x = _mm_unpackhi_pi8(x, x);
					 return _mm_unpacklo_pi16(x, x);}
  static inline mmInt8
  mmQuad3(mmInt8 x)			{x = _mm_unpackhi_pi8(x, x);
					 return _mm_unpackhi_pi16(x, x);}

  static inline mmUInt8
  mmQuad0(mmUInt8 x)			{x = _mm_unpacklo_pi8(x, x);
					 return _mm_unpacklo_pi16(x, x);}
  static inline mmUInt8
  mmQuad1(mmUInt8 x)			{x = _mm_unpacklo_pi8(x, x);
					 return _mm_unpackhi_pi16(x, x);}
  static inline mmUInt8
  mmQuad2(mmUInt8 x)			{x = _mm_unpackhi_pi8(x, x);
					 return _mm_unpacklo_pi16(x, x);}
  static inline mmUInt8
  mmQuad3(mmUInt8 x)			{x = _mm_unpackhi_pi8(x, x);
					 return _mm_unpackhi_pi16(x, x);}

  static inline mmInt16
  mmQuad0(mmInt16 x)			{x = _mm_unpacklo_pi16(x, x);
					 return _mm_unpacklo_pi32(x, x);}
  static inline mmInt16
  mmQuad1(mmInt16 x)			{x = _mm_unpacklo_pi16(x, x);
					 return _mm_unpackhi_pi32(x, x);}
  static inline mmInt16
  mmQuad2(mmInt16 x)			{x = _mm_unpackhi_pi16(x, x);
					 return _mm_unpacklo_pi32(x, x);}
  static inline mmInt16
  mmQuad3(mmInt16 x)			{x = _mm_unpackhi_pi16(x, x);
					 return _mm_unpackhi_pi32(x, x);}

  static inline mmUInt16
  mmQuad0(mmUInt16 x)			{x = _mm_unpacklo_pi16(x, x);
					 return _mm_unpacklo_pi32(x, x);}
  static inline mmUInt16
  mmQuad1(mmUInt16 x)			{x = _mm_unpacklo_pi16(x, x);
					 return _mm_unpackhi_pi32(x, x);}
  static inline mmUInt16
  mmQuad2(mmUInt16 x)			{x = _mm_unpackhi_pi16(x, x);
					 return _mm_unpacklo_pi32(x, x);}
  static inline mmUInt16
  mmQuad3(mmUInt16 x)			{x = _mm_unpackhi_pi16(x, x);
					 return _mm_unpackhi_pi32(x, x);}
#endif
#if defined(SSE)
  static inline mmFlt
  mmQuad0(mmFlt x)			{return mmSetAll<0>(x);}
  static inline mmFlt
  mmQuad1(mmFlt x)			{return mmSetAll<1>(x);}
  static inline mmFlt
  mmQuad2(mmFlt x)			{return mmSetAll<2>(x);}
  static inline mmFlt
  mmQuad3(mmFlt x)			{return mmSetAll<3>(x);}
#endif
    
/************************************************************************
*  N番目の要素の取り出し						*
************************************************************************/
#if defined(SSE)
#  if defined(SSE2)
  template <u_int N> static inline int
  mmNth(mmInt16 x)			{return _mm_extract_epi16(x, N);}
  template <u_int N> static inline int
  mmNth(mmUInt16 x)			{return _mm_extract_epi16(x, N);}
#  else
  template <u_int N> static inline int
  mmNth(mmInt16 x)			{return _mm_extract_pi16(x, N);}
  template <u_int N> static inline int
  mmNth(mmUInt16 x)			{return _mm_extract_pi16(x, N);}
#  endif
#endif
    
/************************************************************************
*  型変換								*
************************************************************************/
  template <class S, class T> static S	mmCvt(T x);
  template <class S, class T> static S	mmCvtH(T x);
  template <class S, class T> static S	mmCvt(T x, T y);
#if defined(SSE2)
#  if defined(SSE4)
// s_char -> short, int, int64
  template <> inline mmInt16
  mmCvt<mmInt16>(mmInt8 x)		{return _mm_cvtepi8_epi16(x);}
  template <> inline mmInt32
  mmCvt<mmInt32>(mmInt8 x)		{return _mm_cvtepi8_epi32(x);}
  template <> inline mmInt64
  mmCvt<mmInt64>(mmInt8 x)		{return _mm_cvtepi8_epi64(x);}

// u_char -> short, u_short, int, u_int, int64, u_int64
  template <> inline mmInt16
  mmCvt<mmInt16>(mmUInt8 x)		{return _mm_cvtepu8_epi16(x);}
  template <> inline mmUInt16
  mmCvt<mmUInt16>(mmUInt8 x)		{return _mm_cvtepu8_epi16(x);}
  template <> inline mmInt32
  mmCvt<mmInt32>(mmUInt8 x)		{return _mm_cvtepu8_epi32(x);}
  template <> inline mmUInt32
  mmCvt<mmUInt32>(mmUInt8 x)		{return _mm_cvtepu8_epi32(x);}
  template <> inline mmInt64
  mmCvt<mmInt64>(mmUInt8 x)		{return _mm_cvtepu8_epi64(x);}
  template <> inline mmUInt64
  mmCvt<mmUInt64>(mmUInt8 x)		{return _mm_cvtepu8_epi64(x);}

// short -> int, int64
  template <> inline mmInt32
  mmCvt<mmInt32>(mmInt16 x)		{return _mm_cvtepi16_epi32(x);}
  template <> inline mmInt64
  mmCvt<mmInt64>(mmInt16 x)		{return _mm_cvtepi16_epi64(x);}

// u_short -> int, u_int, int64, u_int64
  template <> inline mmInt32
  mmCvt<mmInt32>(mmUInt16 x)		{return _mm_cvtepu16_epi32(x);}
  template <> inline mmUInt32
  mmCvt<mmUInt32>(mmUInt16 x)		{return _mm_cvtepu16_epi32(x);}
  template <> inline mmInt64
  mmCvt<mmInt64>(mmUInt16 x)		{return _mm_cvtepu16_epi64(x);}
  template <> inline mmUInt64
  mmCvt<mmUInt64>(mmUInt16 x)		{return _mm_cvtepu16_epi64(x);}

// int -> int64
  template <> inline mmInt64
  mmCvt<mmInt64>(mmInt32 x)		{return _mm_cvtepi32_epi64(x);}
    
// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvt<mmInt64>(mmUInt32 x)		{return _mm_cvtepu32_epi64(x);}
  template <> inline mmUInt64
  mmCvt<mmUInt64>(mmUInt32 x)		{return _mm_cvtepu32_epi64(x);}
#  else	// !SSE4
// u_char -> short, u_short
  template <> inline mmInt16
  mmCvt<mmInt16>(mmUInt8 x)		{return _mm_unpacklo_epi8(
						  x, mmZero<mmUInt8>());}
  template <> inline mmUInt16
  mmCvt<mmUInt16>(mmUInt8 x)		{return _mm_unpacklo_epi8(
						  x, mmZero<mmUInt8>());}
  
// u_short -> int, u_int
  template <> inline mmInt32
  mmCvt<mmInt32>(mmUInt16 x)		{return _mm_unpacklo_epi16(
						  x, mmZero<mmUInt16>());}
  template <> inline mmUInt32
  mmCvt<mmUInt32>(mmUInt16 x)		{return _mm_unpacklo_epi16(
						  x, mmZero<mmUInt16>());}

// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvt<mmInt64>(mmUInt32 x)		{return _mm_unpacklo_epi32(
						  x, mmZero<mmUInt32>());}
  template <> inline mmUInt64
  mmCvt<mmUInt64>(mmUInt32 x)		{return _mm_unpacklo_epi32(
						  x, mmZero<mmUInt32>());}
#  endif
// u_char -> short, u_short
  template <> inline mmInt16
  mmCvtH<mmInt16>(mmUInt8 x)		{return _mm_unpackhi_epi8(
						  x, mmZero<mmUInt8>());}
  template <> inline mmUInt16
  mmCvtH<mmUInt16>(mmUInt8 x)		{return _mm_unpackhi_epi8(
						  x, mmZero<mmUInt8>());}

// u_short -> int, u_int
  template <> inline mmInt32
  mmCvtH<mmInt32>(mmUInt16 x)		{return _mm_unpackhi_epi16(
						  x, mmZero<mmUInt16>());}
  template <> inline mmUInt32
  mmCvtH<mmUInt32>(mmUInt16 x)		{return _mm_unpackhi_epi16(
						  x, mmZero<mmUInt16>());}

// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvtH<mmInt64>(mmUInt32 x)		{return _mm_unpackhi_epi32(
						  x, mmZero<mmUInt16>());}
  template <> inline mmUInt64
  mmCvtH<mmUInt64>(mmUInt32 x)		{return _mm_unpackhi_epi32(
						  x, mmZero<mmUInt16>());}

// short, u_short -> s_char
  template <> inline mmInt8
  mmCvt<mmInt8>(mmInt16 x, mmInt16 y)	{return _mm_packs_epi16(x, y);}
  template <> inline mmInt8
  mmCvt<mmInt8>(mmUInt16 x, mmUInt16 y)	{return _mm_packs_epi16(x, y);}
  
// short, u_short -> u_char
  template <> inline mmUInt8
  mmCvt<mmUInt8>(mmInt16 x, mmInt16 y)	{return _mm_packus_epi16(x, y);}
  template <> inline mmUInt8
  mmCvt<mmUInt8>(mmUInt16 x, mmUInt16 y){return _mm_packus_epi16(x, y);}
  
// int, u_int -> short
  template <> inline mmInt16
  mmCvt<mmInt16>(mmInt32 x, mmInt32 y)	{return _mm_packs_epi32(x, y);}
  template <> inline mmInt16
  mmCvt<mmInt16>(mmUInt32 x, mmUInt32 y){return _mm_packs_epi32(x, y);}
#else
// u_char -> short, u_short
  template <> inline mmInt16
  mmCvt<mmInt16>(mmUInt8 x)		{return _mm_unpacklo_pi8(
						  x, mmZero<mmUInt8>());}
  template <> inline mmUInt16
  mmCvt<mmUInt16>(mmUInt8 x)		{return _mm_unpacklo_pi8(
						  x, mmZero<mmUInt8>());}
  template <> inline mmInt16
  mmCvtH<mmInt16>(mmUInt8 x)		{return _mm_unpackhi_pi8(
						  x, mmZero<mmUInt8>());}
  template <> inline mmUInt16
  mmCvtH<mmUInt16>(mmUInt8 x)		{return _mm_unpackhi_pi8(
						  x, mmZero<mmUInt8>());}

// u_short -> int, u_int
  template <> inline mmInt32
  mmCvt<mmInt32>(mmUInt16 x)		{return _mm_unpacklo_pi16(
						  x, mmZero<mmUInt16>());}
  template <> inline mmUInt32
  mmCvt<mmUInt32>(mmUInt16 x)		{return _mm_unpacklo_pi16(
						  x, mmZero<mmUInt16>());}
  template <> inline mmInt32
  mmCvtH<mmInt32>(mmUInt16 x)		{return _mm_unpackhi_pi16(
						  x, mmZero<mmUInt16>());}
  template <> inline mmUInt32
  mmCvtH<mmUInt32>(mmUInt16 x)		{return _mm_unpackhi_pi16(
						  x, mmZero<mmUInt16>());}

// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvt<mmInt64>(mmUInt32 x)		{return _mm_unpacklo_pi32(
						  x, mmZero<mmUInt32>());}
  template <> inline mmUInt64
  mmCvt<mmUInt64>(mmUInt32 x)		{return _mm_unpacklo_pi32(
						  x, mmZero<mmUInt32>());}
  template <> inline mmInt64
  mmCvtH<mmInt64>(mmUInt32 x)		{return _mm_unpackhi_pi32(
						  x, mmZero<mmUInt32>());}
  template <> inline mmUInt64
  mmCvtH<mmUInt64>(mmUInt32 x)		{return _mm_unpackhi_pi32(
						  x, mmZero<mmUInt32>());}

// short, u_short -> s_char
  template <> inline mmInt8
  mmCvt<mmInt8>(mmInt16 x, mmInt16 y)	{return _mm_packs_pi16(x, y);}
  template <> inline mmInt8
  mmCvt<mmInt8>(mmUInt16 x, mmUInt16 y)	{return _mm_packs_pi16(x, y);}
  
// short, u_short -> u_char
  template <> inline mmUInt8
  mmCvt<mmUInt8>(mmInt16 x, mmInt16 y)	{return _mm_packs_pu16(x, y);}
  template <> inline mmUInt8
  mmCvt<mmUInt8>(mmUInt16 x, mmUInt16 y){return _mm_packs_pu16(x, y);}
  
// int, u_int -> short
  template <> inline mmInt16
  mmCvt<mmInt16>(mmInt32 x, mmInt32 y)	{return _mm_packs_pi32(x, y);}
  template <> inline mmInt16
  mmCvt<mmInt16>(mmUInt32 x, mmUInt32 y){return _mm_packs_pi32(x, y);}
#endif
#if defined(SSE2)
// s_char, u_char, short, u_short, int -> float
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt8 x)		{return _mm_cvtpi8_ps(
						  _mm_movepi64_pi64(x));}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmUInt8 x)		{return _mm_cvtpu8_ps(
						  _mm_movepi64_pi64(x));}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt16 x)		{return _mm_cvtpi16_ps(
						  _mm_movepi64_pi64(x));}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmUInt16 x)		{return _mm_cvtpu16_ps(
						  _mm_movepi64_pi64(x));}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt32 x)		{return _mm_cvtepi32_ps(x);}

// float -> s_char, short, int
  template <> inline mmInt8
  mmCvt<mmInt8>(mmFlt x)		{return _mm_movpi64_epi64(
						  _mm_cvtps_pi8(x));}
  template <> inline mmInt16
  mmCvt<mmInt16>(mmFlt x)		{return _mm_movpi64_epi64(
						  _mm_cvtps_pi16(x));}
  template <> inline mmInt32
  mmCvt<mmInt32>(mmFlt x)		{return _mm_cvtps_epi32(x);}

// int, float -> double
  template <> inline mmDbl
  mmCvt<mmDbl>(mmInt32 x)		{return _mm_cvtepi32_pd(x);}
  template <> inline mmDbl
  mmCvt<mmDbl>(mmFlt x)			{return _mm_cvtps_pd(x);}

// double -> int, float
  template <> inline mmInt32
  mmCvt<mmInt32>(mmDbl x)		{return _mm_cvtpd_epi32(x);}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmDbl x)			{return _mm_cvtpd_ps(x);}
#elif defined(SSE)
// s_char, u_char, short, u_short, int, int64 -> float
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt8 x)		{return _mm_cvtpi8_ps(x);}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmUInt8 x)		{return _mm_cvtpu8_ps(x);}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt16 x)		{return _mm_cvtpi16_ps(x);}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmUInt16 x)		{return _mm_cvtpu16_ps(x);}
  template <> inline mmFlt
  mmCvt<mmFlt>(mmInt32 x)		{return
					   _mm_cvtpi32_ps(mmZero<mmFlt>(), x);}

// float -> s_char, short, int, int64
  template <> inline mmInt8
  mmCvt<mmInt8>(mmFlt x)		{return _mm_cvtps_pi8(x);}
  template <> inline mmInt16
  mmCvt<mmInt16>(mmFlt x)		{return _mm_cvtps_pi16(x);}
  template <> inline mmInt32
  mmCvt<mmInt32>(mmFlt x)		{return _mm_cvtps_pi32(x);}
#endif

/************************************************************************
*  キャスト								*
************************************************************************/
#if defined(SSE2)
  template <class T> static inline T
  mmCast(mmFlt x)			{return _mm_castps_si128(x);}
  template <class T> static inline T
  mmCast(mmDbl x)			{return _mm_castpd_si128(x);}
  template <class T> static inline mmFlt
  mmCastToFlt(mmInt<T> x)		{return _mm_castsi128_ps(x);}
  template <class T> static inline mmDbl
  mmCastToDbl(mmInt<T> x)		{return _mm_castsi128_pd(x);}
#endif
    
/************************************************************************
*  マスクの型変換							*
************************************************************************/
  template <class S, class T> static S	mmCvtMask(T x);
  template <class S, class T> static S	mmCvtMaskH(T x);
  template <class S, class T> static S	mmCvtMask(T x, T y);
#if defined(SSE2)
// u_char -> short, u_short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmUInt8 x)		{return _mm_unpacklo_epi8(x, x);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmUInt8 x)	{return _mm_unpacklo_epi8(x, x);}
  template <> inline mmInt16
  mmCvtMaskH<mmInt16>(mmUInt8 x)	{return _mm_unpackhi_epi8(x, x);}
  template <> inline mmUInt16
  mmCvtMaskH<mmUInt16>(mmUInt8 x)	{return _mm_unpackhi_epi8(x, x);}
  
// s_char -> short, u_short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmInt8 x)		{return _mm_unpacklo_epi8(x, x);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmInt8 x)		{return _mm_unpacklo_epi8(x, x);}
  template <> inline mmInt16
  mmCvtMaskH<mmInt16>(mmInt8 x)		{return _mm_unpackhi_epi8(x, x);}
  template <> inline mmUInt16
  mmCvtMaskH<mmUInt16>(mmInt8 x)	{return _mm_unpackhi_epi8(x, x);}
  
// u_short -> int, u_int
  template <> inline mmInt32
  mmCvtMask<mmInt32>(mmUInt16 x)	{return _mm_unpacklo_epi16(x, x);}
  template <> inline mmUInt32
  mmCvtMask<mmUInt32>(mmUInt16 x)	{return _mm_unpacklo_epi16(x, x);}
  template <> inline mmInt32
  mmCvtMaskH<mmInt32>(mmUInt16 x)	{return _mm_unpackhi_epi16(x, x);}
  template <> inline mmUInt32
  mmCvtMaskH<mmUInt32>(mmUInt16 x)	{return _mm_unpackhi_epi16(x, x);}

// short -> int, u_int
  template <> inline mmInt32
  mmCvtMask<mmInt32>(mmInt16 x)		{return _mm_unpacklo_epi16(x, x);}
  template <> inline mmUInt32
  mmCvtMask<mmUInt32>(mmInt16 x)	{return _mm_unpacklo_epi16(x, x);}
  template <> inline mmInt32
  mmCvtMaskH<mmInt32>(mmInt16 x)	{return _mm_unpackhi_epi16(x, x);}
  template <> inline mmUInt32
  mmCvtMaskH<mmUInt32>(mmInt16 x)	{return _mm_unpackhi_epi16(x, x);}

// int -> int64, u_int64
  template <> inline mmInt64
  mmCvtMask<mmInt64>(mmInt32 x)		{return _mm_unpacklo_epi32(x, x);}
  template <> inline mmUInt64
  mmCvtMask<mmUInt64>(mmInt32 x)	{return _mm_unpacklo_epi32(x, x);}
  template <> inline mmInt64
  mmCvtMaskH<mmInt64>(mmInt32 x)	{return _mm_unpackhi_epi32(x, x);}
  template <> inline mmUInt64
  mmCvtMaskH<mmUInt64>(mmInt32 x)	{return _mm_unpackhi_epi32(x, x);}

// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvtMask<mmInt64>(mmUInt32 x)	{return _mm_unpacklo_epi32(x, x);}
  template <> inline mmUInt64
  mmCvtMask<mmUInt64>(mmUInt32 x)	{return _mm_unpacklo_epi32(x, x);}
  template <> inline mmInt64
  mmCvtMaskH<mmInt64>(mmUInt32 x)	{return _mm_unpackhi_epi32(x, x);}
  template <> inline mmUInt64
  mmCvtMaskH<mmUInt64>(mmUInt32 x)	{return _mm_unpackhi_epi32(x, x);}

// short, u_short -> s_char
  template <> inline mmInt8
  mmCvtMask<mmInt8>(mmInt16 x, mmInt16 y)
					{return _mm_packs_epi16(x, y);}
  template <> inline mmInt8
  mmCvtMask<mmInt8>(mmUInt16 x, mmUInt16 y)
					{return _mm_packs_epi16(x, y);}
  
// short, u_short -> u_char
  template <> inline mmUInt8
  mmCvtMask<mmUInt8>(mmInt16 x, mmInt16 y)
					{return _mm_packs_epi16(x, y);}
  template <> inline mmUInt8
  mmCvtMask<mmUInt8>(mmUInt16 x, mmUInt16 y)
					{return _mm_packs_epi16(x, y);}
  
// int, u_int -> short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmInt32 x, mmInt32 y)
					{return _mm_packs_epi32(x, y);}
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmUInt32 x, mmUInt32 y)
					{return _mm_packs_epi32(x, y);}

// int, u_int -> u_short
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmInt32 x, mmInt32 y)
					{return _mm_packs_epi32(x, y);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmUInt32 x, mmUInt32 y)
					{return _mm_packs_epi32(x, y);}
#else
// s_char -> short, u_short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmInt8 x)		{return _mm_unpacklo_pi8(x, x);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmInt8 x)		{return _mm_unpacklo_pi8(x, x);}
  template <> inline mmInt16
  mmCvtMaskH<mmInt16>(mmInt8 x)		{return _mm_unpackhi_pi8(x, x);}
  template <> inline mmUInt16
  mmCvtMaskH<mmUInt16>(mmInt8 x)	{return _mm_unpackhi_pi8(x, x);}

// u_char -> short, u_short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmUInt8 x)		{return _mm_unpacklo_pi8(x, x);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmUInt8 x)	{return _mm_unpacklo_pi8(x, x);}
  template <> inline mmInt16
  mmCvtMaskH<mmInt16>(mmUInt8 x)	{return _mm_unpackhi_pi8(x, x);}
  template <> inline mmUInt16
  mmCvtMaskH<mmUInt16>(mmUInt8 x)	{return _mm_unpackhi_pi8(x, x);}

// short -> int, u_int
  template <> inline mmInt32
  mmCvtMask<mmInt32>(mmInt16 x)		{return _mm_unpacklo_pi16(x, x);}
  template <> inline mmUInt32
  mmCvtMask<mmUInt32>(mmInt16 x)	{return _mm_unpacklo_pi16(x, x);}
  template <> inline mmInt32
  mmCvtMaskH<mmInt32>(mmInt16 x)	{return _mm_unpackhi_pi16(x, x);}
  template <> inline mmUInt32
  mmCvtMaskH<mmUInt32>(mmInt16 x)	{return _mm_unpackhi_pi16(x, x);}

// u_short -> int, u_int
  template <> inline mmInt32
  mmCvtMask<mmInt32>(mmUInt16 x)	{return _mm_unpacklo_pi16(x, x);}
  template <> inline mmUInt32
  mmCvtMask<mmUInt32>(mmUInt16 x)	{return _mm_unpacklo_pi16(x, x);}
  template <> inline mmInt32
  mmCvtMaskH<mmInt32>(mmUInt16 x)	{return _mm_unpackhi_pi16(x, x);}
  template <> inline mmUInt32
  mmCvtMaskH<mmUInt32>(mmUInt16 x)	{return _mm_unpackhi_pi16(x, x);}

// int -> int64, u_int64
  template <> inline mmInt64
  mmCvtMask<mmInt64>(mmInt32 x)		{return _mm_unpacklo_pi32(x, x);}
  template <> inline mmUInt64
  mmCvtMask<mmUInt64>(mmInt32 x)	{return _mm_unpacklo_pi32(x, x);}
  template <> inline mmInt64
  mmCvtMaskH<mmInt64>(mmInt32 x)	{return _mm_unpackhi_pi32(x, x);}
  template <> inline mmUInt64
  mmCvtMaskH<mmUInt64>(mmInt32 x)	{return _mm_unpackhi_pi32(x, x);}

// u_int -> int64, u_int64
  template <> inline mmInt64
  mmCvtMask<mmInt64>(mmUInt32 x)	{return _mm_unpacklo_pi32(x, x);}
  template <> inline mmUInt64
  mmCvtMask<mmUInt64>(mmUInt32 x)	{return _mm_unpacklo_pi32(x, x);}
  template <> inline mmInt64
  mmCvtMaskH<mmInt64>(mmUInt32 x)	{return _mm_unpackhi_pi32(x, x);}
  template <> inline mmUInt64
  mmCvtMaskH<mmUInt64>(mmUInt32 x)	{return _mm_unpackhi_pi32(x, x);}

// short, u_short -> s_char
  template <> inline mmInt8
  mmCvtMask<mmInt8>(mmInt16 x, mmInt16 y)
					{return _mm_packs_pi16(x, y);}
  template <> inline mmInt8
  mmCvtMask<mmInt8>(mmUInt16 x, mmUInt16 y)
					{return _mm_packs_pi16(x, y);}
  
// short, u_short -> u_char
  template <> inline mmUInt8
  mmCvtMask<mmUInt8>(mmInt16 x, mmInt16 y)
					{return _mm_packs_pi16(x, y);}
  template <> inline mmUInt8
  mmCvtMask<mmUInt8>(mmUInt16 x, mmUInt16 y)
					{return _mm_packs_pi16(x, y);}
  
// int, u_int -> short
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmInt32 x, mmInt32 y)
					{return _mm_packs_pi32(x, y);}
  template <> inline mmInt16
  mmCvtMask<mmInt16>(mmUInt32 x, mmUInt32 y)
					{return _mm_packs_pi32(x, y);}
  
// int, u_int -> u_short
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmInt32 x, mmInt32 y)
					{return _mm_packs_pi32(x, y);}
  template <> inline mmUInt16
  mmCvtMask<mmUInt16>(mmUInt32 x, mmUInt32 y)
					{return _mm_packs_pi32(x, y);}
#endif
#if defined(SSE2)
// s_char, u_char, short, u_short, int, u_int -> float
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmInt32 x)		{return mmCastToFlt(x);}
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmUInt32 x)		{return mmCastToFlt(x);}
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmInt16 x)		{return mmCvtMask<mmFlt>(
						  mmCvtMask<mmInt32>(x));}
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmUInt16 x)		{return mmCvtMask<mmFlt>(
						  mmCvtMask<mmInt32>(x));}
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmInt8 x)		{return mmCvtMask<mmFlt>(
						  mmCvtMask<mmInt16>(x));}
  template <> inline mmFlt
  mmCvtMask<mmFlt>(mmUInt8 x)		{return mmCvtMask<mmFlt>(
						  mmCvtMask<mmInt16>(x));}

// float -> int
  template <> inline mmInt32
  mmCvtMask<mmInt32>(mmFlt x)		{return mmCast<mmInt32>(x);}

// int64, float -> double
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmInt64 x)		{return mmCastToDbl(x);}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmUInt64 x)		{return mmCastToDbl(x);}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmInt32 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt64>(x));}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmUInt32 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt64>(x));}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmInt16 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt32>(x));}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmUInt16 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt32>(x));}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmInt8 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt16>(x));}
  template <> inline mmDbl
  mmCvtMask<mmDbl>(mmUInt8 x)		{return mmCvtMask<mmDbl>(
						  mmCvtMask<mmInt16>(x));}

// double -> int64
  template <> inline mmInt64
  mmCvtMask<mmInt64>(mmDbl x)		{return mmCast<mmInt64>(x);}
#endif
    
/************************************************************************
*  論理演算								*
************************************************************************/
#if defined(SSE2)
  template <class T> static inline mmInt<T>
  operator &(mmInt<T> x, mmInt<T> y)	{return _mm_and_si128(x, y);}
  template <class T> static inline mmInt<T>
  operator |(mmInt<T> x, mmInt<T> y)	{return _mm_or_si128(x, y);}
  template <class T> static inline mmInt<T>
  mmAndNot(mmInt<T> x, mmInt<T> y)	{return _mm_andnot_si128(x, y);}
  template <class T> static inline mmInt<T>
  operator ^(mmInt<T> x, mmInt<T> y)	{return _mm_xor_si128(x, y);}
#else
  template <class T> static inline mmInt<T>
  operator &(mmInt<T> x, mmInt<T> y)	{return _mm_and_si64(x, y);}
  template <class T> static inline mmInt<T>
  operator |(mmInt<T> x, mmInt<T> y)	{return _mm_or_si64(x, y);}
  template <class T> static inline mmInt<T>
  mmAndNot(mmInt<T> x, mmInt<T> y)	{return _mm_andnot_si64(x, y);}
  template <class T> static inline mmInt<T>
  operator ^(mmInt<T> x, mmInt<T> y)	{return _mm_xor_si64(x, y);}
#endif
#if defined(SSE)
  static inline mmFlt
  operator &(mmFlt x, mmFlt y)		{return _mm_and_ps(x, y);}
  static inline mmFlt
  operator |(mmFlt x, mmFlt y)		{return _mm_or_ps(x, y);}
  static inline mmFlt
  mmAndNot(mmFlt x, mmFlt y)		{return _mm_andnot_ps(x, y);}
  static inline mmFlt
  operator ^(mmFlt x, mmFlt y)		{return _mm_xor_ps(x, y);}
#endif
#if defined(SSE2)
  static inline mmDbl
  operator &(mmDbl x, mmDbl y)		{return _mm_and_pd(x, y);}
  static inline mmDbl
  operator |(mmDbl x, mmDbl y)		{return _mm_or_pd(x, y);}
  static inline mmDbl
  mmAndNot(mmDbl x, mmDbl y)		{return _mm_andnot_pd(x, y);}
  static inline mmDbl
  operator ^(mmDbl x, mmDbl y)		{return _mm_xor_pd(x, y);}
#endif
    
/************************************************************************
*  選択									*
************************************************************************/
  template <class T> static inline T
  mmSelect(T x, T y, T mask)		{return (mask & x) | mmAndNot(mask, y);}
  template <class T> static inline T
  mmSelectNot(T x, T y, T mask)		{return mmAndNot(mask, x) | (mask & y);}

/************************************************************************
*  シフト演算								*
************************************************************************/
#if defined(SSE2)
  static inline mmInt16
  operator <<(mmInt16 x, u_int N)	{return _mm_slli_epi16(x, N);}
  static inline mmUInt16
  operator <<(mmUInt16 x, u_int N)	{return _mm_slli_epi16(x, N);}
  static inline mmInt32
  operator <<(mmInt32 x, u_int N)	{return _mm_slli_epi32(x, N);}
  static inline mmUInt32
  operator <<(mmUInt32 x, u_int N)	{return _mm_slli_epi32(x, N);}
  static inline mmInt64
  operator <<(mmInt64 x, u_int N)	{return _mm_slli_epi64(x, N);}
  static inline mmInt64
  operator <<(mmUInt64 x, u_int N)	{return _mm_slli_epi64(x, N);}
    
  static inline mmInt16
  operator >>(mmInt16 x, u_int N)	{return _mm_srai_epi16(x, N);}
  static inline mmUInt16
  operator >>(mmUInt16 x, u_int N)	{return _mm_srli_epi16(x, N);}
  static inline mmInt32
  operator >>(mmInt32 x, u_int N)	{return _mm_srai_epi32(x, N);}
  static inline mmUInt32
  operator >>(mmUInt32 x, u_int N)	{return _mm_srli_epi32(x, N);}
  static inline mmUInt64
  operator >>(mmInt64 x, u_int N)	{return _mm_srli_epi64(x, N);}
#else
  static inline mmInt16
  operator <<(mmInt16 x, u_int N)	{return _mm_slli_pi16(x, N);}
  static inline mmUInt16
  operator <<(mmUInt16 x, u_int N)	{return _mm_slli_pi16(x, N);}
  static inline mmInt32
  operator <<(mmInt32 x, u_int N)	{return _mm_slli_pi32(x, N);}
  static inline mmUInt32
  operator <<(mmUInt32 x, u_int N)	{return _mm_slli_pi32(x, N);}

  static inline mmInt16
  operator >>(mmInt16 x, u_int N)	{return _mm_srai_pi16(x, N);}
  static inline mmUInt16
  operator >>(mmUInt16 x, u_int N)	{return _mm_srli_pi16(x, N);}
  static inline mmInt32
  operator >>(mmInt32 x, u_int N)	{return _mm_srai_pi32(x, N);}
  static inline mmUInt32
  operator >>(mmUInt32 x, u_int N)	{return _mm_srli_pi32(x, N);}
#endif
  
/************************************************************************
*  四則演算								*
************************************************************************/
#if defined(SSE2)
  static inline mmInt8
  operator +(mmInt8 x, mmInt8 y)	{return _mm_adds_epi8(x, y);}
  static inline mmUInt8
  operator +(mmUInt8 x, mmUInt8 y)	{return _mm_adds_epu8(x, y);}
  static inline mmInt16
  operator +(mmInt16 x, mmInt16 y)	{return _mm_adds_epi16(x, y);}
  static inline mmUInt16
  operator +(mmUInt16 x, mmUInt16 y)	{return _mm_adds_epu16(x, y);}
  static inline mmInt32
  operator +(mmInt32 x, mmInt32 y)	{return _mm_add_epi32(x, y);}
  static inline mmUInt32
  operator +(mmUInt32 x, mmUInt32 y)	{return _mm_add_epi32(x, y);}
  static inline mmInt64
  operator +(mmInt64 x, mmInt64 y)	{return _mm_add_epi64(x, y);}

  static inline mmInt8
  operator -(mmInt8 x, mmInt8 y)	{return _mm_subs_epi8(x, y);}
  static inline mmUInt8
  operator -(mmUInt8 x, mmUInt8 y)	{return _mm_subs_epu8(x, y);}
  static inline mmInt16
  operator -(mmInt16 x, mmInt16 y)	{return _mm_subs_epi16(x, y);}
  static inline mmUInt16
  operator -(mmUInt16 x, mmUInt16 y)	{return _mm_subs_epu16(x, y);}
  static inline mmInt32
  operator -(mmInt32 x, mmInt32 y)	{return _mm_sub_epi32(x, y);}
  static inline mmInt64
  operator -(mmInt64 x, mmInt64 y)	{return _mm_sub_epi64(x, y);}

  static inline mmInt16
  operator *(mmInt16 x, mmInt16 y)	{return _mm_mullo_epi16(x, y);}
  static inline mmUInt16
  operator *(mmUInt16 x, mmUInt16 y)	{return _mm_mullo_epi16(x, y);}
  static inline mmInt16
  mmMulH(mmInt16 x, mmInt16 y)		{return _mm_mulhi_epi16(x, y);}
  static inline mmUInt16
  mmMulH(mmUInt16 x, mmUInt16 y)	{return _mm_mulhi_epu16(x, y);}
  static inline mmUInt32
  operator *(mmUInt32 x, mmUInt32 y)	{return _mm_mul_epu32(x, y);}
#else
  static inline mmInt8
  operator +(mmInt8 x, mmInt8 y)	{return _mm_adds_pi8(x, y);}
  static inline mmUInt8
  operator +(mmUInt8 x, mmUInt8 y)	{return _mm_adds_pu8(x, y);}
  static inline mmInt16
  operator +(mmInt16 x, mmInt16 y)	{return _mm_adds_pi16(x, y);}
  static inline mmUInt16
  operator +(mmUInt16 x, mmUInt16 y)	{return _mm_adds_pu16(x, y);}
  static inline mmInt32
  operator +(mmInt32 x, mmInt32 y)	{return _mm_add_pi32(x, y);}
  static inline mmUInt32
  operator +(mmUInt32 x, mmUInt32 y)	{return _mm_add_pi32(x, y);}

  static inline mmInt8
  operator -(mmInt8 x, mmInt8 y)	{return _mm_subs_pi8(x, y);}
  static inline mmUInt8
  operator -(mmUInt8 x, mmUInt8 y)	{return _mm_subs_pu8(x, y);}
  static inline mmInt16
  operator -(mmInt16 x, mmInt16 y)	{return _mm_subs_pi16(x, y);}
  static inline mmUInt16
  operator -(mmUInt16 x, mmUInt16 y)	{return _mm_subs_pu16(x, y);}
  static inline mmInt32
  operator -(mmInt32 x, mmInt32 y)	{return _mm_sub_pi32(x, y);}

  static inline mmInt16
  operator *(mmInt16 x, mmInt16 y)	{return _mm_mullo_pi16(x, y);}
  static inline mmUInt16
  operator *(mmUInt16 x, mmUInt16 y)	{return _mm_mullo_pi16(x, y);}
  static inline mmInt16
  mmMulH(mmInt16 x, mmInt16 y)		{return _mm_mulhi_pi16(x, y);}
#  if defined(SSE)
  static inline mmUInt16
  mmMulH(mmUInt16 x, mmUInt16 y)	{return _mm_mulhi_pu16(x, y);}
#  endif    
#endif
#if defined(SSE)
  static inline mmFlt
  operator +(mmFlt x, mmFlt y)		{return _mm_add_ps(x, y);}
  static inline mmFlt
  operator -(mmFlt x, mmFlt y)		{return _mm_sub_ps(x, y);}
  static inline mmFlt
  operator *(mmFlt x, mmFlt y)		{return _mm_mul_ps(x, y);}
  static inline mmFlt
  operator /(mmFlt x, mmFlt y)		{return _mm_div_ps(x, y);}
  static inline mmFlt
  mmSqrt(mmFlt x)			{return _mm_sqrt_ps(x);}
#endif
#if defined(SSE2)
  static inline mmDbl
  operator +(mmDbl x, mmDbl y)		{return _mm_add_pd(x, y);}
  static inline mmDbl
  operator -(mmDbl x, mmDbl y)		{return _mm_sub_pd(x, y);}
  static inline mmDbl
  operator *(mmDbl x, mmDbl y)		{return _mm_mul_pd(x, y);}
  static inline mmDbl
  operator /(mmDbl x, mmDbl y)		{return _mm_div_pd(x, y);}
  static inline mmDbl
  mmSqrt(mmDbl x)			{return _mm_sqrt_pd(x);}
#endif
    
/************************************************************************
*  単項マイナス演算							*
************************************************************************/
  template <class T> static inline mmInt<T>
  operator -(mmInt<T> x)		{return mmZero<mmInt<T> >() - x;}
#if defined(SSE)
  static inline mmFlt
  operator -(mmFlt x)			{return mmZero<mmFlt>() - x;}
#endif
#if defined(SSE2)
  static inline mmDbl
  operator -(mmDbl x)			{return mmZero<mmDbl>() - x;}
#endif
    
/************************************************************************
*  比較演算								*
************************************************************************/
#if defined(SSE2)
// 等しい
  static inline mmInt8
  operator ==(mmInt8 x, mmInt8 y)	{return _mm_cmpeq_epi8(x, y);}
  static inline mmUInt8
  operator ==(mmUInt8 x, mmUInt8 y)	{return _mm_cmpeq_epi8(x, y);}
  static inline mmInt16
  operator ==(mmInt16 x, mmInt16 y)	{return _mm_cmpeq_epi16(x, y);}
  static inline mmUInt16
  operator ==(mmUInt16 x, mmUInt16 y)	{return _mm_cmpeq_epi16(x, y);}
  static inline mmInt32
  operator ==(mmInt32 x, mmInt32 y)	{return _mm_cmpeq_epi32(x, y);}
  static inline mmUInt32
  operator ==(mmUInt32 x, mmUInt32 y)	{return _mm_cmpeq_epi32(x, y);}

// より大きい
  static inline mmInt8
  operator >(mmInt8 x, mmInt8 y)	{return _mm_cmpgt_epi8(x, y);}
  static inline mmInt16
  operator >(mmInt16 x, mmInt16 y)	{return _mm_cmpgt_epi16(x, y);}
  static inline mmInt32
  operator >(mmInt32 x, mmInt32 y)	{return _mm_cmpgt_epi32(x, y);}

// より小さい
  static inline mmInt8
  operator <(mmInt8 x, mmInt8 y)	{return _mm_cmplt_epi8(x, y);}
  static inline mmInt16
  operator <(mmInt16 x, mmInt16 y)	{return _mm_cmplt_epi16(x, y);}
  static inline mmInt32
  operator <(mmInt32 x, mmInt32 y)	{return _mm_cmplt_epi32(x, y);}
#else
// 等しい
  static inline mmInt8
  operator ==(mmInt8 x, mmInt8 y)	{return _mm_cmpeq_pi8(x, y);}
  static inline mmUInt8
  operator ==(mmUInt8 x, mmUInt8 y)	{return _mm_cmpeq_pi8(x, y);}
  static inline mmInt16
  operator ==(mmInt16 x, mmInt16 y)	{return _mm_cmpeq_pi16(x, y);}
  static inline mmUInt16
  operator ==(mmUInt16 x, mmUInt16 y)	{return _mm_cmpeq_pi16(x, y);}
  static inline mmInt32
  operator ==(mmInt32 x, mmInt32 y)	{return _mm_cmpeq_pi32(x, y);}
  static inline mmUInt32
  operator ==(mmUInt32 x, mmUInt32 y)	{return _mm_cmpeq_pi32(x, y);}

// より大きい
  static inline mmInt8
  operator >(mmInt8 x, mmInt8 y)	{return _mm_cmpgt_pi8(x, y);}
  static inline mmInt16
  operator >(mmInt16 x, mmInt16 y)	{return _mm_cmpgt_pi16(x, y);}
  static inline mmInt32
  operator >(mmInt32 x, mmInt32 y)	{return _mm_cmpgt_pi32(x, y);}

// より小さい
  static inline mmInt8
  operator <(mmInt8 x, mmInt8 y)	{return _mm_cmpgt_pi8(y, x);}
  static inline mmInt16
  operator <(mmInt16 x, mmInt16 y)	{return _mm_cmpgt_pi16(y, x);}
  static inline mmInt32
  operator <(mmInt32 x, mmInt32 y)	{return _mm_cmpgt_pi32(y, x);}
#endif
#if defined(SSE)
  static inline mmFlt
  operator ==(mmFlt x, mmFlt y)		{return _mm_cmpeq_ps(x, y);}
  static inline mmFlt
  operator !=(mmFlt x, mmFlt y)		{return _mm_cmpneq_ps(x, y);}
  static inline mmFlt
  operator >(mmFlt x, mmFlt y)		{return _mm_cmpgt_ps(x, y);}
  static inline mmFlt
  operator >=(mmFlt x, mmFlt y)		{return _mm_cmpge_ps(x, y);}
  static inline mmFlt
  operator <(mmFlt x, mmFlt y)		{return _mm_cmplt_ps(x, y);}
  static inline mmFlt
  operator <=(mmFlt x, mmFlt y)		{return _mm_cmple_ps(x, y);}
#endif
#if defined(SSE2)
  static inline mmDbl
  operator ==(mmDbl x, mmDbl y)		{return _mm_cmpeq_pd(x, y);}
  static inline mmDbl
  operator !=(mmDbl x, mmDbl y)		{return _mm_cmpneq_pd(x, y);}
  static inline mmDbl
  operator >(mmDbl x, mmDbl y)		{return _mm_cmpgt_pd(x, y);}
  static inline mmDbl
  operator >=(mmDbl x, mmDbl y)		{return _mm_cmpge_pd(x, y);}
  static inline mmDbl
  operator <(mmDbl x, mmDbl y)		{return _mm_cmplt_pd(x, y);}
  static inline mmDbl
  operator <=(mmDbl x, mmDbl y)		{return _mm_cmple_pd(x, y);}
#endif
  
/************************************************************************
*  Min/Max								*
************************************************************************/
  template <class T> static inline mmInt<T>
  mmMin(mmInt<T> x, mmInt<T> y)		{return mmSelect(x, y, x < y);}
  template <class T> static inline mmInt<T>
  mmMax(mmInt<T> x, mmInt<T> y)		{return mmSelect(x, y, x > y);}
#if defined(SSE)
#  if defined(SSE2)
#    if defined(SSE4)
  template <> inline mmInt8
  mmMin(mmInt8 x, mmInt8 y)		{return _mm_min_epi8(x, y);}
  template <> inline mmInt8
  mmMax(mmInt8 x, mmInt8 y)		{return _mm_max_epi8(x, y);}
  template <> inline mmUInt16
  mmMin(mmUInt16 x, mmUInt16 y)		{return _mm_min_epu16(x, y);}
  template <> inline mmUInt16
  mmMax(mmUInt16 x, mmUInt16 y)		{return _mm_max_epu16(x, y);}
  template <> inline mmInt32
  mmMin(mmInt32 x, mmInt32 y)		{return _mm_min_epi32(x, y);}
  template <> inline mmInt32
  mmMax(mmInt32 x, mmInt32 y)		{return _mm_max_epi32(x, y);}
  template <> inline mmUInt32
  mmMin(mmUInt32 x, mmUInt32 y)		{return _mm_min_epu32(x, y);}
  template <> inline mmUInt32
  mmMax(mmUInt32 x, mmUInt32 y)		{return _mm_max_epu32(x, y);}
#    endif
  template <> inline mmUInt8
  mmMin(mmUInt8 x, mmUInt8 y)		{return _mm_min_epu8(x, y);}
  template <> inline mmUInt8
  mmMax(mmUInt8 x, mmUInt8 y)		{return _mm_max_epu8(x, y);}
  template <> inline mmInt16
  mmMin(mmInt16 x, mmInt16 y)		{return _mm_min_epi16(x, y);}
  template <> inline mmInt16
  mmMax(mmInt16 x, mmInt16 y)		{return _mm_max_epi16(x, y);}
#  else	// !SSE2
  template <> inline mmUInt8
  mmMin(mmUInt8 x, mmUInt8 y)		{return _mm_min_pu8(x, y);}
  template <> inline mmUInt8
  mmMax(mmUInt8 x, mmUInt8 y)		{return _mm_max_pu8(x, y);}
  template <> inline mmInt16
  mmMin(mmInt16 x, mmInt16 y)		{return _mm_min_pi16(x, y);}
  template <> inline mmInt16
  mmMax(mmInt16 x, mmInt16 y)		{return _mm_max_pi16(x, y);}
#  endif
#endif    
#if defined(SSE)
  static inline mmFlt
  mmMin(mmFlt x, mmFlt y)		{return _mm_min_ps(x, y);}
  static inline mmFlt
  mmMax(mmFlt x, mmFlt y)		{return _mm_max_ps(x, y);}
#endif
#if defined(SSE2)
  static inline mmDbl
  mmMin(mmDbl x, mmDbl y)		{return _mm_min_pd(x, y);}
  static inline mmDbl
  mmMax(mmDbl x, mmDbl y)		{return _mm_max_pd(x, y);}
#endif
  
/************************************************************************
*  より小さいか等しい／より大きいか等しい				*
************************************************************************/
  template <class T> static inline mmInt<T>
  operator >=(mmInt<T> x, mmInt<T> y)	{return mmMax(x, y) == x;}
  template <class T> static inline mmInt<T>
  operator <=(mmInt<T> x, mmInt<T> y)	{return mmMin(x, y) == x;}
    
/************************************************************************
*  絶対値								*
************************************************************************/
  template <class T> static inline T
  mmAbs(T x)				{return mmMax(x, -x);}
#if defined(SSSE3)
  template <> inline mmInt8
  mmAbs(mmInt8 x)			{return _mm_abs_epi8(x);}
  template <> inline mmInt16
  mmAbs(mmInt16 x)			{return _mm_abs_epi16(x);}
  template <> inline mmInt32
  mmAbs(mmInt32 x)			{return _mm_abs_epi32(x);}
#endif
  
/************************************************************************
*  差の絶対値								*
************************************************************************/
  template <class T> static inline T
  mmDiff(T x, T y)			{return mmAbs(x - y);}
  template <> inline mmUInt8
  mmDiff(mmUInt8 x, mmUInt8 y)		{return (x - y) | (y - x);}
  template <> inline mmUInt16
  mmDiff(mmUInt16 x, mmUInt16 y)	{return (x - y) | (y - x);}
  
/************************************************************************
*  平均									*
************************************************************************/
  template <class T> static inline mmInt<T>
  mmAvg(mmInt<T> x, mmInt<T> y)		{return (x + y) >> 1;}
#if defined(SSE2)
  template <> inline mmUInt8
  mmAvg(mmUInt8 x, mmUInt8 y)		{return _mm_avg_epu8(x, y);}
  template <> inline mmUInt16
  mmAvg(mmUInt16 x, mmUInt16 y)		{return _mm_avg_epu16(x, y);}
#elif defined(SSE)
  template <> inline mmUInt8
  mmAvg(mmUInt8 x, mmUInt8 y)		{return _mm_avg_pu8(x, y);}
  template <> inline mmUInt16
  mmAvg(mmUInt16 x, mmUInt16 y)		{return _mm_avg_pu16(x, y);}
#endif
  
/************************************************************************
*  差の半分								*
************************************************************************/
  template <class T> static inline mmInt<T>
  mmSubAvg(mmInt<T> x, mmInt<T> y)	{return (x - y) >> 1;}
  
/************************************************************************
*  内積									*
************************************************************************/
#if defined(SSE3)
  static inline mmFlt
  mmInpro(mmFlt x, mmFlt y)		{mmFlt z = x * y;
					 z = _mm_hadd_ps(z, z);
					 return _mm_hadd_ps(z, z);}
  static inline mmDbl
  mmInpro(mmDbl x, mmDbl y)		{mmDbl z = x * y;
					 return _mm_hadd_pd(z, z);}
#endif
}
#endif	// MMX
#endif	// !__mmInstructions_h
