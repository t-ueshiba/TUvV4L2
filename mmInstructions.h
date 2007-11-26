/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，使用，第三者へ開示する
 *  等の著作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: mmInstructions.h,v 1.3 2007-11-26 07:55:48 ueshiba Exp $
 */
#if !defined(__mmInstructions_h) && defined(__INTEL_COMPILER)
#define __mmInstructions_h

#if defined(SSE3)
#  define SSE2
#endif
#if defined(SSE2)
#  define SSE
#endif
#if defined(SSE)
#  define MMX
#endif

#if defined(SSE3)
#  include <pmmintrin.h>
#elif defined(SSE2)
#  include <emmintrin.h>
#elif defined(SSE)
#  include <xmmintrin.h>
#elif defined(MMX)
#  include <mmintrin.h>
#endif

namespace TU
{
/************************************************************************
*  整数演算								*
************************************************************************/
#if defined(SSE2)
  typedef __m128i	mmInt;

  static inline mmInt	mmLoad(const mmInt* x)	   {return _mm_load_si128(x);}
#  if defined(SSE3)
  static inline mmInt	mmLoadU(const mmInt* x)	   {return _mm_lddqu_si128(x);}
#  else
  static inline mmInt	mmLoadU(const mmInt* x)	   {return _mm_loadu_si128(x);}
#  endif
  static inline void	mmStore(mmInt* x, mmInt y) {_mm_store_si128(x, y);}
  static inline void	mmStoreU(mmInt* x, mmInt y){_mm_storeu_si128(x, y);}
  static inline mmInt	mmZero()		   {return
							_mm_setzero_si128();}
  static inline mmInt	mmSet8(u_char x)	   {return _mm_set1_epi8(x);}
  static inline mmInt	mmSet(short x)		   {return _mm_set1_epi16(x);}
  static inline mmInt	mmSet32(int x)		   {return _mm_set1_epi32(x);}
  static inline mmInt	mmSet32(int z, int y,
				int x, int w)	   {return _mm_set_epi32(
							z, y, x, w);}
  static inline mmInt	mmSetRMost(short x)	   {return _mm_set_epi16(
							0,0,0,0,0,0,0,x);}
  static inline mmInt	mmPack(mmInt x, mmInt y)   {return _mm_packs_epi16(
							x, y);}
  static inline mmInt	mmPackUS(mmInt x, mmInt y) {return _mm_packus_epi16(
							x, y);}
  static inline mmInt	mmPack32(mmInt x, mmInt y) {return _mm_packs_epi32(
							x, y);}
  static inline mmInt	mmUnpackL8(mmInt x,
				   mmInt y)	   {return _mm_unpacklo_epi8(
							x, y);}
  static inline mmInt	mmUnpackH8(mmInt x,
				   mmInt y)	   {return _mm_unpackhi_epi8(
							x, y);}
  static inline mmInt	mmUnpackL(mmInt x, mmInt y){return _mm_unpacklo_epi16(
							x, y);}
  static inline mmInt	mmUnpackH(mmInt x, mmInt y){return _mm_unpackhi_epi16(
							x, y);}
  static inline mmInt	mmUnpackL32(mmInt x,
				    mmInt y)	   {return _mm_unpacklo_epi32(
							x, y);}
  static inline mmInt	mmUnpackH32(mmInt x,
				    mmInt y)	   {return _mm_unpackhi_epi32(
							x, y);}
  static inline mmInt	mmUnpackL64(mmInt x,
				    mmInt y)	   {return _mm_unpacklo_epi64(
							x, y);}
  static inline mmInt	mmUnpackH64(mmInt x,
				    mmInt y)	   {return _mm_unpackhi_epi64(
							x, y);}
  static inline mmInt	mmAnd(mmInt x, mmInt y)    {return _mm_and_si128(
							x, y);}
  static inline mmInt	mmAndNot(mmInt x, mmInt y) {return _mm_andnot_si128(
							x, y);}
  static inline mmInt	mmOr(mmInt x, mmInt y)	   {return _mm_or_si128(x, y);}
  static inline mmInt	mmXor(mmInt x, mmInt y)	   {return _mm_xor_si128(x, y);}
  static inline mmInt	mmAdd(mmInt x, mmInt y)	   {return _mm_adds_epi16(
							x, y);}
  static inline mmInt	mmSub(mmInt x, mmInt y)	   {return _mm_subs_epi16(
							x, y);}
  static inline mmInt	mmAvg(mmInt x, mmInt y)	   {return _mm_srai_epi16(
							mmAdd(x, y), 1);}
  static inline mmInt	mmSubAvg(mmInt x, mmInt y) {return _mm_srai_epi16(
							mmSub(x, y), 1);}
  static inline mmInt	mmMax(mmInt x, mmInt y)	   {return _mm_max_epi16(
							x, y);}
  static inline mmInt	mmMin(mmInt x, mmInt y)	   {return _mm_min_epi16(
							x, y);}
  static inline mmInt	mmCmpLE(mmInt x, mmInt y)  {return _mm_cmpgt_epi16(
							y, x);}
  static inline mmInt	mmAdd8(mmInt x, mmInt y)   {return _mm_adds_epu8(
							x, y);}
  static inline mmInt	mmSub8(mmInt x, mmInt y)   {return _mm_subs_epu8(
							x, y);}
  static inline mmInt	mmAvg8(mmInt x, mmInt y)   {return _mm_avg_epu8(x, y);}
  static inline mmInt	mmMax8(mmInt x, mmInt y)   {return _mm_max_epu8(x, y);}
  static inline mmInt	mmMin8(mmInt x, mmInt y)   {return _mm_min_epu8(x, y);}
#elif defined(MMX)
  typedef __m64		mmInt;

  static inline mmInt	mmLoad(const mmInt* x)	   {return *x;}
  static inline mmInt	mmLoadU(const mmInt* x)	   {return *x;}
  static inline void	mmStore(mmInt* x, mmInt y) {*x = y;}
  static inline void	mmStoreU(mmInt* x, mmInt y){*x = y;}
  static inline mmInt	mmZero()		   {return _mm_setzero_si64();}
  static inline mmInt	mmSet8(u_char x)	   {return _mm_set1_pi8(x);}
  static inline mmInt	mmSet(short x)		   {return _mm_set1_pi16(x);}
  static inline mmInt	mmSet32(int x)		   {return _mm_set1_pi32(x);}
  static inline mmInt	mmSet32(int y, int x)	   {return _mm_set_pi32(y, x);}
  static inline mmInt	mmSetRMost(short x)	   {return _mm_set_pi16(
							0, 0, 0, x);}
  static inline mmInt	mmPack(mmInt x, mmInt y)   {return _m_packsswb(x, y);}
  static inline mmInt	mmPackUS(mmInt x, mmInt y) {return _m_packuswb(x, y);}
  static inline mmInt	mmPack32(mmInt x, mmInt y) {return _m_packssdw(x, y);}
  static inline mmInt	mmUnpackL8(mmInt x,
				   mmInt y)	   {return _m_punpcklbw(x, y);}
  static inline mmInt	mmUnpackH8(mmInt x,
				   mmInt y)	   {return _m_punpckhbw(x, y);}
  static inline mmInt	mmUnpackL(mmInt x, mmInt y){return _m_punpcklwd(x, y);}
  static inline mmInt	mmUnpackH(mmInt x, mmInt y){return _m_punpckhwd(x, y);}
  static inline mmInt	mmUnpackL32(mmInt x,
				    mmInt y)	   {return _m_punpckldq(x, y);}
  static inline mmInt	mmUnpackH32(mmInt x,
				    mmInt y)	   {return _m_punpckhdq(x, y);}
  static inline mmInt	mmAnd(mmInt x, mmInt y)    {return _m_pand(x, y);}
  static inline mmInt	mmAndNot(mmInt x, mmInt y) {return _m_pandn(x, y);}
  static inline mmInt	mmOr(mmInt x, mmInt y)     {return _m_por(x, y);}
  static inline mmInt	mmXor(mmInt x, mmInt y)    {return _m_pxor(x, y);}
  static inline mmInt	mmAdd(mmInt x, mmInt y)	   {return _m_paddsw(x, y);}
  static inline mmInt	mmSub(mmInt x, mmInt y)	   {return _m_psubsw(x, y);}
  static inline mmInt	mmAvg(mmInt x, mmInt y)	   {return _m_psrawi(
							mmAdd(x, y), 1);}
  static inline mmInt	mmSubAvg(mmInt x, mmInt y) {return _m_psrawi(
							mmSub(x, y), 1);}
  static inline mmInt	mmCmpLE(mmInt x, mmInt y)  {return _m_pcmpgtw(y, x);}
  static inline mmInt	mmAdd8(mmInt x, mmInt y)   {return _m_paddusb(x, y);}
  static inline mmInt	mmSub8(mmInt x, mmInt y)   {return _m_psubusb(x, y);}
#  if defined(SSE)
  static inline mmInt	mmMax(mmInt x, mmInt y)	   {return _m_pmaxsw(x, y);}
  static inline mmInt	mmMin(mmInt x, mmInt y)	   {return _m_pminsw(x, y);}
  static inline mmInt	mmAvg8(mmInt x, mmInt y)   {return _m_pavgb(x, y);}
  static inline mmInt	mmMax8(mmInt x, mmInt y)   {return _m_pmaxub(x, y);}
  static inline mmInt	mmMin8(mmInt x, mmInt y)   {return _m_pminub(x, y);}
#  else
  static inline mmInt	mmMax(mmInt x, mmInt y)
			{
			    mmInt	mask = mmCmpLE(y, x);
			    return mmOr(mmAnd(mask, x), mmAndNot(mask, y));
			}
  static inline mmInt	mmMin(mmInt x, mmInt y)
			{
			    mmInt	mask = mmCmpLE(x, y);
			    return mmOr(mmAnd(mask, x), mmAndNot(mask, y));
			}
  static inline mmInt	mmAvg8(mmInt x, mmInt y)
			{
			    return mmPackUS(mmAvg(mmUnpackL8(x, mmZero()),
						  mmUnpackL8(y, mmZero())),
					    mmAvg(mmUnpackH8(x, mmZero()),
						  mmUnpackH8(y, mmZero())));
			}
  static inline mmInt	mmMax8(mmInt x, mmInt y)
			{
			    mmInt	mask = _m_pcmpgtb(x, y);
			    return mmOr(mmAnd(mask, x), mmAndNot(mask, y));
			}
  static inline mmInt	mmMin8(mmInt x, mmInt y)
			{
			    mmInt	mask = _m_pcmpgtb(y, x);
			    return mmOr(mmAnd(mask, x), mmAndNot(mask, y));
			}
#  endif
#endif

/************************************************************************
*  単精度浮動小数点演算							*
************************************************************************/
#if defined(SSE)
  typedef __m128	mmFlt;

#  if defined(SSE2)
  static inline mmFlt	mmToFltL(mmInt x)	    {return _mm_cvtpi16_ps(
						     _mm_movepi64_pi64(x));}
  static inline mmFlt	mmToFltH(mmInt x)	    {return _mm_cvtpi16_ps(
						     _mm_movepi64_pi64(
						      _mm_srli_si128(x, 8)));}
  static inline mmFlt	mmToFlt0(mmInt x)	    {return _mm_cvtpu8_ps(
						     _mm_movepi64_pi64(x));}
  static inline mmFlt	mmToFlt1(mmInt x)	    {return _mm_cvtpu8_ps(
						     _mm_movepi64_pi64(
						      _mm_srli_si128(x, 4)));}
  static inline mmFlt	mmToFlt2(mmInt x)	    {return _mm_cvtpu8_ps(
						     _mm_movepi64_pi64(
						      _mm_srli_si128(x, 8)));}
  static inline mmFlt	mmToFlt3(mmInt x)	    {return _mm_cvtpu8_ps(
						     _mm_movepi64_pi64(
						      _mm_srli_si128(x, 12)));}
  static inline mmFlt	mmToFlt32(mmInt x)	    {return
							_mm_cvtepi32_ps(x);}
  static inline mmFlt	mmCastToFlt(mmInt x)	    {return
							_mm_castsi128_ps(x);}
  static inline mmInt	mmCastToInt(mmFlt x)	    {return
							_mm_castps_si128(x);}
#  else
  static inline mmFlt	mmToFlt(mmInt x)	    {return _mm_cvtpi16_ps(x);}
  static inline mmFlt	mmToFlt0(mmInt x)	    {return _mm_cvtpu8_ps(x);}
  static inline mmFlt	mmToFlt1(mmInt x)	    {return _mm_cvtpu8_ps(
						      _m_psrlqi(x, 32));}
  static inline mmInt	mmToIntF(mmFlt x)	    {return _mm_cvtps_pi16(x);}
  static inline mmInt	mmToInt8F(mmFlt x)	    {return _mm_cvtps_pi8(x);}
#  endif
  static inline mmFlt	mmLoadF(const float* x)	    {return _mm_load_ps(x);}
  static inline mmFlt	mmLoadFU(const float* x)    {return _mm_loadu_ps(x);}
  static inline void	mmStoreF(float* x, mmFlt y) {_mm_store_ps(x, y);}
  static inline void	mmStoreFU(float* x, mmFlt y){_mm_storeu_ps(x, y);}
  static inline void	mmStoreRMostF(float* x,
				      mmFlt y)	    {return _mm_store_ss(x,y);}
  static inline mmFlt	mmSetF(float x)		    {return _mm_set_ps1(x);}
  static inline mmFlt	mmSetF(float z, float y,
			       float x, float w)    {return _mm_set_ps(z, y,
								       x, w);}
  static inline mmFlt	mmZeroF()		    {return _mm_setzero_ps();}
  static inline mmFlt	mmUnpackLF(mmFlt x, mmFlt y){return
							_mm_unpacklo_ps(x, y);}
  static inline mmFlt	mmUnpackHF(mmFlt x, mmFlt y){return
							_mm_unpackhi_ps(x, y);}
  static inline mmFlt	mmUnpackLF2(mmFlt x, mmFlt y)
			{
			    return _mm_shuffle_ps(x, y, _MM_SHUFFLE(1,0,1,0));
			}
  static inline mmFlt	mmUnpackHF2(mmFlt x, mmFlt y)
			{
			    return _mm_shuffle_ps(x, y, _MM_SHUFFLE(3,2,3,2));
			}
  static inline mmFlt	mmRotateLF(mmFlt x)
			{
			    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(2,1,0,3));
			}
  static inline mmFlt	mmRotateRF(mmFlt x)
			{
			    return _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,3,2,1));
			}
  static inline mmFlt	mmReplaceRMostF(mmFlt x,
					mmFlt y)    {return _mm_move_ss(x, y);}
  static inline mmFlt	mmAndF(mmFlt x, mmFlt y)    {return _mm_and_ps(x, y);}
  static inline mmFlt	mmAndNotF(mmFlt x, mmFlt y) {return
							_mm_andnot_ps(x, y);}
  static inline mmFlt	mmOrF(mmFlt x, mmFlt y)	    {return _mm_or_ps(x, y);}
  static inline mmFlt	mmXorF(mmFlt x, mmFlt y)    {return _mm_xor_ps(x, y);}
  static inline mmFlt	mmAddF(mmFlt x, mmFlt y)    {return _mm_add_ps(x, y);}
  static inline mmFlt	mmSubF(mmFlt x, mmFlt y)    {return _mm_sub_ps(x, y);}
  static inline mmFlt	mmMulF(mmFlt x, mmFlt y)    {return _mm_mul_ps(x, y);}
  static inline mmFlt	mmDivF(mmFlt x, mmFlt y)    {return _mm_div_ps(x, y);}
  static inline mmFlt	mmSqrtF(mmFlt x)	    {return _mm_sqrt_ps(x);}
  static inline mmFlt	mmCmpLEF(mmFlt x, mmFlt y)  {return
							_mm_cmple_ps(x, y);}
  static inline mmFlt	mmCmpLTF(mmFlt x, mmFlt y)  {return
							_mm_cmplt_ps(x, y);}
  static inline mmFlt	mmCmpEQF(mmFlt x, mmFlt y)  {return
							_mm_cmpeq_ps(x, y);}
  static inline mmFlt	mmCmpNEF(mmFlt x, mmFlt y)  {return
							_mm_cmpneq_ps(x, y);}
  static inline mmFlt	mmDiffF(mmFlt x, mmFlt y)
			{
			    return _mm_max_ps(mmSubF(x, y), mmSubF(y, x));
			}
#  if defined(SSE3)
  static inline mmFlt	mmInpro4F(mmFlt x, mmFlt y)
			{
			    mmFlt z = mmMulF(x, y);
			    z = _mm_hadd_ps(z, z);
			    return _mm_hadd_ps(z, z);
			}
#  endif
#endif

/************************************************************************
*  汎用ユティリティ演算							*
************************************************************************/
#if defined(MMX)
  static const int	mmNBytes  = sizeof(mmInt),
			mmNWords  = mmNBytes / sizeof(short),
			mmNDWords = mmNBytes / sizeof(float);

  static inline mmInt	mmPack(mmInt x)	     {return mmPack(x, mmZero());}
  static inline mmInt	mmPackUS(mmInt x)    {return mmPackUS(x, mmZero());}
  static inline mmInt	mmPack32(mmInt x)    {return mmPack32(x, mmZero());}
  static inline mmInt	mmUnpackL8(mmInt x)  {return mmUnpackL8(x, mmZero());}
  static inline mmInt	mmUnpackH8(mmInt x)  {return mmUnpackH8(x, mmZero());}
  static inline mmInt	mmUnpackL(mmInt x)   {return mmUnpackL(x, mmZero());}
  static inline mmInt	mmUnpackH(mmInt x)   {return mmUnpackH(x, mmZero());}
  static inline mmInt	mmUnpackL32(mmInt x) {return mmUnpackL32(x, mmZero());}
  static inline mmInt	mmUnpackH32(mmInt x) {return mmUnpackH32(x, mmZero());}
  static inline mmInt	mmSelect(mmInt x, mmInt y, mmInt mask)
			{
			    return mmOr(mmAnd(mask, x), mmAndNot(mask, y));
			}
  static inline mmInt	mmDiff8(mmInt x, mmInt y)
			{
			    return mmOr(mmSub8(x, y), mmSub8(y, x));
			}
  static inline mmInt	mmCmpLE8(mmInt x, mmInt y)
			{
			    return mmPack(mmCmpLE(mmUnpackL8(x),
						  mmUnpackL8(y)),
					  mmCmpLE(mmUnpackH8(x),
						  mmUnpackH8(y)));
			}
#endif
}

#endif
