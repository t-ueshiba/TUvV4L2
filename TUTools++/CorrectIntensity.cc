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
 *  $Id: CorrectIntensity.cc,v 1.6 2009-02-02 08:09:24 ueshiba Exp $
 */
#include "TU/CorrectIntensity.h"
#include "TU/mmInstructions.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
#if defined(SSE)
template <class T> static inline void
mmCorrect(T* p, mmFlt a, mmFlt b)
{
    const mmUInt8	val = mmLoadU((const u_char*)p);
#  if defined(SSE2)
    mmStoreU((u_char*)p,
	     mmCvt<mmUInt8>(
		 mmCvt<mmInt16>(
		     mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(val)),
		     mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(
					mmShiftElmR<mmFlt::NElms>(val)))),
		 mmCvt<mmInt16>(
		     mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(
					mmShiftElmR<2*mmFlt::NElms>(val))),
		     mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(
					mmShiftElmR<3*mmFlt::NElms>(val))))));
#  else
    mmStoreU((u_char*)p,
	     mmCvt<mmUInt8>(
		 mmCvt<mmInt16>(a + b * mmCvt<mmFlt>(val)),
		 mmCvt<mmInt16>(a + b * mmCvt<mmFlt>(
				    mmShiftElmR<mmFlt::NElms>(val)))));
#  endif
}

template <> inline void
mmCorrect(short* p, mmFlt a, mmFlt b)
{
#  if defined(SSE2)
    const mmInt16	val = mmLoadU(p);
    mmStoreU(p, mmCvt<mmInt16>(
		 mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(val)),
		 mmCvt<mmInt32>(a + b * mmCvt<mmFlt>(
				    mmShiftElmR<mmFlt::NElms>(val)))));
#  else
    mmStoreU(p, mmCvt<mmInt16>(a + b * mmCvt<mmFlt>(mmLoadU(p))));
#  endif
}

template <> inline void
mmCorrect(float* p, mmFlt a, mmFlt b)
{
    mmStoreU(p, a + b * mmLoadU(p));
}
#endif

static inline u_char
toUChar(float val)
{
    return (val < 0.0 ? 0 : val > 255.0 ? 255 : u_char(val));
}
    
static inline short
toShort(float val)
{
    return (val < -32768.0 ? -32768 : val > 32767.0 ? 32767 : short(val));
}
    
/************************************************************************
*  class CorrectIntensity						*
************************************************************************/
//! 与えられた画像の輝度を補正する．
/*!
  \param image		入力画像を与え，補正結果もこれに返される
  \param vs		輝度を補正する領域の最初の行を指定するindex
  \param ve		輝度を補正する領域の最後の行の次を指定するindex
*/ 
template <class T> void
CorrectIntensity::operator()(Image<T>& image, int vs, int ve) const
{
    if (ve == 0)
	ve = image.height();
    
    for (int v = vs; v < ve; ++v)
    {
	T*		p = image[v];
	T* const	q = p + image.width();
#if defined(SSE)
	const mmFlt	a = mmSetAll<mmFlt>(_offset),
			b = mmSetAll<mmFlt>(_gain);
	for (T* const r = q - mmNBytes/sizeof(T);
	     p <= r; p += mmNBytes/sizeof(T))
	    mmCorrect(p, a, b);
	mmEmpty();
#endif
	for (; p < q; ++p)
	    *p = val(*p);
    }
}

//! 与えられた画素値に対する輝度補正結果を返す．
/*!
  \param pixel	画素値
  \return	輝度補正結果
*/
template <class T> inline T
CorrectIntensity::val(T pixel) const
{
    return T(toUChar(_offset + _gain * pixel.r),
	     toUChar(_offset + _gain * pixel.g),
	     toUChar(_offset + _gain * pixel.b));
}

template <> inline u_char
CorrectIntensity::val(u_char pixel) const
{
    return toUChar(_offset + _gain * pixel);
}
    
template <> inline short
CorrectIntensity::val(short pixel) const
{
    return toShort(_offset + _gain * pixel);
}
    
template <> inline float
CorrectIntensity::val(float pixel) const
{
    return _offset + _gain * pixel;
}

template void
CorrectIntensity::operator ()(Image<u_char>& image, int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<short>& image,  int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<float>& image,  int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<RGBA>& image,   int vs, int ve) const;
template void
CorrectIntensity::operator ()(Image<ABGR>& image,   int vs, int ve) const;
    
}
