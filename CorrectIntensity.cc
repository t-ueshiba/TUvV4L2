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
 *  $Id$
 */
#include "TU/CorrectIntensity.h"
#include "TU/mmInstructions.h"

/************************************************************************
*  static functions							*
************************************************************************/
namespace TU
{
#if defined(SSE)
namespace mm
{
template <class T> static inline void
correct(T* p, F32vec a, F32vec b)
{
    const size_t	nelms = F32vec::size;
    const Iu8vec	val = load<false>((const u_char*)p);
#  if defined(SSE2)
    store<false>((u_char*)p,
	   cvt<u_char>(
	       cvt<short>(
		   cvt<int>(a + b * cvt<float>(val)),
		   cvt<int>(a + b * cvt<float>(shift_r<nelms>(val)))),
	       cvt<short>(
		   cvt<int>(a + b * cvt<float>(shift_r<2*nelms>(val))),
		   cvt<int>(a + b * cvt<float>(shift_r<3*nelms>(val))))));
#  else
    store<false>((u_char*)p,
		 cvt<u_char>(cvt<short>(a + b * cvt<float>(val)),
			     cvt<short>(a + b * cvt<float>(
					    shift_r<nelms>(val)))));
#  endif
}

template <> inline void
correct(short* p, F32vec a, F32vec b)
{
#  if defined(SSE2)
    const size_t	nelms = F32vec::size;
    const Is16vec	val = load<false>(p);
    store<false>(p, cvt<short>(cvt<int>(a + b * cvt<float>(val)),
			       cvt<int>(a + b * cvt<float>(
					    shift_r<nelms>(val)))));
#  else
    store<false>(p, cvt<short>(a + b * cvt<float>(load<false>(p))));
#  endif
}

template <> inline void
correct(float* p, F32vec a, F32vec b)
{
    store<false>(p, a + b * load<false>(p));
}

}	// namespace mm
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
CorrectIntensity::operator()(Image<T>& image, size_t vs, size_t ve) const
{
    if (ve == 0)
	ve = image.height();
    
    for (size_t v = vs; v < ve; ++v)
    {
	T*		p = image[v].data();
	T* const	q = p + image.width();
#if defined(SSE)
	using namespace	mm;

	const size_t	nelms = vec<T>::size;
	const F32vec	a(_offset), b(_gain);
	for (T* const r = q - nelms; p <= r; p += nelms)
	    correct(p, a, b);
	empty();
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

template __PORT void
CorrectIntensity::operator ()(Image<u_char>& image, size_t vs, size_t ve) const;
template __PORT void
CorrectIntensity::operator ()(Image<short>& image,  size_t vs, size_t ve) const;
template __PORT void
CorrectIntensity::operator ()(Image<float>& image,  size_t vs, size_t ve) const;
    
}
