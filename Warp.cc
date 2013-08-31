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
#if defined(__INTEL_COMPILER)
#  undef SSE4
#  undef SSSE3
#  undef SSE3
#  undef SSE2
#  define SSE
//#  define SSE2
//#  define SSE3
#endif
#include "TU/Warp.h"

#if defined(SSE)
namespace mm
{
  static inline Is16vec
  linearInterpolate(Is16vec x, Is16vec y, Is16vec d)
  {
      return x + ((d * (y - x)) >> 7);
  }

  template <class T> static inline Is16vec
  bilinearInterpolate(const TU::Image<T>& in,
		      Is16vec us, Is16vec vs, Is16vec du, Is16vec dv)
  {
      const Is16vec	ue = us + Is16vec(1);
#  if defined(SSE2)
      Iu8vec		uc = cast<u_char>(Iu32vec(
				*(u_int*)&in[extract<1>(vs)][extract<1>(ue)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(ue)],
				*(u_int*)&in[extract<1>(vs)][extract<1>(us)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(us)]));
#  else
      Iu8vec		uc = cast<u_char>(Iu32vec(
				*(u_int*)&in[extract<0>(vs)][extract<0>(ue)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(us)]));
#  endif
      const Is16vec	ss = linearInterpolate(cvt<short, 0>(uc),
					       cvt<short, 1>(uc), du);
      vs += Is16vec(1);
#  if defined(SSE2)
      uc = cast<u_char>(Iu32vec(*(u_int*)&in[extract<1>(vs)][extract<1>(ue)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(ue)],
				*(u_int*)&in[extract<1>(vs)][extract<1>(us)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(us)]));
#  else
      uc = cast<u_char>(Iu32vec(*(u_int*)&in[extract<0>(vs)][extract<0>(ue)],
				*(u_int*)&in[extract<0>(vs)][extract<0>(us)]));
#  endif
      return linearInterpolate(ss,
			       linearInterpolate(cvt<short, 0>(uc),
						 cvt<short, 1>(uc), du),
			       dv);
  }
    
  template <> inline Is16vec
  bilinearInterpolate(const TU::Image<u_char>& in,
		      Is16vec us, Is16vec vs, Is16vec du, Is16vec dv)
  {
      const Is16vec	ue = us + Is16vec(1);
#  if defined(SSE2)
      Iu8vec		uc(in[extract<7>(vs)][extract<7>(ue)],
			   in[extract<6>(vs)][extract<6>(ue)],
			   in[extract<5>(vs)][extract<5>(ue)],
			   in[extract<4>(vs)][extract<4>(ue)],
			   in[extract<3>(vs)][extract<3>(ue)],
			   in[extract<2>(vs)][extract<2>(ue)],
			   in[extract<1>(vs)][extract<1>(ue)],
			   in[extract<0>(vs)][extract<0>(ue)],
			   in[extract<7>(vs)][extract<7>(us)],
			   in[extract<6>(vs)][extract<6>(us)],
			   in[extract<5>(vs)][extract<5>(us)],
			   in[extract<4>(vs)][extract<4>(us)],
			   in[extract<3>(vs)][extract<3>(us)],
			   in[extract<2>(vs)][extract<2>(us)],
			   in[extract<1>(vs)][extract<1>(us)],
			   in[extract<0>(vs)][extract<0>(us)]);
#  else
      Iu8vec		uc(in[extract<3>(vs)][extract<3>(ue)],
			   in[extract<2>(vs)][extract<2>(ue)],
			   in[extract<1>(vs)][extract<1>(ue)],
			   in[extract<0>(vs)][extract<0>(ue)],
			   in[extract<3>(vs)][extract<3>(us)],
			   in[extract<2>(vs)][extract<2>(us)],
			   in[extract<1>(vs)][extract<1>(us)],
			   in[extract<0>(vs)][extract<0>(us)]);
#  endif
      const Is16vec	ss = linearInterpolate(cvt<short, 0>(uc),
					       cvt<short, 1>(uc), du);
      vs += Is16vec(1);
#  if defined(SSE2)
      uc = Iu8vec(in[extract<7>(vs)][extract<7>(ue)],
		  in[extract<6>(vs)][extract<6>(ue)],
		  in[extract<5>(vs)][extract<5>(ue)],
		  in[extract<4>(vs)][extract<4>(ue)],
		  in[extract<3>(vs)][extract<3>(ue)],
		  in[extract<2>(vs)][extract<2>(ue)],
		  in[extract<1>(vs)][extract<1>(ue)],
		  in[extract<0>(vs)][extract<0>(ue)],
		  in[extract<7>(vs)][extract<7>(us)],
		  in[extract<6>(vs)][extract<6>(us)],
		  in[extract<5>(vs)][extract<5>(us)],
		  in[extract<4>(vs)][extract<4>(us)],
		  in[extract<3>(vs)][extract<3>(us)],
		  in[extract<2>(vs)][extract<2>(us)],
		  in[extract<1>(vs)][extract<1>(us)],
		  in[extract<0>(vs)][extract<0>(us)]);
#  else
      uc = Iu8vec(in[extract<3>(vs)][extract<3>(ue)],
		  in[extract<2>(vs)][extract<2>(ue)],
		  in[extract<1>(vs)][extract<1>(ue)],
		  in[extract<0>(vs)][extract<0>(ue)],
		  in[extract<3>(vs)][extract<3>(us)],
		  in[extract<2>(vs)][extract<2>(us)],
		  in[extract<1>(vs)][extract<1>(us)],
		  in[extract<0>(vs)][extract<0>(us)]);
#  endif
      return linearInterpolate(ss,
			       linearInterpolate(cvt<short, 0>(uc),
						 cvt<short, 1>(uc), du),
			       dv);
  }
}
#endif
    
namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
template <class T> static inline T
bilinearInterpolate(const Image<T>& in, int us, int vs, int du, int dv)
{
    T		in00 = in[vs][us],   in01 = in[vs][us+1],
		in10 = in[vs+1][us], in11 = in[vs+1][us+1];
    int		tmp0, tmp1;
    T		out;
    tmp0 = int(in00.r) + ((du * (int(in01.r) - int(in00.r))) >> 7);
    tmp1 = int(in10.r) + ((du * (int(in11.r) - int(in10.r))) >> 7);
    out.r = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);
    tmp0 = int(in00.g) + ((du * (int(in01.g) - int(in00.g))) >> 7);
    tmp1 = int(in10.g) + ((du * (int(in11.g) - int(in10.g))) >> 7);
    out.g = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);
    tmp0 = int(in00.b) + ((du * (int(in01.b) - int(in00.b))) >> 7);
    tmp1 = int(in10.b) + ((du * (int(in11.b) - int(in10.b))) >> 7);
    out.b = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);
    tmp0 = int(in00.a) + ((du * (int(in01.a) - int(in00.a))) >> 7);
    tmp1 = int(in10.a) + ((du * (int(in11.a) - int(in10.a))) >> 7);
    out.a = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);

    return out;
}

template <> inline YUV444
bilinearInterpolate(const Image<YUV444>& in, int us, int vs, int du, int dv)
{
    YUV444	in00 = in[vs][us],   in01 = in[vs][us+1],
		in10 = in[vs+1][us], in11 = in[vs+1][us+1];
    int		tmp0, tmp1;
    YUV444	out;
    tmp0 = int(in00.y) + ((du * (int(in01.y) - int(in00.y))) >> 7);
    tmp1 = int(in10.y) + ((du * (int(in11.y) - int(in10.y))) >> 7);
    out.y = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);
    tmp0 = int(in00.u) + ((du * (int(in01.u) - int(in00.u))) >> 7);
    tmp1 = int(in10.u) + ((du * (int(in11.u) - int(in10.u))) >> 7);
    out.u = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);
    tmp0 = int(in00.v) + ((du * (int(in01.v) - int(in00.v))) >> 7);
    tmp1 = int(in10.v) + ((du * (int(in11.v) - int(in10.v))) >> 7);
    out.v = tmp0 + ((dv * (tmp1 - tmp0)) >> 7);

    return out;
}

template <> inline u_char
bilinearInterpolate(const Image<u_char>& in, int us, int vs, int du, int dv)
{
    int		in00 = in[vs][us],   in01 = in[vs][us+1],
		in10 = in[vs+1][us], in11 = in[vs+1][us+1];
    int		out0 = in00 + ((du * (in01 - in00)) >> 7),
		out1 = in10 + ((du * (in11 - in10)) >> 7);
    
    return out0 + ((dv * (out1 - out0)) >> 7);
}

/************************************************************************
*  class Warp								*
************************************************************************/
template <class T> void
Warp::warpLine(const Image<T>& in, Image<T>& out, u_int v) const
{
    const short		*usp  = _fracs[v].us.data(), *vsp = _fracs[v].vs.data();
    const u_char	*dup  = _fracs[v].du.data(), *dvp = _fracs[v].dv.data();
    T			*outp = out[v].data() + _fracs[v].lmost;
    T* const		outq  = outp + _fracs[v].width();
#if defined(SSE)
    using namespace	mm;

    for (T* const outr = outq - Iu8vec::size; outp <= outr; )
    {
	using namespace	std;

	const u_int	npixels = Is16vec::size/4;
	Is16vec		uu = load(usp), vv = load(vsp);
	Iu8vec		du = load(dup), dv = load(dvp);
	Iu8vec		du4 = quadup<0>(du), dv4 = quadup<0>(dv);
	storeu((u_char*)outp,
	       cvt<u_char>(bilinearInterpolate(in, uu, vv,
					       cvt<short, 0>(du4),
					       cvt<short, 0>(dv4)),
			   bilinearInterpolate(in,
					       shift_r<npixels>(uu),
					       shift_r<npixels>(vv),
					       cvt<short, 1>(du4),
					       cvt<short, 1>(dv4))));
	outp += Iu8vec::size/4;
	    
	du4 = quadup<1>(du);
	dv4 = quadup<1>(dv);
	storeu((u_char*)outp,
	       cvt<u_char>(bilinearInterpolate(in,
					       shift_r<2*npixels>(uu),
					       shift_r<2*npixels>(vv),
					       cvt<short, 0>(du4),
					       cvt<short, 0>(dv4)),
			   bilinearInterpolate(in,
					       shift_r<3*npixels>(uu),
					       shift_r<3*npixels>(vv),
					       cvt<short, 1>(du4),
					       cvt<short, 1>(dv4))));
	outp += Iu8vec::size/4;
	usp  += Is16vec::size;
	vsp  += Is16vec::size;
	    
	uu  = load(usp);
	vv  = load(vsp);
	du4 = quadup<2>(du);
	dv4 = quadup<2>(dv);
	storeu((u_char*)outp,
	       cvt<u_char>(bilinearInterpolate(in, uu, vv,
					       cvt<short, 0>(du4),
					       cvt<short, 0>(dv4)),
			   bilinearInterpolate(in,
					       shift_r<npixels>(uu),
					       shift_r<npixels>(vv),
					       cvt<short, 1>(du4),
					       cvt<short, 1>(dv4))));
	outp += Iu8vec::size/4;
	    
	du4 = quadup<3>(du);
	dv4 = quadup<3>(dv);
	storeu((u_char*)outp,
	       cvt<u_char>(bilinearInterpolate(in,
					       shift_r<2*npixels>(uu),
					       shift_r<2*npixels>(vv),
					       cvt<short, 0>(du4),
					       cvt<short, 0>(dv4)),
			   bilinearInterpolate(in,
					       shift_r<3*npixels>(uu),
					       shift_r<3*npixels>(vv),
					       cvt<short, 1>(du4),
					       cvt<short, 1>(dv4))));
	outp += Iu8vec::size/4;
	usp  += Is16vec::size;
	vsp  += Is16vec::size;
	
	dup  += Iu8vec::size;
	dvp  += Iu8vec::size;
    }
#  if !defined(SSE2)
    empty();
#  endif	
#endif
    while (outp < outq)
	*outp++ = bilinearInterpolate(in, *usp++, *vsp++, *dup++, *dvp++);
    out[v].setLimits(_fracs[v].lmost, _fracs[v].lmost + _fracs[v].width());
}

template <> __PORT void
Warp::warpLine(const Image<YUV444>& in, Image<YUV444>& out, u_int v) const
{
    const short		*usp  = _fracs[v].us.data(), *vsp = _fracs[v].vs.data();
    const u_char	*dup  = _fracs[v].du.data(), *dvp = _fracs[v].dv.data();
    YUV444		*outp = out[v].data() + _fracs[v].lmost;
    YUV444* const	outq  = outp + _fracs[v].width();

    while (outp < outq)
	*outp++ = bilinearInterpolate(in, *usp++, *vsp++, *dup++, *dvp++);
    out[v].setLimits(_fracs[v].lmost, _fracs[v].lmost + _fracs[v].width());
}

template <> __PORT void
Warp::warpLine(const Image<u_char>& in, Image<u_char>& out, u_int v) const
{
    const short		*usp  = _fracs[v].us.data(), *vsp = _fracs[v].vs.data();
    const u_char	*dup  = _fracs[v].du.data(), *dvp = _fracs[v].dv.data();
    u_char		*outp = out[v].data() + _fracs[v].lmost;
    u_char* const	outq  = outp + _fracs[v].width();
#if defined(SSE)
    using namespace	mm;

    for (u_char* const outr = outq - Iu8vec::size; outp <= outr; )
    {
	Iu8vec	du = load(dup), dv = load(dvp);
	Is16vec	out0 = bilinearInterpolate(in, load(usp), load(vsp),
					   cvt<short>(du), cvt<short>(dv));
	usp += Is16vec::size;
	vsp += Is16vec::size;
	storeu(outp, cvt<u_char>(out0,
				 bilinearInterpolate(in,
						     load(usp), load(vsp),
						     cvt<short, 1>(du),
						     cvt<short, 1>(dv))));
	usp  += Is16vec::size;
	vsp  += Is16vec::size;
	dup  += Iu8vec::size;
	dvp  += Iu8vec::size;
	outp += Iu8vec::size;
    }
#  if !defined(SSE2)
    empty();
#  endif	
#endif
    while (outp < outq)
	*outp++ = bilinearInterpolate(in, *usp++, *vsp++, *dup++, *dvp++);
    out[v].setLimits(_fracs[v].lmost, _fracs[v].lmost + _fracs[v].width());
}

template __PORT void
Warp::warpLine(const Image<RGBA>& in, Image<RGBA>& out, u_int v) const;
template __PORT void
Warp::warpLine(const Image<ARGB>& in, Image<ARGB>& out, u_int v) const;
template __PORT void
Warp::warpLine(const Image<ABGR>& in, Image<ABGR>& out, u_int v) const;
template __PORT void
Warp::warpLine(const Image<BGRA>& in, Image<BGRA>& out, u_int v) const;
}
