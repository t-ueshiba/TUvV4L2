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
 *  $Id: WarpMT.cc,v 1.4 2010-07-07 07:33:39 ueshiba Exp $  
 */
#include "TU/Thread++.h"
#include "TU/Warp.h"

namespace TU
{
/************************************************************************
*  class MultiThread2<Warp, Image<T> >					*
************************************************************************/
template <> __PORT void
MultiThread2<Warp, Image<u_char> >::
  operator ()(const Image<u_char>& in, Image<u_char>& out) const
{
    out.resize(height(), width());
    raiseThreads(in, out);
}
    
template <> __PORT void
MultiThread2<Warp, Image<RGBA> >::
  operator ()(const Image<RGBA>& in, Image<RGBA>& out) const
{
    out.resize(height(), width());
    raiseThreads(in, out);
}
    
template <> __PORT void
MultiThread2<Warp, Image<ABGR> >::
  operator ()(const Image<ABGR>& in, Image<ABGR>& out) const
{
    out.resize(height(), width());
    raiseThreads(in, out);
}
    
}
