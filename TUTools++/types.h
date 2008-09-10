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
 *  $Id: types.h,v 1.8 2008-09-10 05:10:50 ueshiba Exp $
 */
#ifndef __TUtypes_h
#define __TUtypes_h

#ifdef WIN32
typedef unsigned int	size_t;
typedef unsigned char	u_char;
typedef unsigned short	u_short;
typedef unsigned int	u_int;
typedef unsigned long	u_long;
#else
#  include <sys/types.h>
#endif

typedef signed char		s_char;
typedef long long		int64;
typedef unsigned long long	u_int64;
/*
#ifdef __INTEL_COMPILER
extern "C"
{
    unsigned long long	strtoull(const char*, char**, int);
}
#endif
*/
#endif	/*  !__TUtypes_h	*/
