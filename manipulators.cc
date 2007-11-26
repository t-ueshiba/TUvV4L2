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
 *  $Id: manipulators.cc,v 1.4 2007-11-26 07:55:48 ueshiba Exp $
 */
#include <iostream>

namespace TU
{
/************************************************************************
*  Manipulators for std::istream					*
************************************************************************/
std::istream&
ign(std::istream& in)	// manipulator for skipping the rest of a line
{
    char	c;
    while (in.get(c))
	if (c == '\n')
	    break;
    return in;
}

std::istream&
skipl(std::istream& in)
{
    char	c;
    
    while (in.get(c))
	if (c == '\n' || c == '\r')
	    break;
    return in;
}
 
}
