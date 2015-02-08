/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
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
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: createCaptureCmds.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "stereo1394.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static int	prop[6][3];
static CmdDef	CaptureCmds[] =
{
    {C_ToggleButton,  c_ContinuousShot, 0, "Continuous shot", noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_Button,	      c_OneShot,        0, "One shot",	      noProp, CA_None,
     0, 1, 1, 1, 0},
  /*
    {C_ToggleButton,  c_DoHorizontalBackMatch,	0,
     "Hor. back match", noProp, CA_None, 5, 0, 1, 1, 0}, 
    {C_ToggleButton,  c_DoVerticalBackMatch,	0,
     "Ver. back match", noProp, CA_None, 6, 0, 1, 1, 0},
  */
    {C_ToggleButton,  c_Binocular,	0, "Binocular",	      noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_Label,	      c_Cursor,		0, "(   ,   )",	      noProp, CA_None,
     2, 0, 1, 1, 0},
    {C_Label,	      c_DisparityLabel,	0, "Disparity:",  noProp, CA_NoBorder,
     1, 1, 1, 1, 0},
    {C_Label,	      c_Disparity,	0, "     ",	      noProp, CA_None,
     2, 1, 1, 1, 0},
    {C_Label,	      c_DepthRange,	0, "",		noProp, CA_NoBorder,
     0, 8, 3, 1, 0},
    {C_Slider,	      c_WindowSize,	0,
     "Window size", prop[0], CA_None, 0, 2, 3, 1, 0},
    {C_Slider, c_DisparitySearchWidth,	0,
     "Disparity search width", prop[1], CA_None, 0, 3, 3, 1, 0},
    {C_Slider, c_DisparityMax,		0,
     "Maximum disparity", prop[2], CA_None, 0, 4, 3, 1, 0},
    {C_Slider, c_DisparityInconsistency,0,
     "Disparity inconsistency", prop[3], CA_None, 0, 5, 3, 1, 0},
    {C_Slider, c_IntensityDiffMax,	0,
     "Maximum intensity diff.", prop[4], CA_None, 0, 6, 3, 1, 0},
    {C_Slider, c_Regularization,	0,
     "Regularization", prop[5],  CA_None, 0, 7, 3, 1, 0},
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createCaptureCmds()
{
  // Window size
    prop[0][0]  = 3;
    prop[0][1]  = 31;
    prop[0][2]  = 1;

  // Disparity search width
    prop[1][0]  = 16;
    prop[1][1]  = 192;
    prop[1][2]  = 1;

  // Max. disparity
    prop[2][0]  = 48;
    prop[2][1]  = 192;
    prop[2][2]  = 1;

  // Disparity inconsistency
    prop[3][0]  = 0;
    prop[3][1]  = 20;
    prop[3][2]  = 1;

  // Max. intensity diff.
    prop[4][0]  = 0;
    prop[4][1]  = 100;
    prop[4][2]  = 1;
    
  // Regularization
    prop[5][0]  = 1;
    prop[5][1]  = 255;
    prop[5][2]  = 1;

    return CaptureCmds;
}
 
}
}
