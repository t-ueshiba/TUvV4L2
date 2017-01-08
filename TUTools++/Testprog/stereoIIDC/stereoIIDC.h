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
 *  $Id: corrStereo.h 1456 2013-11-12 23:54:08Z ueshiba $
 */
#ifndef __TU_STEREOIIDC_H
#define __TU_STEREOIIDC_H

#include <fstream>
#include <string>
#include <sys/time.h>
#include "TU/v/TUv++.h"

/************************************************************************
*  global data and definitions						*
************************************************************************/
namespace TU
{
namespace v
{
enum
{
    c_Frame,

  // File menu
    c_OpenDisparityMap,
    c_RestoreConfig,
    c_SaveConfig,
    c_SaveMatrices,
    c_SaveRectifiedImages,
    c_SaveThreeD,
    c_SaveThreeDImage,
    
  // Camera control.
    c_ContinuousShot,
    c_OneShot,
    c_Cursor,
    c_DisparityLabel,
    c_Disparity,
    c_Trigger,

  // Stereo matching parameters
    c_Algorithm,
    c_SAD,
    c_GuidedFilter,
    c_TreeFilter,
    c_Binocular,
    c_DoHorizontalBackMatch,
    c_DoVerticalBackMatch,
    c_DisparitySearchWidth,
    c_DisparityMax,
    c_DisparityInconsistency,
    c_WindowSize,
    c_IntensityDiffMax,
    c_DerivativeDiffMax,
    c_Blend,
    c_Regularization,
    c_DepthRange,
    c_WMF,
    c_WMFWindowSize,
    c_WMFSigma,
    c_RefineDisparity,
    
  // Viewing control.
    c_DrawMode,
    c_Texture,
    c_Polygon,
    c_Mesh,
    c_MoveViewpoint,
    c_GazeDistance,
    c_SwingView,
    c_StereoView,
    c_Refresh,
};

/************************************************************************
*  global functions							*
************************************************************************/
template <class CAMERA> CmdDef*
createMenuCmds(CAMERA& camera)
{
    static MenuDef fileMenu[] =
    {
	{"Open stereo images",		M_Open,			false, noSub},
	{"Save stereo images",		M_Save,			false, noSub},
	{"Save rectified images",	c_SaveRectifiedImages,	false, noSub},
	{"Save 3D data",		c_SaveThreeD,		false, noSub},
#if defined(DISPLAY_3D)
	{"Save 3D-rendered image",	c_SaveThreeDImage,	false, noSub},
#endif
	{"-",				M_Line,			false, noSub},
	{"Save camera config.",		c_SaveConfig,		false, noSub},
	{"Save camera matrices",	c_SaveMatrices,		false, noSub},
	{"-",				M_Line,			false, noSub},
	{"Quit",			M_Exit,			false, noSub},
	EndOfMenu
    };

    static float  range[] = {0.5, 3.0, 0};

    static CmdDef radioButtonCmds[] =
    {
	{C_RadioButton, c_Texture, 0, "Texture", noProp, CA_None,
	 0, 0, 1, 1, 0},
	{C_RadioButton, c_Polygon, 0, "Polygon", noProp, CA_None,
	 1, 0, 1, 1, 0},
	{C_RadioButton, c_Mesh,    0, "Mesh",    noProp, CA_None,
	 2, 0, 1, 1, 0},
	EndOfCmds
    };
    
    static CmdDef menuCmds[] =
    {
	{C_MenuButton, M_File,		0,
	 "File",			fileMenu, CA_None, 0, 0, 1, 1, 0},
	{C_MenuButton, M_Format, 0,
	 "Format",			noProp, CA_None, 1, 0, 1, 1, 0},
#if defined(DISPLAY_3D)
	{C_ChoiceFrame,  c_DrawMode,	 c_Texture,
	 "",		radioButtonCmds,	CA_NoBorder, 2, 0, 1, 1, 0},
	{C_Slider, c_GazeDistance,	1.3,
	 "Gaze distance",		range,	CA_None, 3, 0, 1, 1, 0},
	{C_ToggleButton, c_SwingView,   0,
	 "Swing view",			noProp,	CA_None, 4, 0, 1, 1, 0},
	{C_ToggleButton, c_StereoView,	0,
	 "Stereo view",			noProp, CA_None, 5, 0, 1, 1, 0},
#endif
	EndOfCmds
    };

    menuCmds[1].prop = createFormatMenu(camera);

    return menuCmds;
}

inline CmdDef*
createCaptureCmds()
{
    static float	prop[6][3];
    static CmdDef	captureCmds[] =
    {
	{C_ToggleButton, c_ContinuousShot, 0,
	 "Continuous shot",		noProp,  CA_None,     0, 0, 1, 1, 0},
	{C_Button, c_OneShot,		0,
	 "One shot",			noProp,  CA_None,     0, 1, 1, 1, 0},
	{C_ToggleButton,  c_DoHorizontalBackMatch,	0,
	 "Hor. back match",		noProp,  CA_None,     5, 0, 1, 1, 0}, 
	{C_ToggleButton,  c_DoVerticalBackMatch,	0,
	"Ver. back match",		noProp,  CA_None,     6, 0, 1, 1, 0},
	{C_ToggleButton,c_Binocular,	0,
	 "Binocular",			noProp,  CA_None,     1, 0, 1, 1, 0},
	{C_Label, c_Cursor,		0,
	 "(   ,   )",			noProp,  CA_None,     2, 0, 1, 1, 0},
	{C_Label, c_DisparityLabel,	0,
	 "Disparity:",			noProp,  CA_NoBorder, 1, 1, 1, 1, 0},
	{C_Label, c_Disparity,		0,
	 "     ",			noProp,  CA_None,     2, 1, 1, 1, 0},
	{C_Label, c_DepthRange,		0,
	 "",				noProp,	 CA_NoBorder, 0, 8, 3, 1, 0},
	{C_Slider, c_WindowSize,	0,
	 "Window size",			prop[0], CA_None,     0, 2, 3, 1, 0},
	{C_Slider, c_DisparitySearchWidth,	0,
	 "Disparity search width",	prop[1], CA_None,     0, 3, 3, 1, 0},
	{C_Slider, c_DisparityMax,	0,
	 "Maximum disparity",		prop[2], CA_None,     0, 4, 3, 1, 0},
	{C_Slider, c_DisparityInconsistency,	0,
	 "Disparity inconsistency",	prop[3], CA_None,     0, 5, 3, 1, 0},
	{C_Slider, c_IntensityDiffMax,	0,
	 "Maximum intensity diff.",	prop[4], CA_None,     0, 6, 3, 1, 0},
	{C_Slider, c_Regularization,	0,
	 "Regularization",		prop[5], CA_None,     0, 7, 3, 1, 0},
	EndOfCmds
    };
    
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

    return captureCmds;
}

}	// namespace v

inline void
countTime()
{
    static int		nframes = 0;
    static timeval	start;
    
    if (nframes == 10)
    {
	timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

}	// namespace TU
#endif
