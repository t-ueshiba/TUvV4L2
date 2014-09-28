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
 *  $Id: createMenuCmds.cc 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include <vector>
#include <list>
#include <sstream>
#include <boost/foreach.hpp>
#include "stereoV4L2.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static std::vector<MenuDef>			pixelFormatMenus;
static std::list<std::vector<MenuDef> >		frameSizeMenusList;
static std::list<std::vector<std::string> >	frameSizeLabelsList;
    
static MenuDef fileMenu[] =
{
    {"Open stereo images",	M_Open,			false, noSub},
    {"Save stereo images",	M_Save,			false, noSub},
    {"Save rectified images",	c_SaveRectifiedImages,	false, noSub},
    {"Save 3D data",		c_SaveThreeD,		false, noSub},
#if defined(DISPLAY_3D)
    {"Save 3D-rendered image",	c_SaveThreeDImage,	false, noSub},
#endif
    {"-",			M_Line,			false, noSub},
    {"Save camera config.",	c_SaveConfig,		false, noSub},
    {"Save camera matrices",	c_SaveMatrices,		false, noSub},
    {"-",			M_Line,			false, noSub},
    {"Quit",			M_Exit,			false, noSub},
    EndOfMenu
};

static int	range[] = {50, 300, 100};

static CmdDef RadioButtonCmds[] =
{
    {C_RadioButton, c_Texture, 0, "Texture", noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_RadioButton, c_Polygon, 0, "Polygon",  noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_RadioButton, c_Mesh,    0, "Mesh",     noProp, CA_None,
     2, 0, 1, 1, 0},
    EndOfCmds
};
    
static CmdDef MenuCmds[] =
{
    {C_MenuButton, M_File,   0, "File",   fileMenu,   CA_None, 0, 0, 1, 1, 0},
    {C_MenuButton,   c_PixelFormat,    0, "Pixel format",  0,	   CA_None,
     1, 0, 1, 1, 0},
#if defined(DISPLAY_3D)
    {C_ChoiceFrame,  c_DrawMode, c_Texture, "", RadioButtonCmds,   CA_NoBorder,
     2, 0, 1, 1, 0},
    {C_Slider,	     c_GazeDistance, 130, "Gaze distance", range,  CA_None,
     3, 0, 1, 1, 0},
    {C_ToggleButton, c_SwingView,      0, "Swing view",	   noProp, CA_None,
     4, 0, 1, 1, 0},
    {C_ToggleButton, c_StereoView,     0, "Stereo view",   noProp, CA_None,
     5, 0, 1, 1, 0},
#endif
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createMenuCmds(const V4L2Camera& camera)
{
    BOOST_FOREACH (V4L2Camera::PixelFormat pixelFormat,
		   camera.availablePixelFormats())
    {
      // この画素フォーマットに対応するメニュー項目を作る．
	pixelFormatMenus.push_back(MenuDef());
	MenuDef&	pixelFormatMenu = pixelFormatMenus.back();

	pixelFormatMenu.label	= camera.getName(pixelFormat).c_str();
	pixelFormatMenu.id	= pixelFormat;
	pixelFormatMenu.checked = true;

      // この画素フォーマットがサポートする各フレームサイズに対応するメニュー項目を作る．
	frameSizeMenusList.push_back(std::vector<MenuDef>());
	std::vector<MenuDef>&	frameSizeMenus = frameSizeMenusList.back();
	frameSizeLabelsList.push_back(std::vector<std::string>());
	std::vector<std::string>&
	    frameSizeLabels = frameSizeLabelsList.back();
	BOOST_FOREACH (const V4L2Camera::FrameSize& frameSize,
		       camera.availableFrameSizes(pixelFormat))
	{
	  // このフレームサイズに対応するメニュー項目を作る．
	    frameSizeMenus.push_back(MenuDef());
	    MenuDef&		frameSizeMenu = frameSizeMenus.back();
	    const size_t	j = frameSizeMenus.size() - 1;

	    std::ostringstream	s;
	    s << frameSize;
	    frameSizeLabels.push_back(s.str());
	    frameSizeMenu.label = frameSizeLabels.back().c_str();
	    frameSizeMenu.id	= j;
	    frameSizeMenu.checked
		= (camera.pixelFormat() == pixelFormat	    &&
		   frameSize.width.involves(camera.width()) &&
		   frameSize.height.involves(camera.height()));
	    frameSizeMenu.submenu = noSub;
	}
	frameSizeMenus.push_back(MenuDef());
	frameSizeMenus.back().label = 0;

	pixelFormatMenu.submenu = &frameSizeMenus.front();
    }
    pixelFormatMenus.push_back(MenuDef());
    pixelFormatMenus.back().label = 0;

    MenuCmds[1].prop = &pixelFormatMenus.front();

    return MenuCmds;
}
 
}
}
