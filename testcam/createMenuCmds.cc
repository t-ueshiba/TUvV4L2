/*
 *  $Id$
 */
#include <vector>
#include <list>
#include <boost/foreach.hpp>
#include <sstream>
#include "testcam.h"

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
    {"Save",			M_Save,		 false, noSub},
    {"Restore camera config.",	c_RestoreConfig, false, noSub},
    {"Save camera config.",	c_SaveConfig,	 false, noSub},
    {"-",			M_Line,		 false, noSub},
    {"Quit",			M_Exit,		 false, noSub},
    EndOfMenu
};

static CmdDef MenuCmds[] =
{
    {C_MenuButton,	  M_File, 0,	     "File", fileMenu, CA_None,
     0, 0, 1, 1, 0},
    {C_MenuButton, c_PixelFormat, 0, "Pixel format",	    0, CA_None,
     1, 0, 1, 1, 0},
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
