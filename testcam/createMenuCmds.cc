/*
 *  $Id: createMenuCmds.cc,v 1.1 2012-06-19 06:14:31 ueshiba Exp $
 */
#include <vector>
#include <boost/foreach.hpp>
#include "multicam.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static std::vector<MenuDef>	pixelFormatMenus;

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
    {C_MenuButton, M_File, 0, "File", fileMenu,   CA_None, 0, 0, 1, 1, 0},
    {C_MenuButton, c_PixelFormat, 0, "Pixel format", 0, CA_None, 1, 0, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createMenuCmds(const V4L2Camera& camera)
{
     V4L2Camera::PixelFormatRange
	pixelFormats = camera.availablePixelFormats();

    pixelFormatMenus.resize(std::distance(pixelFormats.first,
					  pixelFormats.second) + 1);
    int	i = 0;
    BOOST_FOREACH (V4L2Camera::PixelFormat pixelFormat, pixelFormats)
    {
	pixelFormatMenus[i].label   = camera.getName(pixelFormat).c_str();
	pixelFormatMenus[i].id	    = pixelFormat;
	pixelFormatMenus[i].checked = false;
	pixelFormatMenus[i].submenu = noSub;

	++i;
    }
    pixelFormatMenus[i].label = 0;

    MenuCmds[1].prop = &pixelFormatMenus[0];

    return MenuCmds;
}
 
}
}
