/*
 *  $Id: createFormatMenu.cc,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include <vector>
#include <list>
#include <sstream>
#include <boost/foreach.hpp>
#include "TU/v/vV4L2++.h"

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
    
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*
createFormatMenu(const V4L2Camera& camera)
{
    static const char*	setROILabel = "set ROI";
    
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

  // ROIを指定する項目を作る．
    pixelFormatMenus.push_back(MenuDef());
    MenuDef&	setROIMenu = pixelFormatMenus.back();
    setROIMenu.label	= setROILabel;
    setROIMenu.id	= V4L2Camera::UNKNOWN_PIXEL_FORMAT;
    setROIMenu.checked	= false;
    setROIMenu.submenu	= noSub;
    
    pixelFormatMenus.push_back(MenuDef());
    pixelFormatMenus.back().label = 0;

    return &pixelFormatMenus.front();
}
 
}	// namespace v
}	// namespace TU
