/*
 *  $Id: createFormatMenu.cc,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include "TU/v/vIIDC++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static MenuDef	formatMenus[IIDCCamera::NFORMATS + 1];
static MenuDef	rateMenus[IIDCCamera::NFORMATS][IIDCCamera::NRATES + 1];

/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*
createFormatMenu(const IIDCCamera& camera)
{
    const auto	current_format = camera.getFormat();
    const auto	current_rate   = camera.getFrameRate();
    size_t	nitems = 0;
    for (const auto& format : IIDCCamera::formatNames)
    {
	auto	inq = camera.inquireFrameRate(format.format);
	size_t	nrates = 0;
	for (const auto& frameRate : IIDCCamera::frameRateNames)
	{
	    if (inq & frameRate.frameRate)
	    {
	      // Create submenu items for setting frame rate.
		rateMenus[nitems][nrates].label	= frameRate.name;
		rateMenus[nitems][nrates].id	= frameRate.frameRate;
		rateMenus[nitems][nrates].checked
		    = ((current_format == format.format) &&
		       (current_rate == frameRate.frameRate));
		rateMenus[nitems][nrates].submenu = noSub;
		++nrates;
	    }
	}
	rateMenus[nitems][nrates].label = nullptr;
	
	if (nrates != 0)
	{
	  // Create menu items for setting format.
	    formatMenus[nitems].label	 = format.name;
	    formatMenus[nitems].id	 = format.format;
	    formatMenus[nitems].checked	 = true;
	    formatMenus[nitems].submenu	 = rateMenus[nitems];
	    ++nitems;
	}
    }
    formatMenus[nitems].label = nullptr;

    return formatMenus;
}

}	// namespace v
}	// namespace TU
