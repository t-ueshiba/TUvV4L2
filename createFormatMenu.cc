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
static constexpr struct
{
    IIDCCamera::Format	format;
    const char*		name;
    
} formats[] =
{
    {IIDCCamera::YUV444_160x120,	"160x120-YUV(4:4:4)"},
    {IIDCCamera::YUV422_320x240,	"320x240-YUV(4:2:2)"},
    {IIDCCamera::YUV411_640x480,	"640x480-YUV(4:1:1)"},
    {IIDCCamera::YUV422_640x480,	"640x480-YUV(4:2:2)"},
    {IIDCCamera::RGB24_640x480,		"640x480-RGB"},
    {IIDCCamera::MONO8_640x480,		"640x480-Y(mono)"},
    {IIDCCamera::MONO16_640x480,	"640x480-Y(mono16)"},
    {IIDCCamera::YUV422_800x600,	"800x600-YUV(4:2:2)"},
    {IIDCCamera::RGB24_800x600,		"800x600-RGB"},
    {IIDCCamera::MONO8_800x600,		"800x600-Y(mono)"},
    {IIDCCamera::YUV422_1024x768,	"1024x768-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1024x768,	"1024x768-RGB"},
    {IIDCCamera::MONO8_1024x768,	"1024x768-Y(mono)"},
    {IIDCCamera::MONO16_800x600,	"800x600-Y(mono16)"},
    {IIDCCamera::MONO16_1024x768,	"1024x768-Y(mono16)"},
    {IIDCCamera::YUV422_1280x960,	"1280x960-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1280x960,	"1280x960-RGB"},
    {IIDCCamera::MONO8_1280x960,	"1280x960-Y(mono)"},
    {IIDCCamera::YUV422_1600x1200,	"1600x1200-YUV(4:2:2)"},
    {IIDCCamera::RGB24_1600x1200,	"1600x1200-RGB"},
    {IIDCCamera::MONO8_1600x1200,	"1600x1200-Y(mono)"},
    {IIDCCamera::MONO16_1280x960,	"1280x960-Y(mono16)"},
    {IIDCCamera::MONO16_1600x1200,	"1600x1200-Y(mono16)"},
    {IIDCCamera::Format_7_0,		"Format_7_0"},
    {IIDCCamera::Format_7_1,		"Format_7_1"},
    {IIDCCamera::Format_7_2,		"Format_7_2"},
    {IIDCCamera::Format_7_3,		"Format_7_3"},
    {IIDCCamera::Format_7_4,		"Format_7_4"},
    {IIDCCamera::Format_7_5,		"Format_7_5"},
    {IIDCCamera::Format_7_6,		"Format_7_6"},
    {IIDCCamera::Format_7_6,		"Format_7_7"}
};
static constexpr int	NFORMATS = sizeof(formats)/sizeof(formats[0]);

static constexpr struct
{
    IIDCCamera::FrameRate	frameRate;
    const char*			name;
} frameRates[] =
{
    {IIDCCamera::FrameRate_1_875,	"1.875fps"},
    {IIDCCamera::FrameRate_3_75,	"3.75fps"},
    {IIDCCamera::FrameRate_7_5,		"7.5fps"},
    {IIDCCamera::FrameRate_15,		"15fps"},
    {IIDCCamera::FrameRate_30,		"30fps"},
    {IIDCCamera::FrameRate_60,		"60fps"},
    {IIDCCamera::FrameRate_x,		"custom frame rate"}
};
static constexpr int	NRATES = sizeof(frameRates)/sizeof(frameRates[0]);

static MenuDef		formatMenus[NFORMATS + 1];
static MenuDef		rateMenus[NFORMATS][NRATES + 1];

/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*
createFormatMenu(const IIDCCamera& camera)
{
    auto	current_format = camera.getFormat();
    auto	current_rate   = camera.getFrameRate();
    size_t	nitems = 0;
    for (const auto& format : formats)
    {
	auto	inq = camera.inquireFrameRate(format.format);
	size_t	nrates = 0;
	for (const auto& frameRate : frameRates)
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
	rateMenus[nitems][nrates].label = 0;
	
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
    formatMenus[nitems].label = 0;

    return formatMenus;
}

}	// namespace v
}	// namespace TU
