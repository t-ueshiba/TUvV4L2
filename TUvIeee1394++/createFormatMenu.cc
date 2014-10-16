/*
 *  $Id: createFormatMenu.cc,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include "TU/v/vIeee1394++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
struct Format
{
    Ieee1394Camera::Format	format;
    const char*			name;
    
};
static Format formats[] =
{
    {Ieee1394Camera::YUV444_160x120,	"160x120-YUV(4:4:4)"},
    {Ieee1394Camera::YUV422_320x240,	"320x240-YUV(4:2:2)"},
    {Ieee1394Camera::YUV411_640x480,	"640x480-YUV(4:1:1)"},
    {Ieee1394Camera::YUV422_640x480,	"640x480-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_640x480,	"640x480-RGB"},
    {Ieee1394Camera::MONO8_640x480,	"640x480-Y(mono)"},
    {Ieee1394Camera::MONO16_640x480,	"640x480-Y(mono16)"},
    {Ieee1394Camera::YUV422_800x600,	"800x600-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_800x600,	"800x600-RGB"},
    {Ieee1394Camera::MONO8_800x600,	"800x600-Y(mono)"},
    {Ieee1394Camera::YUV422_1024x768,	"1024x768-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1024x768,	"1024x768-RGB"},
    {Ieee1394Camera::MONO8_1024x768,	"1024x768-Y(mono)"},
    {Ieee1394Camera::MONO16_800x600,	"800x600-Y(mono16)"},
    {Ieee1394Camera::MONO16_1024x768,	"1024x768-Y(mono16)"},
    {Ieee1394Camera::YUV422_1280x960,	"1280x960-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1280x960,	"1280x960-RGB"},
    {Ieee1394Camera::MONO8_1280x960,	"1280x960-Y(mono)"},
    {Ieee1394Camera::YUV422_1600x1200,	"1600x1200-YUV(4:2:2)"},
    {Ieee1394Camera::RGB24_1600x1200,	"1600x1200-RGB"},
    {Ieee1394Camera::MONO8_1600x1200,	"1600x1200-Y(mono)"},
    {Ieee1394Camera::MONO16_1280x960,	"1280x960-Y(mono16)"},
    {Ieee1394Camera::MONO16_1600x1200,	"1600x1200-Y(mono16)"},
    {Ieee1394Camera::Format_7_0,	"Format_7_0"},
    {Ieee1394Camera::Format_7_1,	"Format_7_1"},
    {Ieee1394Camera::Format_7_2,	"Format_7_2"},
    {Ieee1394Camera::Format_7_3,	"Format_7_3"},
    {Ieee1394Camera::Format_7_4,	"Format_7_4"},
    {Ieee1394Camera::Format_7_5,	"Format_7_5"},
    {Ieee1394Camera::Format_7_6,	"Format_7_6"},
    {Ieee1394Camera::Format_7_6,	"Format_7_7"}
};
static const int	NFORMATS = sizeof(formats)/sizeof(formats[0]);

struct FrameRate
{
    Ieee1394Camera::FrameRate	frameRate;
    const char*			name;
};
static FrameRate frameRates[] =
{
    {Ieee1394Camera::FrameRate_1_875,	"1.875fps"},
    {Ieee1394Camera::FrameRate_3_75,	"3.75fps"},
    {Ieee1394Camera::FrameRate_7_5,	"7.5fps"},
    {Ieee1394Camera::FrameRate_15,	"15fps"},
    {Ieee1394Camera::FrameRate_30,	"30fps"},
    {Ieee1394Camera::FrameRate_60,	"60fps"},
    {Ieee1394Camera::FrameRate_x,	"custom frame rate"}
};
static const int	NRATES = sizeof(frameRates)/sizeof(frameRates[0]);

static MenuDef		formatMenus[NFORMATS + 1];
static MenuDef		rateMenus[NFORMATS][NRATES + 1];

/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*
createFormatMenu(const Ieee1394Camera& camera)
{
    Ieee1394Camera::Format	current_format = camera.getFormat();
    Ieee1394Camera::FrameRate	current_rate   = camera.getFrameRate();
    size_t			nitems = 0;
    for (size_t i = 0; i < NFORMATS; ++i)
    {
	u_int	inq = camera.inquireFrameRate(formats[i].format);
	size_t	nrates = 0;
	for (size_t j = 0; j < NRATES; ++j)
	{
	    if (inq & frameRates[j].frameRate)
	    {
	      // Create submenu items for setting frame rate.
		rateMenus[nitems][nrates].label	   = frameRates[j].name;
		rateMenus[nitems][nrates].id	   = frameRates[j].frameRate;
		rateMenus[nitems][nrates].checked
		    = ((current_format == formats[i].format) &&
		       (current_rate == frameRates[j].frameRate));
		rateMenus[nitems][nrates].submenu   = noSub;
		++nrates;
	    }
	}
	rateMenus[nitems][nrates].label = 0;
	
	if (nrates != 0)
	{
	  // Create menu items for setting format.
	    formatMenus[nitems].label	 = formats[i].name;
	    formatMenus[nitems].id	 = formats[i].format;
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
