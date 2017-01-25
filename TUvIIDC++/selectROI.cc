/*
 *  $Id$
 */
#include "TU/v/vIIDC++.h"
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class IIDCModalDialog						*
************************************************************************/
class IIDCModalDialog : public ModalDialog
{
  public:
    typedef IIDCCamera::Format_7_Info	Format_7_Info;
    typedef IIDCCamera::PixelFormat	PixelFormat;
    
  private:
    enum	{c_U0, c_V0, c_Width, c_Height,
		 c_PixelFormat, c_PacketSize, c_OK};

  public:
    IIDCModalDialog(Window& parentWindow, const Format_7_Info& fmt7info);
    
    PixelFormat		selectROI(size_t& u0, size_t& v0,
				  size_t& width, size_t& height,
				  size_t& packetSize)			;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    CmdDef*		createROICmds(const Format_7_Info& fmt7info)	;
    
  private:
    const Format_7_Info&	_fmt7info;
    float			_ranges[5][3];
    MenuDef			_pixelFormatMenus[IIDCCamera::NPIXELFORMATS+1];
    CmdDef			_cmds[8];
};
    
IIDCModalDialog::IIDCModalDialog(Window& parentWindow,
				 const Format_7_Info& fmt7info)
    :ModalDialog(parentWindow, "ROI for Format_7_x", createROICmds(fmt7info)),
     _fmt7info(fmt7info)
{
}
    
IIDCCamera::PixelFormat
IIDCModalDialog::selectROI(size_t& u0, size_t& v0,
			   size_t& width, size_t& height, size_t& packetSize)
{
    show();
    u0		= pane().getValue(c_U0);
    v0		= pane().getValue(c_V0);
    width	= pane().getValue(c_Width);
    height	= pane().getValue(c_Height);
    packetSize	= pane().getValue(c_PacketSize);
    
    return IIDCCamera::uintToPixelFormat(pane().getValue(c_PixelFormat));
}

void
IIDCModalDialog::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_U0:
      {
	size_t	u0 = _fmt7info.unitU0
		   * ((val + _fmt7info.unitU0/2) / _fmt7info.unitU0);
	pane().setValue(c_U0, u0);
      }
	break;
      case c_V0:
      {
	size_t	v0 = _fmt7info.unitV0
		   * ((val + _fmt7info.unitV0/2) / _fmt7info.unitV0);
	pane().setValue(c_V0, v0);
      }
	break;
      case c_Width:
      {
	size_t	w = _fmt7info.unitWidth
		  * ((val + _fmt7info.unitWidth/2) / _fmt7info.unitWidth);
	pane().setValue(c_Width, w);
      }
	break;
      case c_Height:
      {
	size_t	h = _fmt7info.unitHeight
		  * ((val + _fmt7info.unitHeight/2) / _fmt7info.unitHeight);
	pane().setValue(c_Height, h);
      }
	break;
      case c_PacketSize:
      {
	size_t	s = _fmt7info.unitBytePerPacket
		  * ((val + _fmt7info.unitBytePerPacket/2) /
		     _fmt7info.unitBytePerPacket);
	pane().setValue(c_PacketSize, s);
      }
	break;
	
      case c_OK:
	hide();
	break;
    }
}

CmdDef*
IIDCModalDialog::createROICmds(const Format_7_Info& fmt7info)
{
  // Create commands for setting ROI.
    _ranges[0][0] = 0;
    _ranges[0][1] = fmt7info.maxWidth;
    _ranges[0][2] = fmt7info.unitU0;
    _cmds[0]	  = {C_Slider, c_U0,	 fmt7info.u0,	  "    u0", _ranges[0],
		     CA_None, 0, 0, 1, 1, 0};
    
    _ranges[1][0] = 0;
    _ranges[1][1] = fmt7info.maxHeight;
    _ranges[1][2] = fmt7info.unitV0;
    _cmds[1]	  = {C_Slider, c_V0,	 fmt7info.v0,	  "    v0", _ranges[1],
		     CA_None, 0, 1, 1, 1, 0};
    
    _ranges[2][0] = 0;
    _ranges[2][1] = fmt7info.maxWidth;
    _ranges[2][2] = fmt7info.unitWidth;
    _cmds[2]	  = {C_Slider, c_Width,  fmt7info.width,  " width", _ranges[2],
		     CA_None, 0, 2, 1, 1, 0};
    
    _ranges[3][0] = 0;
    _ranges[3][1] = fmt7info.maxHeight;
    _ranges[3][2] = fmt7info.unitHeight;
    _cmds[3]	  = {C_Slider, c_Height, fmt7info.height, "height", _ranges[3],
		     CA_None, 0, 3, 1, 1, 0};

    _ranges[4][0] = fmt7info.unitBytePerPacket;
    _ranges[4][1] = fmt7info.maxBytePerPacket;
    _ranges[4][2] = fmt7info.unitBytePerPacket;
    _cmds[4]	  = {C_Slider, c_PacketSize, fmt7info.bytePerPacket,
		     "packet size", _ranges[4], CA_None, 0, 4, 1, 1, 0};

    _cmds[5]	  = {C_ChoiceMenuButton, c_PixelFormat, 0, "pixel format",
		     _pixelFormatMenus, CA_None, 0, 5, 1, 1, 0};
    _cmds[6]	  = {C_Button, c_OK, 0, "OK", noProp, CA_None, 0, 6, 1, 1, 0},
    _cmds[7]	  = EndOfCmds;

  // Create a menu button for setting pixel format.
    size_t	npixelformats = 0;
    for (const auto& pixelFormat : IIDCCamera::pixelFormatNames)
	if (fmt7info.availablePixelFormats & pixelFormat.pixelFormat)
	{
	    _pixelFormatMenus[npixelformats].label = pixelFormat.name;
	    _pixelFormatMenus[npixelformats].id	   = pixelFormat.pixelFormat;
	    _pixelFormatMenus[npixelformats].checked
		= (fmt7info.pixelFormat == pixelFormat.pixelFormat);
	    _pixelFormatMenus[npixelformats].submenu = noSub;
	    ++npixelformats;
	}
    _pixelFormatMenus[npixelformats].label = nullptr;
    
    return _cmds;
}

/************************************************************************
*  global functions							*
************************************************************************/
IIDCCamera::PixelFormat
selectROI(IIDCCamera& camera, IIDCCamera::Format format,
	  size_t& u0, size_t& v0, size_t& width, size_t& height,
	  size_t& packetSize, Window& window)
{
    switch (format)
    {
      case IIDCCamera::Format_7_0:
      case IIDCCamera::Format_7_1:
      case IIDCCamera::Format_7_2:
      case IIDCCamera::Format_7_3:
      case IIDCCamera::Format_7_4:
      case IIDCCamera::Format_7_5:
      case IIDCCamera::Format_7_6:
      case IIDCCamera::Format_7_7:
      {
	  IIDCModalDialog	modalDialog(window,
					    camera.getFormat_7_Info(format));
	  return modalDialog.selectROI(u0, v0, width, height, packetSize);
      }

      default:
	u0 = v0 = 0;
	width  = camera.width();
	height = camera.height();
	packetSize = 0;
	break;
    }
    
    return camera.pixelFormat();
}

}
}
