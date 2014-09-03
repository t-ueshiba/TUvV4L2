/*
 *  $Id: MyModalDialog.cc,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include "MyModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  static data								*
************************************************************************/
enum	{c_U0, c_V0, c_Width, c_Height, c_OK};

static CmdDef*
createROICmds(const V4L2Camera& camera)
{
    size_t	minU0, minV0, maxWidth, maxHeight;
    camera.getROILimits(minU0, minV0, maxWidth, maxHeight);
    size_t	u0, v0, width, height;
    camera.getROI(u0, v0, width, height);
    
    static int		prop[4][3];
    static CmdDef	cmds[] =
    {
	{C_Slider, c_U0,     int(u0),     "    u0", prop[0], CA_None,
	 0, 0, 1, 1, 0},
	{C_Slider, c_V0,     int(v0),     "    v0", prop[1], CA_None,
	 0, 1, 1, 1, 0},
	{C_Slider, c_Width,  int(width),  " width", prop[2], CA_None,
	 0, 2, 1, 1, 0},
	{C_Slider, c_Height, int(height), "height", prop[3], CA_None,
	 0, 3, 1, 1, 0},
	{C_Button, c_OK,     0,		      "OK", noProp,  CA_None,
	 0, 4, 1, 1, 0},
	EndOfCmds
    };

  // Create commands for setting ROI.
    cmds[0].val = u0;
    prop[0][0]  = minU0;
    prop[0][1]  = maxWidth - minU0 - 1;
    prop[0][2]  = 1;

    cmds[1].val = v0;
    prop[1][0]  = minV0;
    prop[1][1]  = maxHeight - minV0 - 1;
    prop[1][2]  = 1;

    cmds[2].val = width;
    prop[2][0]  = 0;
    prop[2][1]  = maxWidth;
    prop[2][2]  = 1;

    cmds[3].val = height;
    prop[3][0]  = 0;
    prop[3][1]  = maxHeight;
    prop[3][2]  = 1;
    
    return cmds;
}

/************************************************************************
*  class MyModalDialog							*
************************************************************************/
MyModalDialog::MyModalDialog(Window& parentWindow, const V4L2Camera& camera)
    :ModalDialog(parentWindow, "ROI for V4L2 camera", createROICmds(camera))
{
}
    
void
MyModalDialog::getROI(size_t& u0, size_t& v0, size_t& width, size_t& height)
{
    show();
    u0	   = pane().getValue(c_U0);
    v0	   = pane().getValue(c_V0);
    width  = pane().getValue(c_Width);
    height = pane().getValue(c_Height);
}

void
MyModalDialog::callback(CmdId id, CmdVal val)
{
    switch (id)
    {
      case c_OK:
	hide();
	break;
    }
}

}
}
