/*
 *  $Id$
 */
#include "TU/v/vV4L2++.h"
#include "TU/v/ModalDialog.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class MyModalDialog							*
************************************************************************/
class MyModalDialog : public ModalDialog
{
  private:
    enum	{c_U0, c_V0, c_Width, c_Height, c_OK};
    
  public:
    MyModalDialog(Window& parentWindow, const V4L2Camera& camera)	;
    
    void		getROI(size_t& u0, size_t& v0,
			       size_t& width, size_t& height)		;
    virtual void	callback(CmdId id, CmdVal val)			;

  private:
    CmdDef*		createROICmds(const V4L2Camera& camera)		;

  private:
    int		_props[4][3];
    CmdDef	_cmds[6];
};
    
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

CmdDef*
MyModalDialog::createROICmds(const V4L2Camera& camera)
{
    size_t	minU0, minV0, maxWidth, maxHeight;
    camera.getROILimits(minU0, minV0, maxWidth, maxHeight);
    size_t	u0, v0, width, height;
    camera.getROI(u0, v0, width, height);
    
  // Create commands for setting ROI.
    _props[0][0] = minU0;
    _props[0][1] = maxWidth - minU0 - 1;
    _props[0][2] = 1;
    _cmds[0]	 = {C_Slider, c_U0,	int(u0),     "    u0", _props[0],
		    CA_None, 0, 0, 1, 1, 0};

    _props[1][0] = minV0;
    _props[1][1] = maxHeight - minV0 - 1;
    _props[1][2] = 1;
    _cmds[1]	 = {C_Slider, c_V0,	int(v0),     "    v0", _props[1],
		    CA_None, 0, 1, 1, 1, 0};

    _props[2][0] = 0;
    _props[2][1] = maxWidth;
    _props[2][2] = 1;
    _cmds[2]	 = {C_Slider, c_Width,	int(width),  " width", _props[2],
		    CA_None, 0, 2, 1, 1, 0};
    
    _props[3][0] = 0;
    _props[3][1] = maxHeight;
    _props[3][2] = 1;
    _cmds[3]	 = {C_Slider, c_Height,	int(height), "height", _props[3],
		    CA_None, 0, 3, 1, 1, 0};

    _cmds[4]	 = {C_Button, c_OK, 0, "OK", noProp,
		    CA_None, 0, 4, 1, 1, 0};
    _cmds[5]	 = EndOfCmds;
	
    return _cmds;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
setCameraSpecialFormat(V4L2Camera& camera, u_int id, int val, Window& window)
{
    if (id == V4L2Camera::UNKNOWN_PIXEL_FORMAT)
    {
	size_t	u0, v0, width, height;
	if (camera.getROI(u0, v0, width, height))
	{
	    MyModalDialog	modalDialog(window, camera);
	    modalDialog.getROI(u0, v0, width, height);
	    camera.setROI(u0, v0, width, height);
	}

	return true;
    }

    return false;
}

bool
setCameraSpecialFormat(const Array<V4L2Camera*>& cameras,
		       u_int id, int val, Window& window)
{
    if (id == V4L2Camera::UNKNOWN_PIXEL_FORMAT)
    {
	size_t	u0, v0, width, height;
	if (cameras[0]->getROI(u0, v0, width, height))
	{
	    MyModalDialog	modalDialog(window, *cameras[0]);
	    modalDialog.getROI(u0, v0, width, height);

	    for (size_t i = 0; i < cameras.size(); ++i)
		cameras[i]->setROI(u0, v0, width, height);
	}

	return true;
    }

    return false;
}

}
}
