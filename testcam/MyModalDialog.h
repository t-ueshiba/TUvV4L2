/*
 *  $Id: MyModalDialog.h,v 1.1 2009-07-28 00:00:48 ueshiba Exp $
 */
#include "TU/V4L2++.h"
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
  public:
    MyModalDialog(Window& parentWindow, const V4L2Camera& camera)	;
    
    void		getROI(size_t& u0, size_t& v0,
			       size_t& width, size_t& height)		;
    virtual void	callback(CmdId id, CmdVal val)			;
};
    
}
}
