/*
 *  $Id: SliderCmd_.h,v 1.2 2002-07-25 02:38:12 ueshiba Exp $
 */
#ifndef __TUvSliderCmd_h
#define __TUvSliderCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class SliderCmd							*
************************************************************************/
class SliderCmd : public Cmd
{
  public:
    SliderCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~SliderCmd()					;

    virtual const Widget&	widget()		const	;

    virtual CmdVal		getValue()		const	;
    virtual void		setValue(CmdVal val)		;
    virtual void		setProp(void* prop)		;
    void			setPercent(float percent)	;
    
  private:
    void			setValueInternal(CmdVal val)	;

    const Widget	_widget;			// gridboxWidget
    const Widget	_title;				// labelWidget
    const Widget	_slider;			// slider3dWidget
    const Widget	_text;				// labelWidget
    int			_min;
    u_int		_range;
    u_int		_div;
    int			_val;
};

}
}
#endif	// !__TUvSliderCmd_h
