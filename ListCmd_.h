/*
 *  平成9-19年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．創作者によ
 *  る許可なしに本プログラムを使用，複製，改変，第三者へ開示する等の著
 *  作権を侵害する行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the creator are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holders or the creator are not responsible for any
 *  damages in the use of this program.
 *  
 *  $Id: ListCmd_.h,v 1.4 2007-11-29 07:06:07 ueshiba Exp $
 */
#ifndef __TUvListCmd_h
#define __TUvListCmd_h

#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  class ListCmd							*
************************************************************************/
class ListCmd : public Cmd
{
  public:
    ListCmd(Object& parentObject, const CmdDef& cmd)		;
    virtual ~ListCmd()						;

    virtual const Widget&	widget()		const	;

    virtual CmdVal		getValue()		const	;
    virtual void		setValue(CmdVal val)		;
    virtual void		setProp(void* prop)		;
    void			setPercent(float percent)	;
    void			scroll(int n)			;
    
  private:
    const Widget	_widget;			// viewportWidget
    const Widget	_list;				// listWidget
    int			_top;
    u_int		_nitems;
    const u_int		_nitemsShown;
};

}
}
#endif	// !__TUvListCmd_h
