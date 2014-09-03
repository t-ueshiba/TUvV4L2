/*
 *  平成14-19年（独）産業技術総合研究所 著作権所有
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
 *  Copyright 2002-2007.
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
 *  $Id: MyModalDialog.h 1495 2014-02-27 15:07:51Z ueshiba $
 */
#include "TU/Ieee1394++.h"
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
    MyModalDialog(Window& parentWindow,
		  const Ieee1394Camera::Format_7_Info& fmt7info)	;
    
    Ieee1394Camera::PixelFormat	getROI(size_t& u0, size_t& v0,
				       size_t& width, size_t& height)	;
    virtual void		callback(CmdId id, CmdVal val)		;

  private:
    const Ieee1394Camera::Format_7_Info&	_fmt7info;
};
    
}
}
