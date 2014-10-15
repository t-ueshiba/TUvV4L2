/*
 *  $Id$
 */
#ifndef __TU_V_VIEEE1394PP_H
#define __TU_V_VIEEE1394PP_H

#include "TU/v/CmdPane.h"
#include "TU/Ieee1394CameraArray.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*	createFormatMenu(const Ieee1394Camera& camera)		;
bool		handleCameraSpecialFormat(Ieee1394Camera& camera,
					  u_int id, u_int val,
					  Window& window)		;
CmdDef*		createFeatureCmds(const Ieee1394Camera& camera)		;
CmdDef*		createFeatureCmds(const Ieee1394CameraArray& cameras)	;
void		refreshFeatureCmds(const Ieee1394Camera& camera,
				   CmdPane& cmdPane)			;

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_IEEE1394PP_H
