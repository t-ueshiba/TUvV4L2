/*
 *  $Id$
 */
#ifndef __TU_V_VIEEE1394PP_H
#define __TU_V_VIEEE1394PP_H

#include "TU/v/CmdPane.h"
#include "TU/Ieee1394++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*	createFormatMenu(const Ieee1394Camera& camera)		;
bool		handleCameraSpecialFormats(Ieee1394Camera& camera,
					   u_int id, int val,
					   Window& window)		;
bool		handleCameraSpecialFormats(
		    const Array<Ieee1394Camera*>& cameras,
		    u_int id, int val, Window& window)			;
CmdDef*		createFeatureCmds(const Ieee1394Camera& camera)		;
CmdDef*		createFeatureCmds(const Array<Ieee1394Camera*>& cameras);
void		refreshFeatureCmds(const Ieee1394Camera& camera,
				   CmdPane& cmdPane)			;
void		refreshFeatureCmds(const Array<Ieee1394Camera*>& cameras,
				   CmdPane& cmdPane)			;
bool		handleCameraFeatures(
		    const Array<Ieee1394Camera*>& cameras,
		    u_int id, int val, CmdPane& cmdPane)		;

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_IEEE1394PP_H
