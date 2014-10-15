/*
 *  $Id$
 */
#ifndef __TU_V_VV4L2PP_H
#define __TU_V_VV4L2PP_H

#include "TU/v/CmdPane.h"
#include "TU/V4L2CameraArray.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*	createFormatMenu(const V4L2Camera& camera)		;
bool		handleCameraSpecialFormat(V4L2Camera& camera,
					  u_int id, u_int val,
					  Window& window)		;
CmdDef*		createFeatureCmds(const V4L2Camera& camera)		;
CmdDef*		createFeatureCmds(const V4L2CameraArray& cameras)	;
void		refreshFeatureCmds(const V4L2Camera& camera,
				   CmdPane& cmdPane)			;

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_V4L2PP_H
