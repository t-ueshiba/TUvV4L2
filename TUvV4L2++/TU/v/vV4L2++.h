/*
 *  $Id$
 */
#ifndef __TU_V_VV4L2PP_H
#define __TU_V_VV4L2PP_H

#include "TU/v/CmdPane.h"
#include "TU/V4L2++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*	createFormatMenu(const V4L2Camera& camera)		;
bool		handleCameraSpecialFormats(V4L2Camera& camera, u_int id,
					   int val, Window& window)	;
bool		handleCameraSpecialFormats(
		    const Array<V4L2Camera*>& cameras,
		    u_int id, int val, Window& window)			;
CmdDef*		createFeatureCmds(const V4L2Camera& camera)		;
CmdDef*		createFeatureCmds(const Array<V4L2Camera*>& cameras)	;
void		refreshFeatureCmds(const V4L2Camera& camera,
				   CmdPane& cmdPane)			;
bool		handleCameraFeatures(
		    const Array<V4L2Camera*>& cameras,
		    u_int id, int val, CmdPane& cmdPane)		;

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_V4L2PP_H
