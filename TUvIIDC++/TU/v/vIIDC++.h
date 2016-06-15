/*
 *  $Id$
 */
#ifndef __TU_V_VIIDCPP_H
#define __TU_V_VIIDCPP_H

#include "TU/v/CmdPane.h"
#include "TU/IIDC++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
MenuDef*	createFormatMenu(const IIDCCamera& camera)		;
bool		setSpecialFormat(IIDCCamera& camera,
				 u_int id, int val, Window& window)	;
bool		setSpecialFormat(const Array<IIDCCamera*>& cameras,
				 u_int id, int val, Window& window)	;
CmdDef*		createFeatureCmds(const IIDCCamera& camera)		;
CmdDef*		createFeatureCmds(const Array<IIDCCamera*>& cameras);
void		refreshFeatureCmds(const IIDCCamera& camera,
				   CmdPane& cmdPane)			;
void		refreshFeatureCmds(const Array<IIDCCamera*>& cameras,
				   CmdPane& cmdPane)			;
bool		setFeatureValue(const Array<IIDCCamera*>& cameras,
				u_int id, int val, CmdPane& cmdPane)	;

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_IIDCPP_H
