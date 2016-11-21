/*
 *  $Id$
 */
#ifndef __TU_V_VIIDCPP_H
#define __TU_V_VIIDCPP_H

#include "TU/v/CmdPane.h"
#include "TU/IIDCCameraUtility.h"
#include "TU/algorithm.h"

namespace TU
{
namespace v
{
/************************************************************************
*  global functions							*
************************************************************************/
/*
 *  functions for a single camera
 */
MenuDef*	createFormatMenu(const IIDCCamera& camera)			;
bool		setSpecialFormat(IIDCCamera& camera,
				 CmdId id, CmdVal val, Window& window)		;
CmdDef*		createFeatureCmds(const IIDCCamera& camera, size_t ncameras=1)	;
void		refreshFeatureCmds(const IIDCCamera& camera, CmdPane& cmdPane)	;
bool		setFeatureCmds(const IIDCCamera& camera,
			       CmdId id, CmdVal val, CmdPane& cmdPane)		;
    
/*
 *  functions for multiple cameras
 */
#if 0
template <class CAMERAS>
typename std::enable_if<is_range<CAMERAS>::value, bool>::type
setSpecialFormat(const CAMERAS& cameras, CmdId id, CmdVal val, Window& window)
{
    switch (id)
    {
      case IIDCCamera::Format_7_0:
      case IIDCCamera::Format_7_1:
      case IIDCCamera::Format_7_2:
      case IIDCCamera::Format_7_3:
      case IIDCCamera::Format_7_4:
      case IIDCCamera::Format_7_5:
      case IIDCCamera::Format_7_6:
      case IIDCCamera::Format_7_7:
      {
	auto	format7 = IIDCCamera::uintToFormat(id);
	v::MyModalDialog
		modalDialog(window,
			    std::begin(cameras)->getFormat_7_Info(format7));
	u_int	u0, v0, width, height;
	auto	pixelFormat = modalDialog.getROI(u0, v0, width, height);
	for (auto& camera : cameras)
	    camera.setFormat_7_ROI(format7, u0, v0, width, height)
		  .setFormat_7_PixelFormat(format7, pixelFormat)
		  .setFormatAndFrameRate(format7,
					 IIDCCamera::uintToFrameRate(val));
      }
	return true;
    }
    
    return false;
}
#endif
template <class CAMERAS>
inline typename std::enable_if<is_range<CAMERAS>::value, CmdDef*>::type
createFeatureCmds(const CAMERAS& cameras)
{
    const auto	ncameras = size(cameras);

    if (ncameras == 0)
	return nullptr;
    
    return createFeatureCmds(*cameras.begin(), ncameras);
}

template <class CAMERAS>
typename std::enable_if<is_range<CAMERAS>::value>::type
refreshFeatureCmds(const CAMERAS& cameras, CmdPane& cmdPane)
{
    const auto	ncameras = size(cameras);
    
    if (ncameras == 0)
	return;

    int	i = cmdPane.getValue(IIDCCAMERA_CHOICE);
    if (i < 0 || i >= ncameras)
	i = 0;

    auto	camera = std::begin(cameras);
    std::advance(camera, i);
    refreshFeatureCmds(*camera, cmdPane);
}

template <class CAMERAS>
typename std::enable_if<is_range<CAMERAS>::value, bool>::type
setFeatureCmds(CAMERAS& cameras, CmdId id, CmdVal val, CmdPane& cmdPane)
{
    const auto	ncameras = size(cameras);
    
    if (ncameras == 0)
	return false;

    if (id == IIDCCAMERA_CHOICE)
    {
	const auto	i = (0 <= int(val) && int(val) < ncameras ? int(val) : 0);
	auto		camera = std::begin(cameras);
	std::advance(camera, i);
	refreshFeatureCmds(*camera, cmdPane);
    }
    else
    {
	const int	i     = cmdPane.getValue(IIDCCAMERA_CHOICE);
	auto		begin = std::begin(cameras);
	auto		end   = std::end(cameras);
	if (0 <= i && i < ncameras)
	{
	    std::advance(begin, i);
	    (end = begin)++;
	}

	setFeature(begin, end, id, int(val), float(val));
    }

    return true;
}

}	// namespace v
}	// namespace TU
#endif	// !__TU_V_IIDCPP_H
