/*
 *  $Id: featureCmds.cc,v 1.2 2012-06-20 07:50:08 ueshiba Exp $
 */
#include "TU/v/vIIDC++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static CmdDef	featureCmds[2 + 4*IIDCCamera::NFEATURES + 1];
    
/************************************************************************
*  static functions							*
************************************************************************/
static void
getSliderRange(const IIDCCamera& camera,
	       IIDCCamera::Feature feature, float range[])
{
    if (camera.isAbsControl(feature))
    {
	camera.getMinMax(feature, range[0], range[1]);
	range[2] = 0;
    }
    else
    {
	u_int	imin, imax;
	camera.getMinMax(feature, imin, imax);
	range[0] = imin;
	range[1] = imax;
	range[2] = 1;
    }
}

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createFeatureCmds(const IIDCCamera& camera, size_t ncameras)
{
    static float	ranges[IIDCCamera::NFEATURES][3];

    size_t	y       = 0;
    size_t	ncmds   = 0;
    size_t	nranges = 0;
    
    if (ncameras > 1)
    {
	static Array<CmdDef>		cameraChoiceCmds;
	static Array<std::string>	cameraChoiceTitles;
	
	cameraChoiceCmds  .resize(ncameras + 1);
	cameraChoiceTitles.resize(ncameras);
	
	for (size_t i = 0; i < ncameras; ++i)
	{
	    (cameraChoiceTitles[i] += "cam-") += ('0' + i);
	    
	    cameraChoiceCmds[i].type	= C_RadioButton;
	    cameraChoiceCmds[i].id	= i;
	    cameraChoiceCmds[i].val	= 0;
	    cameraChoiceCmds[i].title	= cameraChoiceTitles[i].c_str();
	    cameraChoiceCmds[i].gridx	= i;
	    cameraChoiceCmds[i].gridy	= 0;
	}
	cameraChoiceCmds[ncameras].type = C_EndOfList;
	
	featureCmds[ncmds].type		= C_ChoiceFrame;
	featureCmds[ncmds].id		= IIDCCAMERA_CHOICE;
	featureCmds[ncmds].val		= 0;	// 最初のカメラを選択
	featureCmds[ncmds].prop		= cameraChoiceCmds.data();
	featureCmds[ncmds].gridx	= 0;
	featureCmds[ncmds].gridy	= y;
	++ncmds;
	
	featureCmds[ncmds].type		= C_ToggleButton;
	featureCmds[ncmds].id		= IIDCCAMERA_ALL;
	featureCmds[ncmds].val		= 1;	// 全カメラ同時操作モード
	featureCmds[ncmds].title	= "All";
	featureCmds[ncmds].gridx	= 1;
	featureCmds[ncmds].gridy	= y;
	++ncmds;
	
	++y;
    }

    for (const auto& feature : IIDCCamera::featureNames)
    {
	const auto	inq = camera.inquireFeatureFunction(feature.feature);
	
	if (!((inq & IIDCCamera::Presence) &&
	      (inq & IIDCCamera::Manual)   &&
	      (inq & IIDCCamera::ReadOut)))
	    continue;
	
	featureCmds[ncmds].id		= feature.feature;
	featureCmds[ncmds].title	= feature.name;
	featureCmds[ncmds].gridx	= 0;
	featureCmds[ncmds].gridy	= y;

	switch (feature.feature)
	{
	  case IIDCCamera::TRIGGER_MODE:
	  {
	    static MenuDef	triggerModeMenus[IIDCCamera::NTRIGGERMODES];
	    
	    featureCmds[ncmds].type = C_ChoiceMenuButton;
	    featureCmds[ncmds].prop = triggerModeMenus;

	    size_t	nmodes = 0;
	    for (const auto& triggerMode : IIDCCamera::triggerModeNames)
		if (inq & triggerMode.triggerMode)
		{
		    triggerModeMenus[nmodes].label = triggerMode.name;
		    triggerModeMenus[nmodes].id	   = triggerMode.triggerMode;
		    triggerModeMenus[nmodes].checked
			= (camera.getTriggerMode() == triggerMode.triggerMode);
		    triggerModeMenus[nmodes].submenu = noSub;
		    ++nmodes;
		}
			
	    triggerModeMenus[nmodes] = EndOfMenu;
	  }
	    break;

	  case IIDCCamera::WHITE_BALANCE:
	  // Create sliders for setting values.
	    featureCmds[ncmds].type	= C_Slider;
	    featureCmds[ncmds].prop     = ranges[nranges];
	    getSliderRange(camera, feature.feature, ranges[nranges++]);
	    ++ncmds;
	    ++y;
	    
	    featureCmds[ncmds].type	= C_Slider;
	    featureCmds[ncmds].id	= feature.feature
					+ IIDCCAMERA_OFFSET_VR;
	    featureCmds[ncmds].title    = "White bal.(V/R)";
	    featureCmds[ncmds].prop     = ranges[nranges];
	    featureCmds[ncmds].gridx    = 0;
	    featureCmds[ncmds].gridy    = y;
	    getSliderRange(camera, feature.feature, ranges[nranges++]);
	    break;

	  default:
	    featureCmds[ncmds].type = C_Slider;
	    getSliderRange(camera, feature.feature, ranges[nranges++]);
	    break;
	}

	++ncmds;

	if (inq & IIDCCamera::OnOff)
	{
	  // Create toggle button for turning on/off this feature.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature
					  + IIDCCAMERA_OFFSET_ONOFF;
	    featureCmds[ncmds].title      = "On";
	    featureCmds[ncmds].gridx      = 1;
	    featureCmds[ncmds].gridy      = y;
	    ++ncmds;
	}

	if (inq & IIDCCamera::Auto)
	{
	  // Create toggle button for setting manual/auto mode.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature
					  + IIDCCAMERA_OFFSET_AUTO;
	    featureCmds[ncmds].title 
		= (feature.feature == IIDCCamera::TRIGGER_MODE ? "(+)" : "Auto");
	    featureCmds[ncmds].gridx      = 2;
	    featureCmds[ncmds].gridy      = y;
	    ++ncmds;
	}

	if (inq & IIDCCamera::Abs_Control)
	{
	  // Create toggle button for turning on/off abs. value mode.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature
					  + IIDCCAMERA_OFFSET_ABS;
	    featureCmds[ncmds].title      = "Abs.";
	    featureCmds[ncmds].gridx      = 3;
	    featureCmds[ncmds].gridy      = y;
	    ++ncmds;
	}	

	++y;
    }

    featureCmds[ncmds].type = C_EndOfList;

    return featureCmds;
}

void
refreshFeatureCmds(const IIDCCamera& camera, CmdPane& cmdPane)
{
    for (auto featureCmd = featureCmds;
	 featureCmd->type != C_EndOfList; ++featureCmd)
    {
	const auto	id = featureCmd->id;
	u_int		val;
	float		fval;
	
	if (getFeature(camera, id, val, fval))
	{
	    if ((id >= IIDCCamera::BRIGHTNESS + IIDCCAMERA_OFFSET_ONOFF) ||
		(id == IIDCCamera::TRIGGER_MODE))
		cmdPane.setValue(id, val);
	    else
		refreshSliderCmd(camera, id, cmdPane);
	}
    }
}
    
void
refreshSliderCmd(const IIDCCamera& camera, CmdId id, CmdPane& cmdPane)
{
    const auto	feature = IIDCCamera::uintToFeature(id);

    float	range[3];
    getSliderRange(camera, feature, range);
    cmdPane.setProp(id, range);
    
    if (id == IIDCCamera::WHITE_BALANCE)
    {
	cmdPane.setProp(id + IIDCCAMERA_OFFSET_VR, range);
	
	if (camera.isAbsControl(feature))
	{
	    float	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmdPane.setValue(id, ub);
	    cmdPane.setValue(id + IIDCCAMERA_OFFSET_VR, vr);
	}
	else
	{
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmdPane.setValue(id, ub);
	    cmdPane.setValue(id + IIDCCAMERA_OFFSET_VR, vr);
	}
    }
    else
    {
	if (camera.isAbsControl(feature))
	    cmdPane.setValue(id, camera.getValue<float>(feature));
	else
	    cmdPane.setValue(id, camera.getValue<u_int>(feature));
    }
}
    
}	// namespace v
}	// namespace TU
