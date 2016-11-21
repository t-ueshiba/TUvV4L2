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
static struct
{
    IIDCCamera::Feature	feature;
    const char*		name;
    float		prop[3];
} features[] =
{
    {IIDCCamera::TRIGGER_MODE,	"Trigger mode"	 },
    {IIDCCamera::BRIGHTNESS,	"Brightness"	 },
    {IIDCCamera::AUTO_EXPOSURE,	"Auto exposure"	 },
    {IIDCCamera::SHARPNESS,	"Sharpness"	 },
    {IIDCCamera::WHITE_BALANCE,	"White bal.(U/B)"},
    {IIDCCamera::HUE,		"Hue"		 },
    {IIDCCamera::SATURATION,	"Saturation"	 },
    {IIDCCamera::GAMMA,		"Gamma"		 },
    {IIDCCamera::SHUTTER,	"Shutter"	 },
    {IIDCCamera::GAIN,		"Gain"		 },
    {IIDCCamera::IRIS,		"Iris"		 },
    {IIDCCamera::FOCUS,		"Focus"		 },
    {IIDCCamera::TEMPERATURE,	"Temperature"	 },
    {IIDCCamera::TRIGGER_DELAY,	"Trigger delay"	 },
    {IIDCCamera::FRAME_RATE,	"Frame rate"	 },
    {IIDCCamera::ZOOM,		"Zoom"		 },
    {IIDCCamera::PAN,		"Pan"		 },
    {IIDCCamera::TILT,		"Tilt"		 },
};
static constexpr size_t		NFEATURES = sizeof(features)
					  / sizeof(features[0]);
static CmdDef			featureCmds[3*NFEATURES + 2];
static MenuDef			triggerModeMenus[IIDCCamera::NTRIGGERMODES];
static Array<CmdDef>		cameraChoiceCmds;
static Array<std::string>	cameraChoiceTitles;

/************************************************************************
*  static functions							*
************************************************************************/
static void
setSlider(const IIDCCamera& camera, IIDCCamera::Feature feature, CmdDef* cmd)
{
    const auto	fp = static_cast<float*>(cmd[0].prop);

    if (cmd->id == IIDCCamera::WHITE_BALANCE)
    {
	if (camera.isAbsControl(IIDCCamera::WHITE_BALANCE))
	{
	    float	min, max;
	    camera.getMinMax(IIDCCamera::WHITE_BALANCE, min, max);
	    fp[0] = min;
	    fp[1] = max;
	    fp[2] = 0;
	    
	    float	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmd[0].val = ub;
	    cmd[1].val = vr;
	}
	else
	{
	    u_int	min, max;
	    camera.getMinMax(IIDCCamera::WHITE_BALANCE, min, max);
	    fp[0] = min;
	    fp[1] = max;
	    fp[2] = 1;
	    
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmd[0].val = ub;
	    cmd[1].val = vr;
	}
    }
    else
    {
	const auto	feature = IIDCCamera::uintToFeature(cmd->id);
	  
	if (camera.isAbsControl(feature))
	{
	    float	min, max;
	    camera.getMinMax(feature, min, max);
	    fp[0] = min;
	    fp[1] = max;
	    fp[2] = 0;

	    const auto	val = camera.getValue<float>(feature);
	    cmd->val = val;
	}
	else
	{
	    u_int	min, max;
	    camera.getMinMax(feature, min, max);
	    fp[0] = min;
	    fp[1] = max;
	    fp[2] = 1;

	    const auto	val = camera.getValue<u_int>(feature);
	    cmd->val = val;
	}
    }
    
}

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createFeatureCmds(const IIDCCamera& camera, size_t ncameras)
{
    u_int	y     = 0;
    size_t	ncmds = 0;
    
    if (ncameras > 1)
    {
	cameraChoiceCmds  .resize(ncameras + 2);
	cameraChoiceTitles.resize(ncameras + 1);
	
	size_t	i = 0;
	for (; i < ncameras; ++i)
	{
	    (cameraChoiceTitles[i] += "cam-") += ('0' + i);
	    
	    cameraChoiceCmds[i].type		= C_RadioButton;
	    cameraChoiceCmds[i].id		= i;
	    cameraChoiceCmds[i].val		= 0;
	    cameraChoiceCmds[i].title		= cameraChoiceTitles[i].c_str();
	    cameraChoiceCmds[i].prop		= noProp;
	    cameraChoiceCmds[i].attrs		= CA_None;
	    cameraChoiceCmds[i].gridx		= i;
	    cameraChoiceCmds[i].gridy		= 0;
	    cameraChoiceCmds[i].gridWidth	= 1;
	    cameraChoiceCmds[i].gridHeight	= 1;
	    cameraChoiceCmds[i].size		= 0;
	}
	cameraChoiceTitles[i] = "All";
	cameraChoiceCmds[i].type	= C_RadioButton;
	cameraChoiceCmds[i].id		= i;
	cameraChoiceCmds[i].val		= 0;
	cameraChoiceCmds[i].title	= cameraChoiceTitles[i].c_str();
	cameraChoiceCmds[i].prop	= noProp;
	cameraChoiceCmds[i].attrs	= CA_None;
	cameraChoiceCmds[i].gridx	= i;
	cameraChoiceCmds[i].gridy	= 0;
	cameraChoiceCmds[i].gridWidth	= 1;
	cameraChoiceCmds[i].gridHeight	= 1;
	cameraChoiceCmds[i].size	= 0;

	cameraChoiceCmds[++i].type = C_EndOfList;
	
	featureCmds[ncmds].type		= C_ChoiceFrame;
	featureCmds[ncmds].id		= IIDCCAMERA_CHOICE;
	featureCmds[ncmds].val		= int(ncameras);
	featureCmds[ncmds].title	= 0;
	featureCmds[ncmds].prop		= cameraChoiceCmds.data();
	featureCmds[ncmds].attrs	= CA_None;
	featureCmds[ncmds].gridx	= 0;
	featureCmds[ncmds].gridy	= y;
	featureCmds[ncmds].gridWidth	= 1;
	featureCmds[ncmds].gridHeight	= 1;
	featureCmds[ncmds].size		= 0;

	++ncmds;
	++y;
    }

    for (auto& feature : features)
    {
	const auto	inq = camera.inquireFeatureFunction(feature.feature);
	
	if (!((inq & IIDCCamera::Presence) &&
	      (inq & IIDCCamera::Manual)   &&
	      (inq & IIDCCamera::ReadOut)))
	    continue;
	
	featureCmds[ncmds].id	      = feature.feature;
	featureCmds[ncmds].title      = feature.name;
	featureCmds[ncmds].prop	      = feature.prop;
	featureCmds[ncmds].attrs      = CA_None;
	featureCmds[ncmds].gridx      = 0;
	featureCmds[ncmds].gridy      = y;
	featureCmds[ncmds].gridWidth  = 1;
	featureCmds[ncmds].gridHeight = 1;
	featureCmds[ncmds].size	      = 0;

	switch (feature.feature)
	{
	  case IIDCCamera::TRIGGER_MODE:
	  {
	    featureCmds[ncmds].type = C_ChoiceMenuButton;
	    featureCmds[ncmds].prop = triggerModeMenus;

	    size_t	nmodes = 0;
	    for (const auto& triggerMode : IIDCCamera::triggerModeNames)
		if (inq & triggerMode.mode)
		{
		    triggerModeMenus[nmodes].label = triggerMode.name;
		    triggerModeMenus[nmodes].id	   = triggerMode.mode;
		    triggerModeMenus[nmodes].checked
			= (camera.getTriggerMode() == triggerMode.mode);
		    triggerModeMenus[nmodes].submenu = noSub;
		    ++nmodes;
		}
			
	    triggerModeMenus[nmodes] = EndOfMenu;
	  }
	    break;

	  case IIDCCamera::WHITE_BALANCE:
	  // Create sliders for setting values.
	    featureCmds[ncmds].type = C_Slider;
	    setSlider(camera, feature.feature, featureCmds + ncmds);
	    
	    ++ncmds;
	    ++y;
	    featureCmds[ncmds].type	  = C_Slider;
	    featureCmds[ncmds].id	  = feature.feature + IIDCCAMERA_OFFSET_VR;
	    featureCmds[ncmds].title      = "White bal.(V/R)";
	    featureCmds[ncmds].prop       = feature.prop;
	    featureCmds[ncmds].attrs      = CA_None;
	    featureCmds[ncmds].gridx      = 0;
	    featureCmds[ncmds].gridy      = y;
	    featureCmds[ncmds].gridWidth  = 1;
	    featureCmds[ncmds].gridHeight = 1;
	    featureCmds[ncmds].size	  = 0;
	    break;

	  default:
	    featureCmds[ncmds].type = C_Slider;
	    setSlider(camera, feature.feature, featureCmds + ncmds);
	    break;
	}

	++ncmds;

	if (inq & IIDCCamera::OnOff)
	{
	  // Create toggle button for turning on/off this feature.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature + IIDCCAMERA_OFFSET_ONOFF;
	    featureCmds[ncmds].val	  = camera.isActive(feature.feature);
	    featureCmds[ncmds].title      = "On";
	    featureCmds[ncmds].prop       = noProp;
	    featureCmds[ncmds].attrs      = CA_None;
	    featureCmds[ncmds].gridx      = 1;
	    featureCmds[ncmds].gridy      = y;
	    featureCmds[ncmds].gridWidth  = 1;
	    featureCmds[ncmds].gridHeight = 1;
	    featureCmds[ncmds].size	  = 0;
	    ++ncmds;
	}

	if (inq & IIDCCamera::Auto)
	{
	  // Create toggle button for setting manual/auto mode.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature + IIDCCAMERA_OFFSET_AUTO;
	    if (feature.feature == IIDCCamera::TRIGGER_MODE)
	    {
		featureCmds[ncmds].val	  = camera.getTriggerPolarity();
		featureCmds[ncmds].title  = "(+)";
	    }
	    else
	    {
		featureCmds[ncmds].val	  = camera.isAuto(feature.feature);
		featureCmds[ncmds].title  = "Auto";
	    }
	    featureCmds[ncmds].prop       = noProp;
	    featureCmds[ncmds].attrs      = CA_None;
	    featureCmds[ncmds].gridx      = 2;
	    featureCmds[ncmds].gridy      = y;
	    featureCmds[ncmds].gridWidth  = 1;
	    featureCmds[ncmds].gridHeight = 1;
	    featureCmds[ncmds].size	  = 0;
	    ++ncmds;
	}

	if (inq & IIDCCamera::Abs_Control)
	{
		// Create toggle button for turning on/off abs. value mode.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.feature + IIDCCAMERA_OFFSET_ABS;
	    featureCmds[ncmds].val	  = camera.isAbsControl(feature.feature);
	    featureCmds[ncmds].title      = "Abs.";
	    featureCmds[ncmds].prop       = noProp;
	    featureCmds[ncmds].attrs      = CA_None;
	    featureCmds[ncmds].gridx      = 3;
	    featureCmds[ncmds].gridy      = y;
	    featureCmds[ncmds].gridWidth  = 1;
	    featureCmds[ncmds].gridHeight = 1;
	    featureCmds[ncmds].size	  = 0;
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

	if (id >= IIDCCamera::BRIGHTNESS + IIDCCAMERA_OFFSET_ABS)
	    cmdPane.setValue(id,
			     int(camera.isAbsControl(
				     IIDCCamera::uintToFeature(
					 id - IIDCCAMERA_OFFSET_ABS))));
	else if (id >= IIDCCamera::BRIGHTNESS + IIDCCAMERA_OFFSET_AUTO)
	    cmdPane.setValue(id,
			     int(camera.isAuto(
				     IIDCCamera::uintToFeature(
					 id - IIDCCAMERA_OFFSET_AUTO))));
	else if (id >= IIDCCamera::BRIGHTNESS + IIDCCAMERA_OFFSET_ONOFF)
	    cmdPane.setValue(id,
			     int(camera.isActive(
				     IIDCCamera::uintToFeature(
					 id - IIDCCAMERA_OFFSET_ONOFF))));
	else if (id == IIDCCAMERA_CHOICE)
	    continue;
	else if (id == IIDCCamera::TRIGGER_MODE)
	    cmdPane.setValue(id, int(camera.getTriggerMode()));
	else if (id == IIDCCamera::WHITE_BALANCE)
	{
	    if (camera.isAbsControl(IIDCCamera::WHITE_BALANCE))
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
	    const auto	feature = IIDCCamera::uintToFeature(id);

	    if (camera.isAbsControl(feature))
		cmdPane.setValue(id, camera.getValue<float>(feature));
	    else
		cmdPane.setValue(id, camera.getValue<u_int>(feature));
	}
    }
}

bool
setFeatureCmds(IIDCCamera& camera, CmdId id, CmdVal val, CmdPane& cmdPane)
{
    if (setFeature(&camera, &camera + 1, id, int(val), val.f()))
    {
	if (id >= IIDCCamera::BRIGHTNESS + IIDCCAMERA_OFFSET_ABS)
	{
	    float	props[3];
	    
	    if (id == IIDCCamera::WHITE_BALANCE)
	    {
		if (camera.isAbsControl(IIDCCamera::WHITE_BALANCE))
		{
		    float	min, max;
		    camera.getMinMax(IIDCCamera::WHITE_BALANCE, min, max);
		    props[0] = min;
		    props[1] = max;
		    props[2] = 0;
		    cmdPane.setProp(id, props);
		    cmdPane.setProp(id + IIDCCAMERA_OFFSET_VR, props);
		    
		    float	ub, vr;
		    camera.getWhiteBalance(ub, vr);
		    cmdPane.setValue(id, ub);
		    cmdPane.setValue(id + IIDCCAMERA_OFFSET_VR, ub);
		}
		else
		{
		    u_int	min, max;
		    camera.getMinMax(IIDCCamera::WHITE_BALANCE, min, max);
		    props[0] = min;
		    props[1] = max;
		    props[2] = 0;
		    cmdPane.setProp(id, props);
		    cmdPane.setProp(id + IIDCCAMERA_OFFSET_VR, props);
		    
		    u_int	ub, vr;
		    camera.getWhiteBalance(ub, vr);
		    cmdPane.setValue(id, ub);
		    cmdPane.setValue(id + IIDCCAMERA_OFFSET_VR, ub);
		}
	    }
	    else
	    {
		const auto	feature = IIDCCamera::uintToFeature(id);
	  
		if (camera.isAbsControl(feature))
		{
		    float	min, max;
		    camera.getMinMax(feature, min, max);
		    props[0] = min;
		    props[1] = max;
		    props[2] = 0;
		    cmdPane.setProp(id, props);
		    cmdPane.setValue(feature, camera.getValue<float>(feature));
		}
		else
		{
		    u_int	min, max;
		    camera.getMinMax(feature, min, max);
		    props[0] = min;
		    props[1] = max;
		    props[2] = 0;
		    cmdPane.setProp(id, props);
		    cmdPane.setValue(feature, camera.getValue<u_int>(feature));
		}
	    }
	}
	
	return true;
    }
    else
	return false;
}
    
}	// namespace v
}	// namespace TU
