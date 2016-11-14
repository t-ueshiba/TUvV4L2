/*
 *  $Id: createFeatureCmds.cc,v 1.2 2012-06-20 07:50:08 ueshiba Exp $
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
    IIDCCamera::Feature	id;
    const char*		name;
    int			prop[3];
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
  //{IIDCCamera::PAN,		"Pan"		 },
  //{IIDCCamera::TILT,		"Tilt"		 },
};
static constexpr size_t		NFEATURES = sizeof(features)
					  / sizeof(features[0]);
static CmdDef			featureCmds[3*NFEATURES + 2];
static Array<CmdDef>		cameraChoiceCmds;
static Array<std::string>	cameraChoiceTitles;
static constexpr CmdId		CAMERA_CHOICE = IIDCCamera::BRIGHTNESS + 2;
static MenuDef			triggerModeMenus[IIDCCamera::NTRIGGERMODES];
    
/************************************************************************
*  static functions							*
************************************************************************/
static CmdDef*
createFeatureCmds(const IIDCCamera& camera, size_t ncmds)
{
    u_int	y = ncmds;
    for (auto& feature : features)
    {
	const auto	inq = camera.inquireFeatureFunction(feature.id);
	
	if (!((inq & IIDCCamera::Presence) &&
	      (inq & IIDCCamera::Manual)   &&
	      (inq & IIDCCamera::ReadOut)))
	    continue;
	
	featureCmds[ncmds].id	      = feature.id;
	featureCmds[ncmds].title      = feature.name;
	featureCmds[ncmds].prop	      = feature.prop;
	featureCmds[ncmds].attrs      = CA_None;
	featureCmds[ncmds].gridx      = 0;
	featureCmds[ncmds].gridy      = y;
	featureCmds[ncmds].gridWidth  = 1;
	featureCmds[ncmds].gridHeight = 1;
	featureCmds[ncmds].size	      = 0;

	if (feature.id == IIDCCamera::TRIGGER_MODE)
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
	else
	{
	  // Create sliders for setting values.
	    featureCmds[ncmds].type = C_Slider;

	    u_int	min, max;
	    camera.getMinMax(feature.id, min, max);
	    feature.prop[0] = min;
	    feature.prop[1] = max - min;
	    feature.prop[2] = 1;
	    
	    if (feature.id == IIDCCamera::WHITE_BALANCE)
	    {
		++ncmds;
		++y;
		featureCmds[ncmds].type	      = C_Slider;
		featureCmds[ncmds].id	      = feature.id + IIDCCAMERA_OFFSET_VR;
		featureCmds[ncmds].title      = "White bal.(V/R)";
		featureCmds[ncmds].prop       = feature.prop;
		featureCmds[ncmds].attrs      = CA_None;
		featureCmds[ncmds].gridx      = 0;
		featureCmds[ncmds].gridy      = y;
		featureCmds[ncmds].gridWidth  = 1;
		featureCmds[ncmds].gridHeight = 1;
		featureCmds[ncmds].size	      = 0;
		u_int	ub, vr;
		camera.getWhiteBalance(ub, vr);
		featureCmds[ncmds-1].val = ub;
		featureCmds[ncmds  ].val = vr;
	    }
	    else
	    {
		featureCmds[ncmds].val = camera.getValue(feature.id);

		if (inq & IIDCCamera::Abs_Control)
		{
		    ++ncmds;

		  // Create toggle button for turning on/off abs. value mode.
		    featureCmds[ncmds].type	  = C_ToggleButton;
		    featureCmds[ncmds].id	  = feature.id
						  + IIDCCAMERA_OFFSET_ABS;
		    featureCmds[ncmds].val	  = camera.isAbsControl(
							feature.id);
		    featureCmds[ncmds].title      = "Abs.";
		    featureCmds[ncmds].prop       = noProp;
		    featureCmds[ncmds].attrs      = CA_None;
		    featureCmds[ncmds].gridx      = 3;
		    featureCmds[ncmds].gridy      = y;
		    featureCmds[ncmds].gridWidth  = 1;
		    featureCmds[ncmds].gridHeight = 1;
		    featureCmds[ncmds].size	  = 0;
		}
	    }
	}
		
	++ncmds;
	
	if (inq & IIDCCamera::OnOff)
	{
	  // Create toggle button for turning on/off this feature.
	    featureCmds[ncmds].type	  = C_ToggleButton;
	    featureCmds[ncmds].id	  = feature.id + IIDCCAMERA_OFFSET_ONOFF;
	    featureCmds[ncmds].val	  = camera.isActive(feature.id);
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
	    featureCmds[ncmds].id	  = feature.id
		+ IIDCCAMERA_OFFSET_AUTO;
	    if (feature.id == IIDCCamera::TRIGGER_MODE)
	    {
		featureCmds[ncmds].val	  = camera.getTriggerPolarity();
		featureCmds[ncmds].title  = "(+)";
	    }
	    else
	    {
		featureCmds[ncmds].val	  = camera.isAuto(feature.id);
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

	++y;
    }

    featureCmds[ncmds].type = C_EndOfList;

    return featureCmds;
}

/************************************************************************
 *  global functions							*
 ************************************************************************/
CmdDef*
createFeatureCmds(const IIDCCamera& camera)
{
    return createFeatureCmds(camera, 0);
}

CmdDef*
createFeatureCmds(const Array<IIDCCamera*>& cameras)
{
    cameraChoiceCmds  .resize(cameras.size() + 2);
    cameraChoiceTitles.resize(cameras.size() + 1);
	
    size_t	i = 0;
    for (; i < cameras.size(); ++i)
    {
	(cameraChoiceTitles[i] += "cam-") += ('0' + i);
	    
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
    }
    cameraChoiceTitles[i] = "All";
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

    cameraChoiceCmds[++i].type = C_EndOfList;
	
    featureCmds[0].type		= C_ChoiceFrame;
    featureCmds[0].id		= CAMERA_CHOICE;
    featureCmds[0].val		= cameras.size();
    featureCmds[0].title	= 0;
    featureCmds[0].prop		= cameraChoiceCmds.data();
    featureCmds[0].attrs	= CA_None;
    featureCmds[0].gridx	= 0;
    featureCmds[0].gridy	= 0;
    featureCmds[0].gridWidth	= 1;
    featureCmds[0].gridHeight	= 1;
    featureCmds[0].size		= 0;

    return createFeatureCmds(*cameras[0], 1);
}
	
void
refreshFeatureCmds(const IIDCCamera& camera, CmdPane& cmdPane)
{
    for (CmdDef* featureCmd = featureCmds + 1;
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
	else if (id == IIDCCamera::TRIGGER_MODE)
	    cmdPane.setValue(id, int(camera.getTriggerMode()));
	else if (id == IIDCCamera::WHITE_BALANCE)
	{
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmdPane.setValue(id, int(ub));
	    ++featureCmd;
	    cmdPane.setValue(featureCmd->id, int(vr));
	}
	else
	    cmdPane.setValue(id, int(camera.getValue(
					 IIDCCamera::uintToFeature(id))));
    }
}

void
refreshFeatureCmds(const Array<IIDCCamera*>& cameras, CmdPane& cmdPane)
{
    if (cameras.size() == 0)
	return;

    int	n = cmdPane.getValue(CAMERA_CHOICE);
    if (n < 0 || n >= cameras.size())
	n = 0;
    
    refreshFeatureCmds(*cameras[n], cmdPane);
}

bool
setFeatureValue(const Array<IIDCCamera*>& cameras,
		u_int id, int val, CmdPane& cmdPane)
{
    if (cameras.size() == 0)
	return false;

    if (id == CAMERA_CHOICE)
    {
	const size_t	n = (0 <= val && val < cameras.size() ? val : 0);
	refreshFeatureCmds(*cameras[n], cmdPane);

	return true;
    }
    else
    {
	const size_t	n = cmdPane.getValue(CAMERA_CHOICE);
	return setFeatureValue(cameras, id, val, n);
    }
}

}	// namespace v
}	// namespace TU
