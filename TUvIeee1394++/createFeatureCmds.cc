/*
 *  $Id: createFeatureCmds.cc,v 1.2 2012-06-20 07:50:08 ueshiba Exp $
*/
#include "TU/v/vIeee1394++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
struct Feature
{
    Ieee1394Camera::Feature	id;
    const char*			name;
    int				prop[3];
};
static Feature		features[] =
{
    {Ieee1394Camera::TRIGGER_MODE,	"Trigger mode"	 },
    {Ieee1394Camera::BRIGHTNESS,	"Brightness"	 },
    {Ieee1394Camera::AUTO_EXPOSURE,	"Auto exposure"	 },
    {Ieee1394Camera::SHARPNESS,		"Sharpness"	 },
    {Ieee1394Camera::WHITE_BALANCE,	"White bal.(U/B)"},
    {Ieee1394Camera::WHITE_BALANCE,	"White bal.(V/R)"},
    {Ieee1394Camera::HUE,		"Hue"		 },
    {Ieee1394Camera::SATURATION,	"Saturation"	 },
    {Ieee1394Camera::GAMMA,		"Gamma"		 },
    {Ieee1394Camera::SHUTTER,		"Shutter"	 },
    {Ieee1394Camera::GAIN,		"Gain"		 },
    {Ieee1394Camera::IRIS,		"Iris"		 },
    {Ieee1394Camera::FOCUS,		"Focus"		 },
    {Ieee1394Camera::TEMPERATURE,	"Temperature"	 },
    {Ieee1394Camera::ZOOM,		"Zoom"		 },
  //{Ieee1394Camera::PAN,		"Pan"		 },
  //{Ieee1394Camera::TILT,		"Tilt"		 },
};
static const size_t		NFEATURES = sizeof(features)
					  / sizeof(features[0]);
static CmdDef			featureCmds[3*NFEATURES + 2];
static Array<CmdDef>		cameraChoiceCmds;
static Array<std::string>	cameraChoiceTitles;

/************************************************************************
*  static functions							*
************************************************************************/
static CmdDef*
createFeatureCmds(const Ieee1394Camera& camera, size_t ncmds)
{
    u_int	y = ncmds;
    for (size_t i = 0; i < NFEATURES; ++i)
    {
	Feature&	feature = features[i];
	const u_int	inq = camera.inquireFeatureFunction(feature.id);
	
	if (inq & Ieee1394Camera::Presence)
	{
	    u_int	x = 1;
	    
	    if (inq & Ieee1394Camera::OnOff)
	    {
	      // Create toggle button for turning on/off this feature.
		featureCmds[ncmds].type	      = C_ToggleButton;
		featureCmds[ncmds].id	      = feature.id
					      + IEEE1394CAMERA_OFFSET_ONOFF;
		featureCmds[ncmds].val	      = camera.isTurnedOn(feature.id);
		featureCmds[ncmds].title      = "On";
		featureCmds[ncmds].prop       = noProp;
		featureCmds[ncmds].attrs      = CA_None;
		featureCmds[ncmds].gridx      = x++;
		featureCmds[ncmds].gridy      = y;
		featureCmds[ncmds].gridWidth  = 1;
		featureCmds[ncmds].gridHeight = 1;
		featureCmds[ncmds].size	      = 0;
		++ncmds;
	    }

	    if (feature.id == Ieee1394Camera::TRIGGER_MODE)
	    {
		featureCmds[ncmds].type	      = C_Label;
		featureCmds[ncmds].id	      = feature.id;
		featureCmds[ncmds].val	      = 0;
		featureCmds[ncmds].title      = feature.name;
		featureCmds[ncmds].prop       = noProp;
		featureCmds[ncmds].attrs      = CA_NoBorder;
		featureCmds[ncmds].gridx      = 0;
		featureCmds[ncmds].gridy      = y;
		featureCmds[ncmds].gridWidth  = 1;
		featureCmds[ncmds].gridHeight = 1;
		featureCmds[ncmds].size	      = 0;
		++ncmds;
	    }
	    else if (inq & Ieee1394Camera::Manual)
	    {
		if (inq & Ieee1394Camera::Auto)
		{
		  // Create toggle button for setting manual/auto mode.
		    featureCmds[ncmds].type	  = C_ToggleButton;
		    featureCmds[ncmds].id	  = feature.id
						  + IEEE1394CAMERA_OFFSET_AUTO;
		    featureCmds[ncmds].val	  = camera.isAuto(feature.id);
		    featureCmds[ncmds].title	  = "Auto";
		    featureCmds[ncmds].prop       = noProp;
		    featureCmds[ncmds].attrs      = CA_None;
		    featureCmds[ncmds].gridx      = x;
		    featureCmds[ncmds].gridy      = y;
		    featureCmds[ncmds].gridWidth  = 1;
		    featureCmds[ncmds].gridHeight = 1;
		    featureCmds[ncmds].size	  = 0;
		    ++ncmds;
		}

		featureCmds[ncmds].id	      = feature.id;
		featureCmds[ncmds].title      = feature.name;
		featureCmds[ncmds].gridx      = 0;
		featureCmds[ncmds].gridy      = y;
		featureCmds[ncmds].gridWidth  = 1;
		featureCmds[ncmds].gridHeight = 1;
		featureCmds[ncmds].size	      = 0;
		
		if (inq & Ieee1394Camera::ReadOut)
		{
		  // Create sliders for setting values.
		    featureCmds[ncmds].type  = C_Slider;
		    featureCmds[ncmds].prop  = feature.prop;
		    featureCmds[ncmds].attrs = CA_None;

		    u_int	min, max;
		    camera.getMinMax(feature.id, min, max);
		    feature.prop[0] = min;
		    feature.prop[1] = max - min;
		    feature.prop[2] = 1;

		    if (feature.id == Ieee1394Camera::WHITE_BALANCE)
		    {
			++ncmds;
			++i;
			++y;
			features[i].prop[0]	      = min;
			features[i].prop[1]	      = max - min;
			features[i].prop[2]	      = 1;
			featureCmds[ncmds].type	      = C_Slider;
			featureCmds[ncmds].id	      = features[i].id + 0x02;
			featureCmds[ncmds].title      = features[i].name;
			featureCmds[ncmds].prop       = features[i].prop;
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
			featureCmds[ncmds].val = camera.getValue(feature.id);
		}
		else
		{
		  // Create a label for setting on/off and manual/auto modes.
		    featureCmds[ncmds].type  = C_Label;
		    featureCmds[ncmds].prop  = noProp;
		    featureCmds[ncmds].attrs = CA_NoBorder;
		}
		
		++ncmds;
	    }

	    ++y;
	}
    }
    featureCmds[ncmds].type = C_EndOfList;

    return featureCmds;
}

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createFeatureCmds(const Ieee1394Camera& camera)
{
    return createFeatureCmds(camera, 0);
}

CmdDef*
createFeatureCmds(const Ieee1394CameraArray& cameras)
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
    featureCmds[0].id		= 0;
    featureCmds[0].val		= cameras.size();
    featureCmds[0].title	= 0;
    featureCmds[0].prop		= cameraChoiceCmds.data();
    featureCmds[0].attrs	= CA_None;
    featureCmds[0].gridx	= 0;
    featureCmds[0].gridy	= 0;
    featureCmds[0].gridWidth	= 1;
    featureCmds[0].gridHeight	= 1;
    featureCmds[0].size		= 0;

    createFeatureCmds(*cameras[0], 1);
}
	
void
refreshFeatureCmds(const Ieee1394Camera& camera, CmdPane& cmdPane)
{
    for (CmdDef* featureCmd = featureCmds + 1;
	 featureCmd->type != C_EndOfList; ++featureCmd)
    {
	const u_int	id = featureCmd->id;
	
	if (id >= Ieee1394Camera::BRIGHTNESS + IEEE1394CAMERA_OFFSET_AUTO)
	    cmdPane.setValue(id,
			     int(camera.isAuto(
				     Ieee1394Camera::uintToFeature(
					 id - IEEE1394CAMERA_OFFSET_AUTO))));
	else if (id >= Ieee1394Camera::BRIGHTNESS + IEEE1394CAMERA_OFFSET_ONOFF)
	    cmdPane.setValue(id,
			     int(camera.isTurnedOn(
				     Ieee1394Camera::uintToFeature(
					 id - IEEE1394CAMERA_OFFSET_ONOFF))));
	else if (id == Ieee1394Camera::WHITE_BALANCE)
	{
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);
	    cmdPane.setValue(id, int(ub));
	    ++featureCmd;
	    cmdPane.setValue(featureCmd->id, int(vr));
	}
	else
	    cmdPane.setValue(id, int(camera.getValue(
					 Ieee1394Camera::uintToFeature(id))));
    }
}

}	// namespace v
}	// namespace TU
