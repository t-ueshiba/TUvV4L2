/*
 *  $Id$
 */
#include <vector>
#include <string>
#include <boost/foreach.hpp>
#include "TU/v/vV4L2++.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static const size_t		NFEATURES = 30;
static CmdDef			featureCmds[1 + NFEATURES + 1];
static int			props[NFEATURES][3];
static std::vector<MenuDef>	menus[NFEATURES];
static Array<CmdDef>		cameraChoiceCmds;
static Array<std::string>	cameraChoiceTitles;
    
/************************************************************************
*  static functions							*
************************************************************************/
static CmdDef*
createFeatureCmds(const V4L2Camera& camera, size_t n)
{
    BOOST_FOREACH (V4L2Camera::Feature feature, camera.availableFeatures())
    {
	featureCmds[n].id	  = feature;
	featureCmds[n].val	  = camera.getValue(feature);
	featureCmds[n].title      = camera.getName(feature).c_str();
	featureCmds[n].attrs      = CA_None;
	featureCmds[n].gridx      = 0;
	featureCmds[n].gridy      = n;
	featureCmds[n].gridWidth  = 1;
	featureCmds[n].gridHeight = 1;
	featureCmds[n].size	  = 0;
	
	V4L2Camera::MenuItemRange
	    menuItems = camera.availableMenuItems(feature);

	if (menuItems.first == menuItems.second)
	{
	    int	min, max, step;
	    camera.getMinMaxStep(feature, min, max, step);

	    if (min == 0 && max == 1)
	    {
		featureCmds[n].type = C_ToggleButton;
		featureCmds[n].prop = 0;
	    }
	    else
	    {
		featureCmds[n].type = C_Slider;
		props[n][0] = min;
		props[n][1] = max - min;
		props[n][2] = step;
		featureCmds[n].prop = props[n];
	    }
	}
	else
	{
	    featureCmds[n].type = C_ChoiceMenuButton;

	    menus[n].resize(std::distance(menuItems.first,
					  menuItems.second) + 1);
	    size_t	i = 0;
	    BOOST_FOREACH (const V4L2Camera::MenuItem& menuItem, menuItems)
	    {
		menus[n][i].label   = menuItem.name.c_str();
		menus[n][i].id	    = menuItem.index;
		menus[n][i].checked = (i == featureCmds[n].val);
		menus[n][i].submenu = noSub;

		++i;
	    }
	    menus[n][i].label = 0;

	    featureCmds[n].prop = &menus[n][0];
	}

	++n;
    }
    featureCmds[n].type = C_EndOfList;

    return featureCmds;
}

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createFeatureCmds(const V4L2Camera& camera)
{
    return createFeatureCmds(camera, 0);
}

CmdDef*
createFeatureCmds(const Array<V4L2Camera*>& cameras)
{
    cameraChoiceCmds  .resize(cameras.size() + 2);
    cameraChoiceTitles.resize(cameras.size() + 1);
	
    size_t	i = 0;
    for (; i < cameras.size(); ++i)
    {
	(cameraChoiceTitles[i] += "cam-") += ('0' + i);
	    
	cameraChoiceCmds[i].type       = C_RadioButton;
	cameraChoiceCmds[i].id	       = i;
	cameraChoiceCmds[i].val	       = 0;
	cameraChoiceCmds[i].title      = cameraChoiceTitles[i].c_str();
	cameraChoiceCmds[i].prop       = noProp;
	cameraChoiceCmds[i].attrs      = CA_None;
	cameraChoiceCmds[i].gridx      = i;
	cameraChoiceCmds[i].gridy      = 0;
	cameraChoiceCmds[i].gridWidth  = 1;
	cameraChoiceCmds[i].gridHeight = 1;
	cameraChoiceCmds[i].size       = 0;
    }
    cameraChoiceTitles[i] = "All";
    cameraChoiceCmds[i].type       = C_RadioButton;
    cameraChoiceCmds[i].id	   = i;
    cameraChoiceCmds[i].val	   = 0;
    cameraChoiceCmds[i].title      = cameraChoiceTitles[i].c_str();
    cameraChoiceCmds[i].prop       = noProp;
    cameraChoiceCmds[i].attrs      = CA_None;
    cameraChoiceCmds[i].gridx      = i;
    cameraChoiceCmds[i].gridy      = 0;
    cameraChoiceCmds[i].gridWidth  = 1;
    cameraChoiceCmds[i].gridHeight = 1;
    cameraChoiceCmds[i].size       = 0;

    cameraChoiceCmds[++i].type = C_EndOfList;
	
    featureCmds[0].type	      = C_ChoiceFrame;
    featureCmds[0].id	      = V4L2Camera::UNKNOWN_FEATURE;
    featureCmds[0].val	      = cameras.size();
    featureCmds[0].title      = 0;
    featureCmds[0].prop       = cameraChoiceCmds.data();
    featureCmds[0].attrs      = CA_None;
    featureCmds[0].gridx      = 0;
    featureCmds[0].gridy      = 0;
    featureCmds[0].gridWidth  = 1;
    featureCmds[0].gridHeight = 1;
    featureCmds[0].size	      = 0;

    createFeatureCmds(*cameras[0], 1);
}

void
refreshFeatureCmds(const V4L2Camera& camera, CmdPane& cmdPane)
{
    for (CmdDef* featureCmd = featureCmds + 1;
	 featureCmd->type != C_EndOfList; ++featureCmd)
	cmdPane.setValue(featureCmd->id,
			 camera.getValue(
			     V4L2Camera::uintToFeature(featureCmd->id)));
}

void
refreshFeatureCmds(const Array<V4L2Camera*>& cameras, CmdPane& cmdPane)
{
    if (cameras.size() == 0)
	return;

    int	n = cmdPane.getValue(V4L2Camera::UNKNOWN_FEATURE);
    if (n < 0 || n >= cameras.size())
	n = 0;
    
    refreshFeatureCmds(*cameras[n], cmdPane);
}

bool
handleCameraFeatures(const Array<V4L2Camera*>& cameras,
		     u_int id, int val, CmdPane& cmdPane)
{
    if (cameras.size() == 0)
	return false;

    if (id == V4L2Camera::UNKNOWN_FEATURE)
    {
	const size_t	n = (0 <= val && val < cameras.size() ? val : 0);
	refreshFeatureCmds(*cameras[n], cmdPane);

	return true;
    }
    else
    {
	const size_t	n = cmdPane.getValue(V4L2Camera::UNKNOWN_FEATURE);
	return handleCameraFeatures(cameras, id, val, n);
    }
}
    
}
}
