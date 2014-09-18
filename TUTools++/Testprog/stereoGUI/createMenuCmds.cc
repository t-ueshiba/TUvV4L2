/*
 *  $Id: createMenuCmds.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "stereo1394.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static MenuDef	fileMenu[] =
{
    {"Open stereo images",	M_Open,			false, noSub},
    {"Save disparity map",	M_Save,			false, noSub},
    {"Save rectified images",	c_SaveRectifiedImages,	false, noSub},
    {"Save 3D data",		c_SaveThreeD,		false, noSub},
    {"Save 3D-rendered image",	c_SaveThreeDImage,	false, noSub},
    {"-",			M_Line,			false, noSub},
    {"Quit",			M_Exit,			false, noSub},
    EndOfMenu
};

static int	range[] = {50, 1000, 100};

static CmdDef	RadioButtonCmds[] =
{
    {C_RadioButton, c_Texture, 0, "Texture", noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_RadioButton, c_Polygon, 0, "Polygon",  noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_RadioButton, c_Mesh,    0, "Mesh",     noProp, CA_None,
     2, 0, 1, 1, 0},
    EndOfCmds
};
    
static int	prop[6][3];
static CmdDef	MenuCmds[] =
{
    {C_MenuButton,   M_File,		0, "File",	fileMenu, CA_None,
     0, 0, 1, 1, 0},
    {C_Label,	     c_Cursor,		0, "         ",	noProp,   CA_None,
     1, 0, 1, 1, 0},
    {C_Label,	     c_DisparityLabel,	0, "Disparity", noProp,  CA_NoBorder,
     2, 0, 1, 1, 0},
    {C_Label,	     c_Disparity,	0, "             ", noProp, CA_None,
     3, 0, 1, 1, 0},
    {C_Label,	     c_DepthRange,	0, "",		noProp,   CA_NoBorder,
     4, 0, 1, 1, 0},
    {C_Button,	     c_Refresh,		0, "Refresh",	noProp,	  CA_None,
     5, 0, 1, 1, 0}, 
    {C_ToggleButton, c_Binocular,	0, "Binocular",	noProp,   CA_None,
     6, 0, 1, 1, 0}, 
  /*{C_ToggleButton, c_StereoView,	0, "Stereo view", noProp, CA_None,
    8, 0, 1, 1, 0},*/
    {C_Slider,	     c_WindowSize,	0, "Window size", prop[0], CA_None,
     0, 1, 4, 1, 0},
    {C_Slider, c_DisparitySearchWidth,	0,
     "Disparity search width", prop[1], CA_None,
     0, 2, 4, 1, 0},
    {C_Slider, c_DisparityMax,		0,
     "Max. disparity", prop[2], CA_None,
     0, 3, 4, 1, 0},
    {C_Slider,       c_Regularization,  0, "Regularization", prop[5],  CA_None,
     4, 1, 1, 1, 0},
    {C_Slider, c_DisparityInconsistency,0,
     "Disparity inconsistency", prop[3], CA_None,
     4, 2, 1, 1, 0},
    {C_Slider, c_IntensityDiffMax,	0,
     "Max. intensity diff.", prop[4], CA_None,
     4, 3, 1, 1, 0},
    {C_ChoiceFrame,  c_DrawMode, c_Texture, "", RadioButtonCmds,  CA_NoBorder,
     5, 1, 2, 1, 0},
    {C_Slider,       c_GazeDistance,  130, "Gaze distance", range,  CA_None,
     5, 2, 2, 1, 0},
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createMenuCmds()
{
  // Window size
    prop[0][0]  = 3;
    prop[0][1]  = 81;
    prop[0][2]  = 1;

  // Disparity search width
    prop[1][0]  = 16;
    prop[1][1]  = 192;
    prop[1][2]  = 1;

  // Max. disparity
    prop[2][0]  = 48;
    prop[2][1]  = 768;
    prop[2][2]  = 1;

  // Disparity inconsistency
    prop[3][0]  = 0;
    prop[3][1]  = 10;
    prop[3][2]  = 1;

  // Max. intensity diff.
    prop[4][0]  = 0;
    prop[4][1]  = 100;
    prop[4][2]  = 1;

  // Regularization
    prop[5][0]  = 1;
    prop[5][1]  = 255;
    prop[5][2]  = 1;

    return MenuCmds;
}
 
}
}
