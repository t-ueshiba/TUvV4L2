/*
 *  $Id: createMenuCmds.cc 1246 2012-11-30 06:23:09Z ueshiba $
 */
#include "TU/v/TUv++.h"
#include "stereoGUI.h"

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
    
static float	prop[9][3];
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

    {C_ToggleButton,
     c_DoHorizontalBackMatch,		0, "Hor. back match", noProp, CA_None,
     5, 0, 1, 1, 0}, 
    {C_ToggleButton,
     c_DoVerticalBackMatch,		0, "Ver. back match", noProp, CA_None,
     6, 0, 1, 1, 0}, 

    {C_ToggleButton, c_Binocular,	0, "Binocular",	noProp,   CA_None,
     7, 0, 1, 1, 0}, 
  /*{C_ToggleButton, c_StereoView,	0, "Stereo view", noProp, CA_None,
    8, 0, 1, 1, 0},*/

    {C_Slider, c_DisparitySearchWidth,	0, "Disparity search width",
     prop[0], CA_None,
     0, 1, 2, 1, 0},
    {C_Slider, c_DisparityMax,		0, "Max. disparity", prop[1], CA_None,
     0, 2, 2, 1, 0},
    {C_Slider,	     c_WindowSize,	0, "Window size", prop[2], CA_None,
     0, 3, 2, 1, 0},

    {C_Slider,
     c_IntensityDiffMax,	0, "Max. intensity diff.", prop[3], CA_None,
     2, 1, 3, 1, 0},
    {C_Slider,
     c_DerivativeDiffMax,	0, "Max. derivative diff.", prop[4], CA_None,
     2, 2, 3, 1, 0},
    {C_Slider, c_Blend,	0, "blend ratio", prop[5], CA_None,
     2, 3, 3, 1, 0},

    {C_Slider,
     c_DisparityInconsistency,	0, "Disparity inconsistency", prop[6], CA_None,
     5, 1, 2, 1, 0},
    {C_Slider,       c_Regularization,  0, "Regularization", prop[7],  CA_None,
     5, 2, 2, 1, 0},

    {C_ChoiceFrame,  c_DrawMode, c_Texture, "", RadioButtonCmds,  CA_NoBorder,
     7, 1, 1, 1, 0},
    {C_Slider,	     c_GazeDistance,  1.3f, "Gaze distance", prop[8],  CA_None,
     7, 2, 1, 1, 0},
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createMenuCmds()
{
  // Disparity search width
    prop[0][0]  = 16;
    prop[0][1]  = 208;
    prop[0][2]  = 1;

  // Max. disparity
    prop[1][0]  = 48;
    prop[1][1]  = 816;
    prop[1][2]  = 1;

  // Window size
    prop[2][0]  = 3;
    prop[2][1]  = 84;
    prop[2][2]  = 1;

  // Max. intensity diff.
    prop[3][0]  = 0;
    prop[3][1]  = 100;
    prop[3][2]  = 1;

  // Max. derivative diff.
    prop[4][0]  = 0;
    prop[4][1]  = 100;
    prop[4][2]  = 1;

  // blend ratio
    prop[5][0]  = 0;
    prop[5][1]  = 1;
    prop[5][2]  = 0.01;

  // Disparity inconsistency
    prop[6][0]  = 0;
    prop[6][1]  = 10;
    prop[6][2]  = 1;

  // Regularization
    prop[7][0]  = 1;
    prop[7][1]  = 256;
    prop[7][2]  = 1;

  // Gaze distance
    prop[8][0]  = 0.5;
    prop[8][1]  = 10;
    prop[8][2]  = 0.1;

    return MenuCmds;
}
 
}
}
