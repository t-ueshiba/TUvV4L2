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
    {"-",			M_Line,			false, noSub},
    {"Quit",			M_Exit,			false, noSub},
    EndOfMenu
};

static int	prop[7][3];

static CmdDef	RadioButtonCmds[] =
{
    {C_RadioButton, c_SAD,		0, "SAD",	    noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_RadioButton, c_GuidedFilter,	0, "Guided filter", noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_RadioButton, c_TreeFilter,	0, "Tree filter",   noProp, CA_None,
     2, 0, 1, 1, 0},
    EndOfCmds
};
    
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
    {C_ToggleButton,
     c_DoHorizontalBackMatch,		0, "Hor. back match", noProp, CA_None,
     4, 0, 1, 1, 0}, 

    {C_ChoiceFrame, c_Algorithm, c_SAD, "", RadioButtonCmds, CA_NoBorder,
     0, 1, 5, 1, 0},
    {C_Slider, c_DisparitySearchWidth,	0,
     "Disparity search width", prop[0], CA_None,
     0, 2, 5, 1, 0},
    {C_Slider,	    c_WindowSize,	0, "Window size", prop[1], CA_None,
     0, 3, 5, 1, 0},
    {C_Slider,      c_Regularization,	0, "Regularization", prop[2],  CA_None,
     0, 4, 5, 1, 0},
    {C_Slider,
     c_DisparityInconsistency,	0, "Disparity inconsistency", prop[3], CA_None,
     0, 5, 5, 1, 0},
    {C_Slider, c_IntensityDiffMax, 0, "Max. intensity diff.", prop[4], CA_None,
     0, 6, 5, 1, 0},
    {C_ToggleButton, c_WMF,	0, "Weighted median filter",  noProp,  CA_None,
     0, 7, 5, 1, 0},
    {C_Slider,	    c_WMFWindowSize,	5, "WMF window size", prop[5], CA_None,
     0, 8, 5, 1, 0},
    {C_Slider,	    c_WMFSigma,		11, "WMF sigma",      prop[6], CA_None,
     0, 9, 5, 1, 0},

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
    prop[0][1]  = 96;
    prop[0][2]  = 1;

  // Window size
    prop[1][0]  = 3;
    prop[1][1]  = 65;
    prop[1][2]  = 1;

  // Regularization
    prop[2][0]  = 1;
    prop[2][1]  = 64;
    prop[2][2]  = 1;

  // Disparity inconsistency
    prop[3][0]  = 0;
    prop[3][1]  = 10;
    prop[3][2]  = 1;

  // Max. intensity diff.
    prop[4][0]  = 0;
    prop[4][1]  = 100;
    prop[4][2]  = 1;

  // WMF window size.
    prop[5][0]  = 1;
    prop[5][1]  = 64;
    prop[5][2]  = 1;

  // WMF sigma.
    prop[6][0]  = 1;
    prop[6][1]  = 255;
    prop[6][2]  = 2;

    return MenuCmds;
}
 
}
}
