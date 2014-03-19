/*
 *  $Id: createCaptureCmds.cc,v 1.2 2010-11-19 06:31:09 ueshiba Exp $
 */
#include "testcam.h"

namespace TU
{
namespace v
{
/************************************************************************
*  local data								*
************************************************************************/
static CmdDef CaptureCmds[] =
{
    {C_ToggleButton,  c_ContinuousShot, 0, "Continuous shot", noProp, CA_None,
     0, 0, 1, 1, 0},
    {C_Button,	      c_OneShot,        0, "One shot",	      noProp, CA_None,
     0, 1, 1, 1, 0},
    {C_ToggleButton,  c_Trigger,        0, "Trigger",	      noProp, CA_None,
     0, 2, 1, 1, 0},
    {C_ToggleButton,  c_PlayMovie,	0, "Play",	      noProp, CA_None,
     1, 0, 1, 1, 0},
    {C_Button,  c_BackwardMovie, 0, "<",    noProp, CA_None, 2, 0, 1, 1, 0},
    {C_Button,  c_ForwardMovie,  0, ">",    noProp, CA_None, 3, 0, 1, 1, 0},
    {C_Slider,  c_StatusMovie,   0, "",     noProp, CA_None, 1, 1, 3, 1, 0},
    EndOfCmds
};

/************************************************************************
*  global functions							*
************************************************************************/
CmdDef*
createCaptureCmds()
{
    return CaptureCmds;
}
 
}
}
