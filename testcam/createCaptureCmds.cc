/*
 *  $Id: createCaptureCmds.cc,v 1.1 2012-06-19 06:14:31 ueshiba Exp $
 */
#include "multicam.h"

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
