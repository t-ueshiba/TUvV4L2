/*
 *  $Id$
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
