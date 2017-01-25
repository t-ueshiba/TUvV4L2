/*
 *  $Id: testcam.h,v 1.2 2010-11-19 06:31:09 ueshiba Exp $
 */
#include <sys/time.h>
#include <iostream>
#include "TU/v/TUv++.h"

namespace TU
{
namespace v
{
enum
{
  // # of movie frames.
    c_NFrames,

  // Camera control.
    c_ContinuousShot,
    c_OneShot,
    c_PlayMovie,
    c_RecordMovie,
    c_StatusMovie,
    c_ForwardMovie,
    c_BackwardMovie
};

/************************************************************************
*  global functions							*
************************************************************************/
inline const CmdDef*
createCaptureCmds()
{
    static const CmdDef captureCmds[] =
    {
	{C_ToggleButton, c_ContinuousShot, 0, "Continuous shot", noProp,
	 CA_None, 0, 0, 1, 1, 0},
	{C_Button,	 c_OneShot,        0, "One shot",	 noProp,
	 CA_None, 0, 1, 1, 1, 0},
	{C_ToggleButton,c_PlayMovie,	   0, "Play",		 noProp,
	 CA_None, 1, 0, 1, 1, 0},
	{C_Button,	 c_BackwardMovie,  0, "<",		 noProp,
	 CA_None, 2, 0, 1, 1, 0},
	{C_Button,	 c_ForwardMovie,   0, ">",		 noProp,
	 CA_None, 3, 0, 1, 1, 0},
	{C_Slider,	 c_StatusMovie,    0, "Frame",		 noProp,
	 CA_None, 1, 1, 3, 1, 0},
	EndOfCmds
    };

    return captureCmds;
}

template <class CAMERA> inline const CmdDef*
createMenuCmds(const CAMERA& camera)
{
    static MenuDef nframesMenu[] =
    {
	{" 10",  10, false, noSub},
	{"100", 100, true,  noSub},
	{"300", 300, false, noSub},
	{"600", 600, false, noSub},
	EndOfMenu
    };

    static MenuDef fileMenu[] =
    {
	{"Save current image",			M_Save,   false, noSub},
	{"Save camera config. to memory",	M_SaveAs, false, noSub},
	{"Restore camera config. from memory",	M_Open,   false, noSub},
	{"-",					M_Line,   false, noSub},
	{"Quit",				M_Exit,   false, noSub},
	EndOfMenu

    };

    static CmdDef menuCmds[] =
    {
	{C_MenuButton, M_File,   0, "File",   fileMenu, CA_None, 0, 0, 1, 1, 0},
	{C_MenuButton, M_Format, 0, "Format", noProp,   CA_None, 1, 0, 1, 1, 0},
	{C_ChoiceMenuButton, c_NFrames, 100, "# of movie frames", nframesMenu,
	 CA_None, 2, 0, 1, 1, 0},
	EndOfCmds
    };

    menuCmds[1].prop = createFormatMenu(camera);
    
    return menuCmds;
}

}

/************************************************************************
*  global functions							*
************************************************************************/
template <class CAMERA> inline void
saveCameraConfig(CAMERA& camera)
{
}
    
template <class CAMERA> inline void
restoreCameraConfig(CAMERA& camera)
{
}
    
#ifdef __TU_IIDCPP_H
inline void
saveCameraConfig(IIDCCamera& camera)
{
    camera.saveConfig(1);
}
	
inline void
restoreCameraConfig(IIDCCamera& camera)
{
    camera.restoreConfig(1);
}
#endif
    
inline void
countTime()
{
    static int		nframes = 0;
    static timeval	start;
    
    if (nframes == 10)
    {
	timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec) +
	    (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

inline std::ostream&
printTime(std::ostream& out, u_int64_t localtime)
{
    u_int64_t	usec = localtime % 1000;
    u_int64_t	msec = (localtime / 1000) % 1000;
    u_int64_t	sec  = localtime / 1000000;
    return out << sec << '.' << msec << '.' << usec;
}

}
