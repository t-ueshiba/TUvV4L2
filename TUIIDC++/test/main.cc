/*
 *  $Id$
 */
#include <sys/time.h>
#include <signal.h>
#include <iostream>
#include <iomanip>
#include "TU/IIDC++.h"

namespace TU
{
bool	active = true;

/************************************************************************
*  static functions							*
************************************************************************/
static void
count_time()
{
    static int		nframes = 0;
    static timeval	start;
    
    if (nframes == 10)
    {
	timeval	end;
	gettimeofday(&end, NULL);
	double	interval = (end.tv_sec  - start.tv_sec)
			 + (end.tv_usec - start.tv_usec) / 1.0e6;
	std::cerr << nframes / interval << " frames/sec" << std::endl;
	nframes = 0;
    }
    if (nframes++ == 0)
	gettimeofday(&start, NULL);
}

static std::ostream&
print_cycletime(std::ostream& out, uint32_t cycletime)
{
    uint32_t	sec	 = (cycletime & 0xfe000000) >> 25;
    uint32_t	cycle	 = (cycletime & 0x01fff000) >> 12;
    uint32_t	subcycle = (cycletime & 0x00000fff);

    return out << sec << '.' << cycle << '.' << subcycle;
}
    
static void
handler(int sig)
{
    std::cerr << "Signal [" << sig << "] caught!" << std::endl;
    active = false;
}

static void
flow(uint64_t uniqId)
{
    using namespace	std;

  // signal handlerを登録する．
    signal(SIGINT,  handler);
    signal(SIGPIPE, handler);

    IIDCCamera	camera(IIDCCamera::Monocular, uniqId);
  /*
    float	min, max;
    camera.getAbsMinMax(IIDCCamera::BRIGHTNESS, min, max);
    cerr << "Brightness: min = " << min << ", max = " << max << endl;
    camera.setAbsValue(IIDCCamera::BRIGHTNESS, 11.1);
    cerr << "Brightness = " << camera.getAbsValue(IIDCCamera::BRIGHTNESS) << endl;
  */
    camera.embedTimestamp();

    camera.continuousShot();
    cout << "M1" << endl;
    Image<u_char>	image(camera.width(), camera.height());
    image.saveHeader(cout);
    while (active)
    {
	camera.snap();
	camera >> image;

	cerr << "capture time:\t";
	print_cycletime(cerr,
			ntohl(*reinterpret_cast<uint32_t*>(image.data())));
	cerr << endl;

	cerr << "current time:\t";
	uint64_t	localtime;
	print_cycletime(cerr, camera.getCycleTime(localtime)) << endl << endl;

	count_time();
	
	image.saveData(cout);
    }

    cerr << "0x" << hex << setw(16) << setfill('0')
	 << camera.globalUniqueId() << dec << ' ' << camera;
}

}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    uint64_t	uniqId = 0;
    extern int	optind;
    if (optind < argc)
	uniqId = strtoull(argv[optind], 0, 0);
    
    TU::flow(uniqId);
    
    return 0;
}
