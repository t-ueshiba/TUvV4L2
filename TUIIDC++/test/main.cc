#include <signal.h>
#include <iostream>
#include <iomanip>
#include "TU/IIDC++.h"

namespace TU
{
bool	active = true;

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

void
flow(uint64_t uniqId)
{
    using namespace	std;

  // signal handlerを登録する．
    signal(SIGINT,  handler);
    signal(SIGPIPE, handler);

    IIDCCamera	camera(IIDCCamera::Monocular, uniqId);
#if 0
    camera.setFormatAndFrameRate(IIDCCamera::MONO8_1280x960,
				 IIDCCamera::FrameRate_30);
#endif
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
	
	image.saveData(cout);
    }

    cerr << "0x" << hex << setw(16) << setfill('0')
	 << camera.globalUniqueId() << dec << ' ' << camera;
}

}

int
main()
{
  //uint64_t	uniqId = 0x00b09d0100be72c3LL;
    uint64_t	uniqId = 0;
    
  //TU::snap(uniqId);
    TU::flow(uniqId);
    
    return 0;
}
