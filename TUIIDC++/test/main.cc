#include <iostream>
#include <iomanip>
#include "TU/IIDC++.h"

namespace TU
{
void
snap(uint64_t uniqId)
{
    using namespace	std;
    
    IIDCCamera	camera(IIDCCamera::Monocular, uniqId);

    cerr << "0x" << hex << setw(16) << setfill('0')
	 << camera.globalUniqueId() << dec << ' ' << camera;

    camera.continuousShot();
    cerr << "continuousShot()" << endl;
    
    camera.snap();
    camera.snap();
    camera.snap();
    camera.snap();
    cerr << "snap()" << endl;

    Image<u_char>	image;
    camera >> image;

    image.save(std::cout);
}

void
flow(uint64_t uniqId)
{
    using namespace	std;
    
    IIDCCamera	camera(IIDCCamera::Monocular, uniqId);
#if 1
    camera.setFormatAndFrameRate(IIDCCamera::MONO8_1280x960,
				 IIDCCamera::FrameRate_30);
#endif
    cerr << "0x" << hex << setw(16) << setfill('0')
	 << camera.globalUniqueId() << dec << ' ' << camera;

    camera.continuousShot();
    cerr << "continuousShot()" << endl;

    cout << "M1" << endl;
    Image<u_char>	image(camera.width(), camera.height());
    image.saveHeader(cout);
    for (;;)
    {
	camera.snap();
	camera >> image;

	image.saveData(cout);
    }
}

}

int
main()
{
    uint64_t	uniqId = 0x00b09d0100be72c5LL;
  //uint64_t	uniqId = 0x00b09d0100be72c3LL;
    //TU::snap(uniqId);
    TU::flow(uniqId);
    
    return 0;
}
