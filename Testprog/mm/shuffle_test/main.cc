/*
 *  $Id: main.cc,v 1.2 2012-08-16 01:31:15 ueshiba Exp $
 */
#include "TU/mmInstructions.h"

int
main()
{
    using namespace	std;
    using namespace	TU::mm;
    
#if defined AVX2
    vec<int>	x(-7, 6, -5, 4, -3, 2, -1, 0);
#elif defined SSE2
    vec<short>	x(7, -6, 5, -4, 3, -2, 1, 0);
#else
    vec<short>	x(3, 2, 1, 0);
#endif    

    cout << n_tuple<2, 0>(x) << endl
	 << n_tuple<2, 1>(x) << endl
	 << "-----------------\n"
	 << n_tuple<4, 0>(x) << endl
	 << n_tuple<4, 1>(x) << endl
	 << n_tuple<4, 2>(x) << endl
	 << n_tuple<4, 3>(x) << endl
	 << "-----------------\n"
	 << n_tuple<8, 0>(x) << endl
	 << n_tuple<8, 1>(x) << endl
	 << n_tuple<8, 2>(x) << endl
	 << n_tuple<8, 3>(x) << endl
	 << n_tuple<8, 4>(x) << endl
	 << n_tuple<8, 5>(x) << endl
	 << n_tuple<8, 6>(x) << endl
	 << n_tuple<8, 7>(x) << endl
	 << endl;

    cout << dup<0>(x) << endl
	 << dup<1>(x) << endl
	 << "-----------------\n"
	 << quadup<0>(x) << endl
	 << quadup<1>(x) << endl
	 << quadup<2>(x) << endl
	 << quadup<3>(x) << endl
	 << "-----------------\n"
	 << octup<0>(x) << endl
	 << octup<1>(x) << endl
	 << octup<2>(x) << endl
	 << octup<3>(x) << endl
	 << octup<4>(x) << endl
	 << octup<5>(x) << endl
	 << octup<6>(x) << endl
	 << octup<7>(x) << endl
	 << endl;

    cout << (x >> 1) << endl << endl;

    vec<short>		s(-800, 700, -600, 500, -400, 300, -200, 100);
    cout << cvt<int>(s) << endl;
    vec<u_short>	us(800, 700, 600, 500, 400, 300, 200, 100);
    cout << cvt<int>(us) << endl << cvt<u_int, 1>(us) << endl << endl;

  //cout << "rot-l:" << rotate_l(cvt<int>(s)) << endl << endl;
    
#if defined AVX2
    cout << shuffle<2, 1, 3, 0>(x) << endl
	 << endl;
#else
    cout << shuffle_low<2, 1, 3, 0>(x) << endl
	 << shuffle_high<2, 1, 3, 0>(x) << endl
	 << endl;
#endif
#if 0
    vec<float>	xf(1), yf(4, 3, 2, 1), zf;

    zf = xf | yf;

    cout << yf << endl
	 << rotate_r(yf) << endl;
#endif
    return 0;
}

