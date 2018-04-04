/*
 *  $Id: main.cc,v 1.2 2012-04-20 00:53:14 ueshiba Exp $
 */
#include <cstdlib>
#include <stdexcept>
#include "TU/GraphCuts.h"

namespace TU
{
static int	DE[5][2] =
		{
		    {10, 3}, {2, 9}, {1, 5}, {8, 6}, {9, 2}
		};
static int	SE[5][5] =
		{
		    {0, 3, 0, 0, 0},
		    {3, 0, 3, 0, 0},
		    {0, 3, 0, 4, 0},
		    {0, 0, 4, 0, 1},
		    {0, 0, 0, 1, 0}
		};
    
struct EnergyTerm
{
    int	operator ()(int vid, bool Xv) const
	{
	    int	a = (Xv ? 1 : 0);
	    return DE[vid][a];
	}
    int	operator ()(int uid, int vid, bool Xu, bool Xv)
	{
	    if (Xu == Xv)
		return 0;
	    else
		return SE[uid][vid];
	}
};
    
}

int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	boost;
    
    typedef GraphCuts<long, int, bool, listS>	gc_type;
    typedef gc_type::site_type			site_type;
    typedef gc_type::value_type			value_type;

    bool		build = false;
    bool		energy = false;
    gc_type::Algorithm	alg = gc_type::BoykovKolmogorov;
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "beEPB")) != EOF; )
	switch (c)
	{
	  case 'b':
	    build = true;
	    break;
	  case 'e':
	    energy = true;
	    break;
#ifdef WITH_PARALLEL_EDGES
	  case 'E':
	    alg = gc_type::EdmondsKarp;
	    break;
	  case 'P':
	    alg = gc_type::PushRelabel;
	    break;
#endif
	  case 'B':
	    alg = gc_type::BoykovKolmogorov;
	    break;
	}

    try
    {
	gc_type		gc;
	
	if (build || energy)
	{
	    site_type	v[5];

	  // データ項と平滑化項を生成する．
	    for (int i = 0; i < 5; ++i)
		v[i] = gc.createDataTerm(i);
	    gc.createSmoothingTerm(v[0], v[1]);
	    gc.createSmoothingTerm(v[1], v[0]);
	    gc.createSmoothingTerm(v[1], v[2]);
	    gc.createSmoothingTerm(v[2], v[1]);
	    gc.createSmoothingTerm(v[2], v[3]);
	    gc.createSmoothingTerm(v[3], v[2]);
	    gc.createSmoothingTerm(v[3], v[4]);
	    gc.createSmoothingTerm(v[4], v[3]);

	  // 初期ラベルを与える．
	    for (int i = 0; i < 5; ++i)
		gc(v[i]) = false;

	    if (energy)
	    {
		value_type
		    minval = gc.alphaExpansion(true, TU::EnergyTerm(), alg);
		cout << "minimum energy(computed): " << minval
		     << '(' << gc.value(TU::EnergyTerm()) << ')'
		     << endl;
	    }
	    else
	    {
		gc.dataEnergy(v[0], true)  = 3;
		gc.dataEnergy(v[0], false) = 10;
		gc.dataEnergy(v[1], true)  = 9;
		gc.dataEnergy(v[1], false) = 2;
		gc.dataEnergy(v[2], true)  = 5;
		gc.dataEnergy(v[2], false) = 1;
		gc.dataEnergy(v[3], true)  = 6;
		gc.dataEnergy(v[3], false) = 8;
		gc.dataEnergy(v[4], true)  = 2;
		gc.dataEnergy(v[4], false) = 9;

		gc.smoothingEnergy(v[0], v[1]) = gc.smoothingEnergy(v[1], v[0])
					       = 3;
		gc.smoothingEnergy(v[1], v[2]) = gc.smoothingEnergy(v[2], v[1])
					       = 3;
		gc.smoothingEnergy(v[2], v[3]) = gc.smoothingEnergy(v[3], v[2])
					       = 4;
		gc.smoothingEnergy(v[3], v[4]) = gc.smoothingEnergy(v[4], v[3])
					       = 1;

		value_type	maxflow = gc.maxFlow(true, alg);
		cout << "max flow: " << maxflow << endl;
	    }
	}
	else
	{
	    gc.getDimacsMaxFlow(cin);
	    value_type	maxflow = gc.maxFlow(true, alg);
	    cout << "max flow: " << maxflow << endl;
	}

	gc.putMaxFlow(cout) << endl;
	gc.putMinCut(cout);
    }
    catch (std::exception& err)
    {
	cerr << err.what() << endl;
	return 1;
    }
    
    return 0;
}
