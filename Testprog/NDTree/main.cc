/*
 *  $Id: main.cc,v 1.2 2011-06-30 02:46:48 ueshiba Exp $
 */
#include <cstdlib>
#include <stdexcept>
#include <fstream>
#include "TU/NDTree++.h"
#include "TU/Manip.h"

namespace TU
{
/************************************************************************
*  static functions							*
************************************************************************/
}

/************************************************************************
*  global functions							*
************************************************************************/
int
main(int argc, char* argv[])
{
    using namespace	std;
    using namespace	TU;
    
    typedef NDTree<double, 2u>			tree_type;
    typedef tree_type::value_type		value_type;
    typedef tree_type::const_pointer		const_pointer;
    typedef tree_type::position_type		position_type;

    const char*		treeFileName = "tree.dat";
    extern char*	optarg;
    for (int c; (c = getopt(argc, argv, "f:")) != -1; )
	switch (c)
	{
	  case 'f':
	    treeFileName = optarg;
	    break;
	}

  //tree_type	tree(length);
    tree_type	tree;
    
    cerr << ">> ";
    for (char c; cin >> c; )
    {
	try
	{
	    position_type	pos;
	    
	    switch (c)
	    {
	      case 'i':
	      {
		value_type	val;
		cin >> pos >> val >> skipl;
		tree.insert(pos, val);
	      }
		break;
	      case 'e':
		cin >> pos >> skipl;
		tree.erase(pos);
		break;
	      case 'f':
	      {
		cin >> pos >> skipl;
		const_pointer	p = tree.find(pos);
		cerr << "tree(";
		pos.put(cerr);
		cerr << ") = ";
		if (p)
		    cerr << *p << endl;
		else
		    cerr << "NULL" << endl;
	      }
		break;
	      case 's':
		cerr << "tree size is " << tree.size() << "." << endl;
		break;
	      case 'o':
		cerr << "root origin is " << tree.origin();
		break;
	      case 'l':
		cerr << "root cell length is " << tree.length0() << '.'
		     << endl;
		break;
	      case 'c':
		tree.clear();
		break;
	      case 'p':
		tree.put(cerr);
		break;
	      case 'q':
		tree.print(cerr);
		break;
	      case 'P':
	      {
		ofstream	out(treeFileName);
		if (!out)
		    throw runtime_error("Failed to open the output file!");
		tree.put(out);
	      }
		break;
	      case 'G':
	      {
		ifstream	in(treeFileName);
		if (!in)
		    throw runtime_error("Failed to open the input file!");
		tree.get(in);
	      }
		break;
	      default:
		cerr << "unknown command: \'" << c << '\'' << endl;
		break;
	    }
	}
	catch (exception& err)
	{
	    cerr << err.what() << endl;
	}

	cerr << ">> ";
    }

    return 0;
}
