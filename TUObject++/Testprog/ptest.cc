/*
 *  $Id: ptest.cc,v 1.4 2002-07-26 08:56:04 ueshiba Exp $
 */
#include "TU/Object++.h"
#ifdef __GNUG__
#  include "TU/Object++.cc"
#endif

namespace TU
{
const unsigned	id_Int  = 256;
const unsigned	id_Cons = 257;

/*
 *  class Int
 */
class Int : public Object
{
  public:
    static Ptr<Int>	newInt(int i)		{return new Int(i);}
    int			value()		const	{return val;}

    DECLARE_COPY_AND_RESTORE(Int)

  protected:
    void		saveGuts(std::ostream& out) const
			{out.write((const char*)&val, sizeof(val));}
    void		restoreGuts(std::istream& in)
			{in.read((char*)&val, sizeof(val));}
    
  private:
    Int(int i = 0)	:val(i)			{}

    const int		val;

    DECLARE_DESC
    DECLARE_CONSTRUCTORS(Int)
};

const Object::Desc	Int::_desc(id_Int, 0, Int::newObject, MbrpEnd);
template <>
const Object::Desc	Cons<Int>::_desc(id_Cons, 0,
					 Cons<Int>::newObject,
					 &Cons<Int>::_ca,
					 &Cons<Int>::_cd,
					 MbrpEnd);

/*
 *  Output functions
 */
std::ostream&
operator <<(std::ostream& out, Int* p)
{
    return out << p->value() << " ";
}

std::ostream&
operator <<(std::ostream& out, Cons<Int>* cns)
{
    for (; cns->consp(); cns = cns->cdr())
	out << cns->car();
    return out;
}

/*
 *  Program body
 */
Ptr<Cons<Int> >
my_reverse(Cons<Int>* cns)
{
    if (cns->null())
	return 0;
    else
	return (cns->cdr())->reverse()->append(Cons<Int>::cons0(cns->car()));
}

Ptr<Cons<Int> >
sub()
{
    using namespace	std;
	
    Ptr<Cons<Int> >	list = 0;
    int			n;

    while (1)
    {
	cerr << "Input > " << flush;
	cin >> n;
	if (!cin)
	    break;
	
	list = list->cons(Int::newInt(n));
	cout << "Current:\t" << list << endl;
    }

    list = list->append(list);
    cout << "Append:\t" << list << endl;
    
    list = list->reverse();
    cout << "Reverse:\t" << list << endl;

    list = my_reverse(list);
    cout << "MyReverse:\t" << list << endl;

    list = my_reverse(list);
    cout << "MyReverse:\t" << list << endl;

    list = list->append(list);
    cout << "Append:\t" << list << endl;

    list = list->reverse();
    cout << "Reverse:\t" << list << endl;

    return list;
}
 
}

#include <fstream>

int
main()
{
    using namespace	std;
    using namespace	TU;
	
    Ptr<Cons<Int> >	list = sub();
    
    std::ofstream out("tmp.dat", ios::out);
    list->save(out);
    out.close();

    list = list->append(list);
    cout << "Append:\t" << list << endl;

    std::ifstream in("tmp.dat", ios::in);
    list = Cons<Int>::restore(in);
//    if (in.eof())
//	cout << "End of File!" << endl;
    in.close();
    cout << "Restored:\t" << list << endl;
    Ptr<Cons<Int> > list2 = list->copy()->nreverse();
    cout << "Original:\t" << list  << endl;
    cout << "Clone:\t"    << list2 << endl;

    return 0;
}    
