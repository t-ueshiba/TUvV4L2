#include "TU/Object++.h"

namespace TU
{
const unsigned	id_Int = 256;

class Int : public Object
{
  public:
    static Ptr<Int>	new_Int(int i)		{return new Int(i);}
    int			value()		const	{return val;}

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

inline std::ostream&
operator <<(std::ostream& out, const Int* p)
{
    return out << p->value() << " ";
}

inline std::ostream&
operator <<(std::ostream& out, const Int& p)
{
    return out << p.value() << " ";
}
 
}
