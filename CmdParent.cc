/*
 *  $Id: CmdParent.cc,v 1.1.1.1 2002-07-25 02:14:17 ueshiba Exp $
 */
#include "TU/v/TUv++.h"
#include <functional>
#include <algorithm>
#include <stdexcept>

namespace TU
{
namespace v
{
/************************************************************************
*  function objects							*
************************************************************************/
struct IdEqualTo : public std::binary_function<Cmd, CmdId, bool>
{
    bool	operator ()(const Cmd& vcmd, CmdId id) const
			{return vcmd.id() == id;}
};

struct ValueIsNotZero : public std::unary_function<Cmd, bool>
{
    bool	operator ()(const Cmd& vcmd) const
			{return vcmd.getValue() != 0;}
};

/************************************************************************
*  class CmdParent							*
************************************************************************/
/*
 *  public member functions.
 */
CmdVal
CmdParent::getValue(CmdId id) const
{
    const Cmd*	vcmd = findDescendant(id);
    if (vcmd == 0)
	throw std::domain_error("TU::v::CmdParent::getValue: command not found!!");
    return vcmd->getValue();
}

void
CmdParent::setValue(CmdId id, CmdVal val)
{
    Cmd*	vcmd = findDescendant(id);
    if (vcmd == 0)
	throw std::domain_error("TU::v::CmdParent::setValue: command not found!!");
    vcmd->setValue(val);
}

const char*
CmdParent::getString(CmdId id) const
{
    const Cmd*	vcmd = findDescendant(id);
    if (vcmd == 0)
	throw std::domain_error("TU::v::CmdParent::getString: command not found!!");
    return vcmd->getString();
}

void
CmdParent::setString(CmdId id, const char* str)
{
    Cmd*	vcmd = findDescendant(id);
    if (vcmd == 0)
	throw std::domain_error("TU::v::CmdParent::setString: command not found!!");
    vcmd->setString(str);
}

void
CmdParent::setProp(CmdId id, void* prop)
{
    Cmd*	vcmd = findDescendant(id);
    if (vcmd == 0)
	throw std::domain_error("TU::v::CmdParent::setProp: command not found!!");
    vcmd->setProp(prop);
}

/*
 *  protected member functions.
 */
const Cmd*
CmdParent::findChild(CmdId id) const
{
    using namespace std;
    
    List<Cmd>::ConstIterator where = find_if(_cmdList.begin(), _cmdList.end(),
					     bind2nd(IdEqualTo(), id));
    return (where != _cmdList.end() ? where.operator ->() : 0);
}

Cmd*
CmdParent::findChild(CmdId id)
{
    using namespace std;
    
    List<Cmd>::Iterator where = find_if(_cmdList.begin(), _cmdList.end(),
					bind2nd(IdEqualTo(), id));
    return (where != _cmdList.end() ? where.operator ->() : 0);
}

const Cmd*
CmdParent::findChildWithNonZeroValue() const
{
    using namespace std;
    
    List<Cmd>::ConstIterator where = find_if(_cmdList.begin(), _cmdList.end(),
					     ValueIsNotZero());
    return (where != _cmdList.end() ? where.operator ->() : 0);
}

/*
 *  private member functions.
 */
const Cmd*
CmdParent::findDescendant(CmdId id) const
{
    const Cmd*	vcmd = findChild(id);

    if (vcmd != 0)
	return vcmd;

    for (List<Cmd>::ConstIterator iter = _cmdList.begin();
	 iter != _cmdList.end(); ++iter)
	if ((vcmd = iter->findDescendant(id)) != 0)
	    return vcmd;

    return 0;
}

Cmd*
CmdParent::findDescendant(CmdId id)
{
    Cmd*	vcmd = findChild(id);

    if (vcmd != 0)
	return vcmd;

    for (List<Cmd>::Iterator iter = _cmdList.begin();
	 iter != _cmdList.end(); ++iter)
	if ((vcmd = iter->findDescendant(id)) != 0)
	    return vcmd;

    return 0;
}

}
}
