/*
 *  $BJ?@.(B9-19$BG/!JFH!K;:6H5;=QAm9g8&5f=j(B $BCx:n8"=jM-(B
 *  
 *  $BAO:n<T!'?"<G=SIW(B
 *
 *  $BK\%W%m%0%i%`$O!JFH!K;:6H5;=QAm9g8&5f=j$N?&0w$G$"$k?"<G=SIW$,AO:n$7!$(B
 *  $B!JFH!K;:6H5;=QAm9g8&5f=j$,Cx:n8"$r=jM-$9$kHkL)>pJs$G$9!%AO:n<T$K$h(B
 *  $B$k5v2D$J$7$KK\%W%m%0%i%`$r;HMQ!$J#@=!$2~JQ!$;HMQ!$Bh;0<T$X3+<($9$k(B
 *  $BEy$NCx:n8"$r?/32$9$k9T0Y$r6X;_$7$^$9!%(B
 *  
 *  $B$3$N%W%m%0%i%`$K$h$C$F@8$8$k$$$+$J$kB;32$KBP$7$F$b!$Cx:n8"=jM-<T$*(B
 *  $B$h$SAO:n<T$O@UG$$rIi$$$^$;$s!#(B
 *
 *  Copyright 1997-2007.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  Confidential and all rights reserved.
 *  This program is confidential. Any using, copying, changing, giving
 *  information about the source program of any part of this software
 *  to others without permission by the creators are prohibited.
 *
 *  No Warranty.
 *  Copyright holders or creators are not responsible for any damages
 *  in the use of this program.
 *  
 *  $Id: CmdParent.cc,v 1.3 2007-11-26 08:11:50 ueshiba Exp $
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
