/*
 *  $Id: Object++.cc,v 1.2 2002-07-25 02:38:02 ueshiba Exp $
 */
#include "TU/Object++.h"

namespace TU
{
/************************************************************************
*  class Cons:	cons cell						*
************************************************************************/
template <class T> const Cons<T>*
Cons<T>::nthcdr(int n) const
{
    const Cons*	cns = this;
    while (n--)
	cns = cns->cdr();
    return cns;
}

template <class T> Cons<T>*
Cons<T>::nthcdr(int n)
{
    Cons*	cns = this;
    while (n--)
	cns = cns->cdr();
    return cns;
}

template <class T> const Cons<T>*
Cons<T>::last() const
{
    const Cons*	cns = this;
    while (cns->cdr()->consp())
	cns = cns->cdr();
    return cns;
}

template <class T> Cons<T>*
Cons<T>::last()
{
    Cons*	cns = this;
    while (cns->cdr()->consp())
	cns = cns->cdr();
    return cns;
}

template <class T> int
Cons<T>::length() const
{
    int n = 0;
    for (const Cons* cns = this; cns->consp(); cns = cns->cdr())
	++n;
    return n;
}

template <class T> Ptr<Cons<T> >
Cons<T>::reverse() const
{
    Ptr<Cons>	rev;
    for (const Cons* cns = this; cns->consp(); cns = cns->cdr())
	rev = rev->cons(cns->car());
    return rev;
}

template <class T> Ptr<Cons<T> >
Cons<T>::append(Cons* cns) const
{
    if (!consp())
	return cns;
    else
	return cdr()->append(cns)->cons(car());
}

template <class T> const Cons<T>*
Cons<T>::member(const T* item) const
{
    for (const Cons* cns = this; cns->consp(); cns = cns->cdr())
	if (cns->car() == item)
	    return cns;
    return 0;
}

template <class T> Cons<T>*
Cons<T>::member(const T* item)
{
    for (Cons* cns = this; cns->consp(); cns = cns->cdr())
	if (cns->car() == item)
	    return cns;
    return 0;
}

template <class T> Ptr<Cons<T> >
Cons<T>::remove(const T* item) const
{
    Ptr<Cons>	rmv;
    for (const Cons* cns = this; cns->consp(); cns = cns->cdr())
	if (cns->car() != item)
	    rmv = rmv->cons(cns->car());
    return rmv->nreverse();
}

template <class T> Ptr<Cons<T> >
Cons<T>::nreverse()
{
    Cons	*cns = this, *prev = 0, *rev = 0;
    while (cns->consp())
    {
	rev  = cns;
	cns  = cns->cdr();
	prev = rev->rplacd(prev);
    }
    return rev;
}

template <class T> Cons<T>*
Cons<T>::nconc(Cons* cns)
{
    if (null())
	return cns;
    else
    {
	(void)last()->rplacd(cns);
	return this;
    }
}

template <class T> Cons<T>*
Cons<T>::detach(const T* item)
{
    Cons	*head = this, *prev = 0;

    for (Cons* cns = this; consp(); cns = cns->cdr())
	if (cns->car() == item)
	    if (prev->null())
		head = cns->cdr();
	    else
		(void)prev->rplacd(cns->cdr());
	else
	    prev = cns;
    return head;
}
 
}
