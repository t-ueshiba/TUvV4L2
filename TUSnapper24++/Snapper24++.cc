/*
 *  $Id: Snapper24++.cc,v 1.2 2002-07-25 02:38:03 ueshiba Exp $
 */
#include "TU/Snapper24++.h"
#include <cstdlib>
#include <stdexcept>

namespace TU
{
/************************************************************************
*  class Snapper24Image<T>						*
************************************************************************/
template <class T>
Snapper24Image<T>::~Snapper24Image()
{
    ::free((T*)*this);
}

template <class T> void
Snapper24Image<T>::resize(u_int height, u_int width)
{
    ::free((T*)*this);
    XilImage<T>::resize(memalign(width*height*sizeof(T)), height, width);
}

template <class T> void
Snapper24Image<T>::resize(T* p, u_int height, u_int width)
{
    throw std::domain_error("TU::Snapper24Image<T>::resize(T*, u_int, u_int): not implemented!");
}

template <class T> T*
Snapper24Image<T>::memalign(size_t size)
{
    T* const		p = (T*)::memalign(64, size);
    if (p == 0)
	throw std::runtime_error("TU::Snapper24Image<T>::memalign: ::memalign failed!!");
    return p;
}
 
}
