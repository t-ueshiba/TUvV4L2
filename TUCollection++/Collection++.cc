/*
 *  $Id$
 */
#include "TU/Collection++.h"

namespace TU
{
/************************************************************************
*  class ObjList							*
************************************************************************/
template <class T> ObjList<T>*
ObjList<T>::findnode(T* p) const
{
    ObjList<T>* const	head = (ObjList<T>*)this;
    T* const		tmp  = head->_p;	
    head->_p = p;
    ObjList<T>* node = head;
    for (; node->_next->_p != p; node = node->_next);
    head->_p = tmp;
    return (node->_next == head ? 0 : node);
}

/************************************************************************
*  class ObjDList							*
************************************************************************/
template <class T> ObjDList<T>*
ObjDList<T>::findnode(T* p) const
{
    ObjDList<T>* const	head = (ObjDList<T>*)this;
    T* const		tmp  = head->_p;	
    head->_p = p;
    ObjDList<T>* node = head;
    for (; (node = node->_next)->_p != p; );
    head->_p = tmp;
    return (node->_next == head ? 0 : node);
}
 
}
