/*
 *  $Id$
 */
#ifndef __TU_STEREOITERATORS_H
#define __TU_STEREOITERATORS_H

#include <limits>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class diff_iterator<COL, T>						*
************************************************************************/
template <class COL, class T>
class diff_iterator
    : public boost::iterator_adaptor<diff_iterator<COL, T>,
				     zip_iterator<std::tuple<COL, COL> >,
				     Array<T>,
				     boost::use_default,
				     const Array<T>&>
{
  private:
    using super	= boost::iterator_adaptor<diff_iterator<COL, T>,
					  zip_iterator<std::tuple<COL, COL> >,
					  Array<T>,
					  boost::use_default,
					  const Array<T>&>;
    friend class	boost::iterator_core_access;

  public:
    using	typename super::value_type;
    using	typename super::reference;
    using	typename super::base_type;
    
  public:
		diff_iterator(const base_type& col, size_t dsw, T thresh)
		    :super(col), _P(dsw), _thresh(thresh)		{};
    
  private:
    reference	dereference() const
		{
		    const auto	iter_tuple = super::base().get_iterator_tuple();
		    const auto	colL = *std::get<0>(iter_tuple);
		    auto	colR =  std::get<1>(iter_tuple);
		    for (auto& val : _P)
		    {
			val = std::min(T(diff(colL, *colR)), _thresh);
			++colR;
		    }

		    return _P;
		}
    
  private:
    mutable value_type	_P;
    const T		_thresh;
};

template <class COL, class T> inline diff_iterator<COL, T>
make_diff_iterator(COL colL, COL colR, size_t dsw, T thresh)
{
    return {make_zip_iterator(std::make_tuple(colL, colR)), dsw, thresh};
}

/************************************************************************
*  class MinIdx								*
************************************************************************/
class MinIdx
{
  public:
    MinIdx(size_t disparityMax)	:_disparityMax(disparityMax)	{}

    template <class SCORES_>
    size_t	operator ()(const SCORES_& R) const
		{
		    auto	RminL = std::cbegin(R);
		    for (auto iter = std::cbegin(R); iter != std::cend(R);
			 ++iter)
			if (*iter < *RminL)
			    RminL = iter;
		    return _disparityMax - (RminL - std::cbegin(R));
		}
  private:
    const size_t	_disparityMax;
};

}
#endif	// !__TU_STEREOITERATORS_H
