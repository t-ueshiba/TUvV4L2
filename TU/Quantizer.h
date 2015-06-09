/*
 *  $Id$
 */
#ifndef __TU_QUANTIZER_H
#define __TU_QUANTIZER_H

#include <vector>
#include "TU/Array++.h"

namespace TU
{
/************************************************************************
*  class QuantizerBase<T>						*
************************************************************************/
template <class T>
class QuantizerBase
{
  public:
    typedef T	value_type;
    
  public:
    const T&	operator [](size_t i)	const	{ return _bins[i]; }
    size_t	size()			const	{ return _bins.size(); }

  protected:
    template <class PAIR>
    void	quantize(std::vector<PAIR>& in_out, size_t nbins)	;
	    
  private:
    std::vector<T>	_bins;
};

template <>
class QuantizerBase<u_char>
{
  public:
    size_t	operator [](size_t i)	const	{ return i; }
    size_t	size()			const	{ return 256; }
};
    
template <class T> template <class PAIR> void
QuantizerBase<T>::quantize(std::vector<PAIR>& io, size_t nbins)
{
    std::sort(io.begin(), io.end(),
	      [](const PAIR& x, const PAIR& y){return *x.first < *y.first;});
    
  // 入力データをnbin個のbinに量子化した場合のbin幅の最大値を求める
    T	binWidthMax = 2*(*io.back().first - *io.front().first)/nbins;
    for (T th = 1e-5, l = 0; binWidthMax - l > th; )
    {
	T	binMin = *io.front().first;	// bin中の最小値
	T	m = 0.5*(l + binWidthMax);
	size_t	n = 0;
	bool	ok = true;

	for (const auto& x : io)
	    if (*x.first > binMin + m)	// bin中の最大値を越えるなら...
	    {
		if (++n == nbins)	// 新たにbinを作ると
		{			// bin数の上限を越えるなら...
		    ok = false;		// bin幅が不十分
		    break;
		}
		binMin = *x.first;	// 新たなbin中の最小値
	    }

	if (ok)			// 現在のmの値でbin幅が十分ならば...
	    binWidthMax = m;	// mを新たな上限値に
	else			// そうでなければ...
	    l = m;		// mを新たな下限値に
    }

  // 入力データの各元をbinの代表元にマップ
    _bins.clear();
    auto	binBase = io.cbegin();
    size_t	idx = 0;
    for (auto x = io.cbegin(); x != io.cend(); ++x)
	if (*x->first > *binBase->first + binWidthMax)
	{
	    _bins.push_back(*binBase[(x - binBase) >> 1].first);  // 代表元
			    
	    for (; binBase != x; ++binBase)
		*binBase->second = idx;
	    ++idx;
	}
    _bins.push_back(*binBase[(io.cend() - binBase) >> 1].first);
    for (; binBase != io.cend(); ++binBase)
	*binBase->second = idx;
}

/************************************************************************
*  class Quantizer<T>							*
************************************************************************/
template <class T>
class Quantizer : public QuantizerBase<T>
{
  public:
    template <class ITER>
    typename std::enable_if<
        std::is_same<
	    typename std::iterator_traits<ITER>::value_type, u_char>::value,
	range<ITER> >::type
		operator ()(ITER ib, ITER ie, size_t)
		{
		    return range<ITER>(ib, ie);
		}
    template <class ITER>
    typename std::enable_if<
        !std::is_same<
	    typename std::iterator_traits<ITER>::value_type, u_char>::value,
	const Array<size_t>&>::type
		operator ()(ITER ib, ITER ie, size_t nbins)	;
    friend std::ostream&
		operator <<(std::ostream& out, const Quantizer<T>& quantizer)
		{
		    return out << quantizer._indices;
		}
    
  private:
    Array<size_t>	_indices;
};

template <class T> template <class ITER>
typename std::enable_if<
    !std::is_same<
	typename std::iterator_traits<ITER>::value_type, u_char>::value,
    const Array<size_t>&>::type
Quantizer<T>::operator ()(ITER ib, ITER ie, size_t nbins)
{
    typedef std::pair<ITER, Array<size_t>::iterator>	pair_type;

    _indices.resize(std::distance(ib, ie));
    
    std::vector<pair_type>	io;
    for (auto idx = _indices.begin(); ib != ie; ++ib, ++idx)
	io.push_back(pair_type(ib, idx));

    this->quantize(io, nbins);

    return _indices;
}
    
/************************************************************************
*  class Quantizer2<T>							*
************************************************************************/
template <class T>
class Quantizer2 : public QuantizerBase<T>
{
  public:
    template <class ROW>
    typename std::enable_if<
	std::is_same<
	    typename std::iterator_traits<subiterator<ROW> >::value_type,
	    u_char>::value,
	range<ROW> >::type
		operator ()(ROW ib, ROW ie, size_t nbins)
		{
		    return range<ROW>(ib, ie);
		}
    template <class ROW>
    typename std::enable_if<
	!std::is_same<
	    typename std::iterator_traits<subiterator<ROW> >::value_type,
	    u_char>::value,
	const Array2<Array<size_t> >&>::type
		operator ()(ROW ib, ROW ie, size_t nbins)	;
    friend std::ostream&
		operator <<(std::ostream& out, const Quantizer2<T>& quantizer)
		{
		    return out << quantizer._indices;
		}
	
  private:
    Array2<Array<size_t> >	_indices;
};

template <class T> template <class ROW>
typename std::enable_if<
    !std::is_same<
	typename std::iterator_traits<subiterator<ROW> >::value_type,
	u_char>::value,
    const Array2<Array<size_t> >&>::type
Quantizer2<T>::operator ()(ROW ib, ROW ie, size_t nbins)
{
    typedef std::pair<subiterator<ROW>,
		      Array<size_t>::iterator>	pair_type;
    
    _indices.resize(std::distance(ib, ie),
		    (ib == ie ? 0 : std::distance(ib->begin(), ib->end())));
    
    std::vector<pair_type>	io;
    for (auto row = _indices.begin(); ib != ie; ++ib, ++row)
    {
	auto	idx = row->begin();
	for (auto col = ib->begin(); col != ib->end(); ++col, ++idx)
	    io.push_back(pair_type(col, idx));
    }

    this->quantize(io, nbins);

    return _indices;
}

}
#endif	// !__TU_QUANTIZER_H
