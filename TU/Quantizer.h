/*
 *  $Id$
 */
#ifndef __TU_QUANTIZER_H
#define __TU_QUANTIZER_H

#include <vector>
#include "TU/Vector++.h"

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

  protected:
    typedef typename std::is_floating_point<T>::type	is_floating_point;

  private:
    template <class PITER>
    class BinProps
    {
      public:
	typedef Vector3f			vector_type;
	
      public:
	BinProps(PITER begin, PITER end)	;

	PITER		begin()		const	{ return _begin; }
	PITER		end()		const	{ return _end; }
	T		mean() const
			{
			    return T(u_char(_mean[0]),
				     u_char(_mean[1]),
				     u_char(_mean[2]));
			}
	float		maxVariance() const
			{
			    return std::max({_variance[0],
					     _variance[1],
					     _variance[2]});
			}
	BinProps	split()			;
	
      private:
	static float		square(float x)	{ return x*x; }
	static vector_type	square(const vector_type& x)
				{
				    return {x[0]*x[0], x[1]*x[1], x[2]*x[2]};
				}
	
      private:
	PITER		_begin;
	PITER		_end;
	size_t		_n;
	vector_type	_mean;
	vector_type	_variance;
    };
    
  public:
    const T&	operator [](size_t i)	const	{ return _bins[i]; }
    size_t	size()			const	{ return _bins.size(); }

  protected:
    template <class PAIR>
    void	quantize(std::vector<PAIR>& in_out, size_t nbins,
			 std::true_type)				;
    template <class PAIR>
    void	quantize(std::vector<PAIR>& in_out, size_t nbins,
			 std::false_type)				;
	    
  private:
    std::vector<T>	_bins;
};

template <class T> template <class PAIR> void
QuantizerBase<T>::quantize(std::vector<PAIR>& io, size_t nbins, std::true_type)
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

template <class T> template <class PAIR> void
QuantizerBase<T>::quantize(std::vector<PAIR>& io, size_t nbins, std::false_type)
{
    typedef typename std::vector<PAIR>::iterator	piterator;
    typedef BinProps<piterator>				props_type;

    const auto	comp = [](const props_type& x, const props_type& y)
			{ return x.maxVariance() < y.maxVariance(); };
	
    std::vector<props_type>	props;
    props.push_back(props_type(io.begin(), io.end()));
    std::make_heap(props.begin(), props.end(), comp);

    for (size_t n = 1; n < nbins; ++n)
    {
	std::pop_heap(props.begin(), props.end(), comp);
	auto	prop1 = props.back();
	props.pop_back();

	auto	prop2 = prop1.split();

	props.push_back(prop1);
	std::push_heap(props.begin(), props.end(), comp);
	props.push_back(prop2);
	std::push_heap(props.begin(), props.end(), comp);
    }

    for (size_t idx = 0; idx < props.size(); ++idx)
    {
	_bins.push_back(props[idx].mean());
	for (auto& x: props[idx])
	    *x.second = idx;
    }
}

/*
 *  QuantizerBase<T>::BinProps<PITER>
 */
template <class T> template <class PITER>
QuantizerBase<T>::BinProps<PITER>::BinProps(PITER begin, PITER end)
    :_begin(begin), _end(end), _n(0), _mean(), _variance()
{
    for (auto iter = _begin; iter != _end; ++iter)
    {
	vector_type	x;
	x[0] = iter->first->r;
	x[1] = iter->first->g;
	x[2] = iter->first->b;
	
	++_n;
	_mean     += x;
	_variance += square(x);
    }

    _mean /= _n;
    (_variance /= _n) -= square(_mean);
}
    
template <class T>
template <class PITER> typename QuantizerBase<T>::template BinProps<PITER>
QuantizerBase<T>::BinProps<PITER>::split()
{
    typedef typename std::iterator_traits<PITER>::value_type	pair_type;

  // 要素への反復子の配列を作り，昇順にソート
    const u_char T::*	c = (_variance[0] > _variance[1] ?
			     _variance[0] > _variance[2] ? &T::r : &T::b :
			     _variance[1] > _variance[2] ? &T::g : &T::b);
    std::sort(_begin, _end,
	      [=](const pair_type& x, const pair_type& y)
	      { return (x.first)->*c < (y.first)->*c; });
    
  // 大津の判別基準により最適なしきい値を決定．
    const size_t	i = (_variance[0] > _variance[1] ?
			     _variance[0] > _variance[2] ? 0 : 2 :
			     _variance[1] > _variance[2] ? 1 : 2);
    auto		border = _begin;
    size_t		m = 0;			// しきい値以下の要素数
    vector_type		sum;			// しきい値以下の累積値
    vector_type		sqrsum;			// しきい値以下の自乗累積値
    size_t		n1 = 0;
    vector_type		mean1;
    vector_type		variance1;
    float		interVarianceMax = 0;	// クラス間分散の最大値
    for (auto iter = _begin, head = iter; iter != _end; ++iter)
    {
	vector_type	x;
	x[0] = iter->first->r;
	x[1] = iter->first->g;
	x[2] = iter->first->b;
	
	if (iter->first->*c != head->first->*c)
	{
	    const auto	interVariance = square(sum[i] - m*_mean[i])
				      / (m*(_n - m));
	    if (interVariance > interVarianceMax)
	    {
		interVarianceMax = interVariance;
		border		 = iter;
		n1		 = m;
		mean1		 = sum/n1;
		variance1	 = sqrsum/n1 - square(mean1);
		
	    }
	    head = iter;
	}
	
	++m;
	sum    += x;
	sqrsum += square(x);
    }

    BinProps	props(border, _end);
    
    _end      = border;
    _n	      = n1;
    _mean     = mean1;
    _variance = variance1;
    
    return props;
}
    
/*
 *  QuantizerBase<u_char> : specialized
 */
template <>
class QuantizerBase<u_char>
{
  public:
    size_t	operator [](size_t i)	const	{ return i; }
    size_t	size()			const	{ return 256; }
};
    
/************************************************************************
*  class Quantizer<T>							*
************************************************************************/
template <class T>
class Quantizer : public QuantizerBase<T>
{
  private:
    typedef QuantizerBase<T>	super;
    
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

    super::quantize(io, nbins, typename super::is_floating_point());

    return _indices;
}
    
/************************************************************************
*  class Quantizer2<T>							*
************************************************************************/
template <class T>
class Quantizer2 : public QuantizerBase<T>
{
  private:
    typedef QuantizerBase<T>	super;
    
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

    super::quantize(io, nbins, typename super::is_floating_point());

    return _indices;
}

}
#endif	// !__TU_QUANTIZER_H
