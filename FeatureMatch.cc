/*!
  \file		FeatureMatch.cc
  \author	Toshio UESHIBA
  \brief	クラス TU::FeatureMatch の実装
*/
#include "TU/FeatureMatch.h"

namespace TU
{
/************************************************************************
*  class FeatureMatch::Sampler						*
************************************************************************/
//! 指定された個数の点対応をランダムに取り出す．
/*!
  \param npoints	取り出すべき点対応の個数
  \return		取り出された点対応の並び
*/
FeatureMatch::Sampler::Container
FeatureMatch::Sampler::sample(size_t npoints) const
{
    using namespace	std;

    if (npoints > _size)
	throw runtime_error("Sampler<C>::sample(): not enough point matches!");
    
    Container	samples;
    while (samples.size() != npoints)
    {
	const_iterator	item = _begin;
	advance(item, _random.generateInt32() % _size);
	if (find(samples.begin(), samples.end(), *item) == samples.end())
	    samples.push_back(*item);
    }
    
    return samples;
}

}
