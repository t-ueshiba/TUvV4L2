/*
 *  $Id: Ransac++.h,v 1.1 2007-02-05 23:24:03 ueshiba Exp $
 */
#ifndef __TURansacPP_h
#define __TURansacPP_h

#include <math.h>
#include <stdexcept>
#include "TU/types.h"

namespace TU
{
/************************************************************************
*  function ransac							*
************************************************************************/
//! RANSACによってoutlierを含む点集合にモデルを当てはめる．
/*!
  テンプレートパラメータPointSetは点集合を表すクラスであり，以下の条件を
  満たすこと：
  \verbatim
  1. forward_iteratorをサポートするコンテナである．
  2. このコンテナの型をPointSet::Containerという名前でtypedefしている．
  3. inlierの割合をメンバ関数
	double	Pointset::inlierRate() const;
     によって知ることができる．
  4. メンバ関数
	PoinstSet::Container	Pointset::samle(u_int npoints) const;
     によってランダムにnpoints個の部分集合を取り出せる．
  \endverbatim
  テンプレートパラメータModelは当てはめるべきモデルを表すクラスであり，
  以下の条件を満たすこと：
  \verbatim
  1. メンバ関数
	template <class Iterator>
	void	Model::initialize(Iterator first, Iterator last);
     によって点集合にモデルを当てはめることができる．
  2. 1.に必要な最少点数をメンバ関数
	u_int	Model::ndataMin() const;
     によって知ることができる．
  \endverbatim
  テンプレートパラメータConformは点のモデルへの適合性を判定する関数
  オブジェクトであり，
  \verbatim
	bool	Conform::operator()(const PointSet::Container::valu_type& p,
				    const Model& model);
  \endverbatim
  なるインタフェースによってpがmodelに適合しているか判定できること．
  \param pointSet	inlierとoutlierを含む点集合
  \param model		pointSetに含まれるinlierを当てはめるモデル．
  \param hitRate	RANSACによって正しくinlierを引き当てる確率．
			0 <= hitRate < 1でなければならない．
  \return		pointSetに含まれるinlier
*/
template <class PointSet, class Model, class Conform>
typename PointSet::Container
ransac(const PointSet& pointSet, Model& model, Conform conform,
       double hitRate=0.99)
{
    typedef typename PointSet::Container	Container;
    
  // 与えられたhitRate，PointSetに含まれるinlierの割合およびModelの生成に
  // 要する最少点数から，サンプリングの必要回数を求める．
    if (hitRate < 0.0 || hitRate >= 1.0)
	throw std::invalid_argument("ransac<PointSet, Model>: given hit rate is not within [0, 1)!!");
    const double	inlierRate = pointSet.inlierRate();
    if (inlierRate < 0.0 || inlierRate >= 1.0)
	throw std::invalid_argument("ransac<PointSet, Model>: inlier rate is not within [0, 1)!!");
    double	tmp = 1.0;
    for (u_int n = model.ndataMin(); n-- > 0; )
	tmp *= inlierRate;
    const u_int	ntrials = u_int(ceil(log(1.0 - hitRate) / log(1.0 - tmp)));

  // 試行（最小個数の点をサンプル，モデル生成，inlier検出）をntrials回行う．
    Container	inlierSetA, inlierSetB;
    Container	*inliers = &inlierSetA, *inliersMax = &inlierSetB;
    for (int n = 0; n < ntrials; ++n)
    {
      // 点集合からモデルの計算に必要な最小個数の点をサンプルする．
	const Container&	minimalSet = pointSet.sample(model.ndataMin());

      // サンプルした点からモデルを生成する．
	model.initialize(minimalSet.begin(), minimalSet.end());

      // 全点の中で生成したモデルに適合する(inlier)ものを集める．
	inliers->clear();
	for (typename PointSet::const_iterator iter = pointSet.begin();
	     iter != pointSet.end(); ++iter)
	    if (conform(*iter, model))
		inliers->push_back(*iter);

      // これまでのどのモデルよりもinlierの数が多ければその集合を記録する．
	if (inliers->size() > inliersMax->size())
	    std::swap(inliers, inliersMax);
    }
  // 最大集合に含まれる点を真のinlierとし，それら全てからモデルを生成する．
    model.initialize(inliersMax->begin(), inliersMax->end());

    return *inliersMax;
}
    
}

#endif // !__TURansacPP_h
