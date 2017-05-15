/*!
  \file		Ransac.h
  \author	Toshio UESHIBA
  \brief	RANSACを行う関数の定義と実装
*/
#ifndef __TU_RANSAC_H
#define __TU_RANSAC_H

#include <math.h>
#include <stdexcept>

namespace TU
{
/************************************************************************
*  function ransac							*
************************************************************************/
//! RANSACによってoutlierを含む点集合にモデルを当てはめる．
/*!
  テンプレートパラメータPointSetは点集合を表すクラスであり，以下の条件を
  満たすこと：
  -# forward_iteratorをサポートするコンテナである．
  -# このコンテナの型を
	PointSet::Container
     という名前でtypedefしている．
  -# inlierの割合をメンバ関数
	double	Pointset::inlierRate() const;
     によって知ることができる．
  -# メンバ関数
	PoinstSet::Container	Pointset::sample(size_t npoints) const;
     によってランダムにnpoints個の部分集合を取り出せる．

  テンプレートパラメータModelは当てはめるべきモデルを表すクラスであり，
  以下の条件を満たすこと：
  -# メンバ関数
	template <class Iterator>
	void	Model::fit(Iterator first, Iterator last);
     によって点集合にモデルを当てはめることができる．
  -# 1.に必要な最少点数をメンバ関数
	size_t	Model::ndataMin() const;
     によって知ることができる．

  テンプレートパラメータConformは点のモデルへの適合性を判定する関数
  オブジェクトであり，
	bool	Conform::operator()(const PointSet::Container::valu_type& p,
				    const Model& model);
  なるインタフェースによってpがmodelに適合しているか判定できること．

  \param pointSet	inlierとoutlierを含む点集合
  \param model		pointSetに含まれるinlierを当てはめるモデル
  \param conform	点のモデルへの適合性を判定する関数オブジェクト
  \param hitRate	RANSACによって正しくinlierを引き当てる確率.
			0 <= hitRate < 1でなければならない
  \return		pointSetに含まれるinlier
*/
template <class PointSet, class Model, class Conform>
typename PointSet::Container
ransac(const PointSet& pointSet, Model& model, Conform conform,
       double hitRate=0.99)
{
    typedef typename PointSet::Container	Container;

    if (std::distance(pointSet.begin(), pointSet.end()) < model.ndataMin())
	throw std::runtime_error("ransac<PointSet, Model>: not enough points in the given point set!!");
    
  // 与えられたhitRate，PointSetに含まれるinlierの割合およびModelの生成に
  // 要する最少点数から，サンプリングの必要回数を求める．
    if (hitRate < 0.0 || hitRate >= 1.0)
	throw std::invalid_argument("ransac<PointSet, Model>: given hit rate is not within [0, 1)!!");
    const double	inlierRate = pointSet.inlierRate();
    if (inlierRate < 0.0 || inlierRate >= 1.0)
	throw std::invalid_argument("ransac<PointSet, Model>: inlier rate is not within [0, 1)!!");
    double	tmp = 1.0;
    for (size_t n = model.ndataMin(); n-- > 0; )
	tmp *= inlierRate;
    const size_t	ntrials = size_t(ceil(log(1.0 - hitRate) /
					      log(1.0 - tmp)));

  // 試行（最小個数の点をサンプル，モデル生成，inlier検出）をntrials回行う．
    Container	inlierSetA, inlierSetB;
    Container	*inliers = &inlierSetA, *inliersMax = &inlierSetB;
    for (size_t n = 0; n < ntrials; ++n)
    {
      // 点集合からモデルの計算に必要な最小個数の点をサンプルする．
	const Container&	minimalSet = pointSet.sample(model.ndataMin());

      // サンプルした点からモデルを生成する．
	try
	{
	    model.fit(minimalSet.begin(), minimalSet.end());
	}
	catch (const std::runtime_error& err)	// 当てはめに失敗したら
	{					// (ex. 特異な点配置など)
	    continue;				// 再度サンプルする.
	}
	
      // 全点の中で生成したモデルに適合する(inlier)ものを集める．
	inliers->clear();
	for (typename PointSet::const_iterator iter = pointSet.begin();
	     iter != pointSet.end(); ++iter)
	    if (conform(*iter, model))
		inliers->push_back(*iter);

      // これまでのどのモデルよりもinlierの数が多ければその集合を記録する．
	if (inliers->size() > inliersMax->size())
	    swap(inliers, inliersMax);
    }
  // 最大集合に含まれる点を真のinlierとし，それら全てからモデルを生成する．
  // サンプルされた点（minimalSetに含まれる点）が持つ自由度がモデルの自由度
  // よりも大きい場合は，これらに誤差0でモデルを当てはめられるとは限らないので，
  // これらの点が必ずinlierに含まれる保証はない．よって，inlierが足りなくて
  // 次の当てはめが失敗することもあり得る．
    model.fit(inliersMax->begin(), inliersMax->end());

    return *inliersMax;
}
    
}
#endif // !__TU_RANSAC_H
