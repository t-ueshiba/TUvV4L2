/*!
  \file		Ransac.h
  \author	Toshio UESHIBA
  \brief	RANSACを行う関数の定義と実装
*/
#ifndef __TU_RANSAC_H
#define __TU_RANSAC_H

#include <vector>
#include <random>
#include <stdexcept>
#include "TU/algorithm.h"

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
  \param inlierRate	[ib, ie)に含まれる点のうちinlierの割合.
			0 < inlierRate < 1でなければならない
  \param hitRate	RANSACによって正しくinlierを引き当てる確率.
			0 <= hitRate < 1でなければならない
  \return		pointSetに含まれるinlier
*/
template <class IN, class MODEL, class CONFORM, class T>
std::vector<typename std::iterator_traits<IN>::value_type>
ransac(IN ib, IN ie,
       MODEL& model, CONFORM&& conform, T inlierRate, T hitRate=0.99)
{
    using point_type	= typename std::iterator_traits<IN>::value_type;
    using points_type	= std::vector<point_type>;

    if (size_t(std::distance(ib, ie)) < model.ndataMin())
	throw std::runtime_error(
		"ransac(): not enough points in the given point set!!");

    if (inlierRate >= 1)	// [ib, ie) が全てinlierなら...
    {
	model.fit(ib, ie);	// [ib, ie) にモデルを当てはめる.
	return std::move(points_type(ib, ie));
    }
    
    if (inlierRate <= 0)
	throw std::invalid_argument(
		"ransac(): given inline rate is not within (0, 1]!!");
    if (hitRate < 0 || hitRate >= 1)
	throw std::invalid_argument(
		"ransac(): given hit rate is not within [0, 1)!!");
    
  // 与えられたhitRate，PointSetに含まれるinlierの割合およびModelの生成に
  // 要する最少点数から，サンプリングの必要回数を求める．
    T		tmp = 1;
    for (auto n = model.ndataMin(); n-- > 0; )
	tmp *= inlierRate;
    const auto	ntrials = size_t(std::ceil(std::log(1 - hitRate) /
					   std::log(1 - tmp)));

  // 試行（最小個数の点をサンプル，モデル生成，inlier検出）をntrials回行う．
    points_type maximalSet;
    for (size_t n = 0; n < ntrials; ++n)
    {
	try
	{
	  // 点集合からモデルの計算に必要な最少個数の点をサンプルする．
	    points_type	minimalSet;
	    std::sample(ib, ie, std::back_inserter(minimalSet),
			model.ndataMin(),
			std::mt19937(std::random_device()()));
	    
	  // サンプルした点からモデルを生成する．
	    model.fit(minimalSet.begin(), minimalSet.end());
	}
	catch (const std::runtime_error& err)	// 当てはめに失敗したら
	{					// (ex. 特異な点配置など)
	    continue;				// 再度サンプルする.
	}
	
      // 全点の中で生成したモデルに適合する(inlier)ものを集める．
	points_type	inliers;
	std::copy_if(ib, ie, std::back_inserter(inliers),
		     [&](const auto& in){ return conform(in, model); });

      // これまでのどのモデルよりもinlierの数が多ければその集合を記録する．
      // なお，サンプルされた点（minimalSetの点）が持つ自由度がモデルの自由度
      // よりも大きい場合は，これらに誤差0でモデルを当てはめられるとは限らない
      // ので，minimalSetの点が全てinliersに含まれる保証はない．よって，
      // inliersのサイズがモデル計算に必要な最少点数以上であることもチェックする．
	if (inliers.size() >= model.ndataMin() &&
	    inliers.size() >  maximalSet.size())
	    maximalSet = std::move(inliers);
    }

  // maximalSetに含まれる点を真のinlierとし，それら全てからモデルを生成する．
    model.fit(maximalSet.begin(), maximalSet.end());

    return std::move(maximalSet);
}
    
}
#endif // !__TU_RANSAC_H
