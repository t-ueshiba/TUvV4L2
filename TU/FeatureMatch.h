/*
 *  平成21-22年（独）産業技術総合研究所 著作権所有
 *  
 *  創作者：植芝俊夫
 *
 *  本プログラムは（独）産業技術総合研究所の職員である植芝俊夫が創作し，
 *  （独）産業技術総合研究所が著作権を所有する秘密情報です．著作権所有
 *  者による許可なしに本プログラムを使用，複製，改変，第三者へ開示する
 *  等の行為を禁止します．
 *  
 *  このプログラムによって生じるいかなる損害に対しても，著作権所有者お
 *  よび創作者は責任を負いません。
 *
 *  Copyright 2009-2010.
 *  National Institute of Advanced Industrial Science and Technology (AIST)
 *
 *  Creator: Toshio UESHIBA
 *
 *  [AIST Confidential and all rights reserved.]
 *  This program is confidential. Any using, copying, changing or
 *  giving any information concerning with this program to others
 *  without permission by the copyright holder are strictly prohibited.
 *
 *  [No Warranty.]
 *  The copyright holder or the creator are not responsible for any
 *  damages caused by using this program.
 *
 *  $Id$
 */
#ifndef __TU_FEATUREMATCH_H
#define __TU_FEATUREMATCH_H

#include <utility>
#include <vector>
#include "TU/Random.h"
#include "TU/Ransac.h"
#include "TU/Feature.h"

namespace TU
{
/************************************************************************
*  class FeatureMatch							*
************************************************************************/
class FeatureMatch
{
  public:
    typedef double			value_type;
    typedef std::pair<Point2f, Point2f>	Match;		//!< 2画像間の点対応
    typedef std::vector<Match>		MatchSet;	//!< 点対応の集合

    struct Parameters
    {
	Parameters()
	    :separation(0.85),
	     diffAngleMax(15.0*M_PI/180.0),
	     inlierRate(0.1),
	     conformThresh(4.0),
	     nmatchesMin(6)				{}

	value_type	separation;	//!< 1位のマッチングの2位に対する割合
	value_type	diffAngleMax;	//!< 2つの特徴がマッチできる最大角度差
	value_type	inlierRate;	//!< マッチング候補内のインライアの割合
	value_type	conformThresh;	//!< インライアとなる最大当てはめ誤差
	size_t		nmatchesMin;	//!< 画像間対応に必要な最小マッチング数
    };

  private:
    class Sampler
    {
      public:
	typedef MatchSet			Container;
	typedef MatchSet::const_iterator	const_iterator;

	Sampler(const_iterator begin, const_iterator end, double inlierRate)
	    :_begin(begin), _end(end),
	     _size(std::distance(_begin, _end)),
	     _inlierRate(inlierRate), _random()		{}

	value_type	inlierRate()		const	{ return _inlierRate; }
	const_iterator	begin()			const	{ return _begin; }
	const_iterator	end()			const	{ return _end; }
	size_t		size()			const	{ return _size; }
	Container	sample(size_t npoints)	const	;
    
      private:
	const const_iterator	_begin;		//!< 点対応列の先頭
	const const_iterator	_end;		//!< 点対応列の末尾の次
	const size_t		_size;		//!< 点対応の総数
	const value_type	_inlierRate;	//!< inlierの割合
	mutable Random		_random;	//!< 乱数発生器
    };
	
    template <class MAP>
    class Conform : public std::binary_function<Match, MAP, bool>
    {
      public:
	Conform(value_type thresh)    :_sqThresh(thresh * thresh)	{}

	bool	operator ()(const Match& match, const MAP& map)	const
		{
		    return map.sqdist(match) < _sqThresh;
		}
    
      private:
	const value_type	_sqThresh;	//!< 適合判定のしきい値の二乗
    };
    
  public:
    FeatureMatch()				:_params()		{}
    FeatureMatch(const Parameters& params)	:_params(params)	{}
    
    FeatureMatch&	setParameters(const Parameters& parameters)	;
    const Parameters&	getParameters()				const	;
    
    template <class MAP, class ITER>
    MatchSet		operator ()(MAP& map,
				    ITER begin0, ITER end0,
				    ITER begin1, ITER end1)	const	;

  private:
    template <class ITER>
    MatchSet	findCandidateMatches(ITER begin0, ITER end0,
				     ITER begin1, ITER end1)	const	;
    template <class ITER>
    bool	findBestMatch(
		    ITER feature, ITER& feature_best,
		    const std::vector<std::vector<ITER> >& buckets) const;
    template <class T>
    static T	fraction(T angle, size_t size)
		{
		    return (angle / (2.0*M_PI)) * size;
		}
    
  private:
    Parameters	_params;
};

inline FeatureMatch&
FeatureMatch::setParameters(const Parameters& params)
{
    _params = params;
    return *this;
}

inline const FeatureMatch::Parameters&
FeatureMatch::getParameters() const
{
    return _params;
}

template <class MAP, class ITER> FeatureMatch::MatchSet
FeatureMatch::operator ()(MAP& map,
			  ITER begin0, ITER end0, ITER begin1, ITER end1) const
{
    using namespace	std;
    
  // 特徴間の全対応候補を求める．
    MatchSet	candidates = findCandidateMatches(begin0, end0,
						  begin1, end1);

  // RANSACによって誤対応を除去するとともに，画像間変換を求める．
    Sampler	sampler(candidates.begin(),
			candidates.end(), _params.inlierRate);
    MatchSet	matchSet = ransac(sampler, map,
				  Conform<MAP>(_params.conformThresh));
    cerr << setw(3) << matchSet.size() << " matches selected from "
	 << candidates.size() << " candidates." << endl;

    return matchSet;
}

//! 2枚の画像から取り出した特徴を用いて双方向探索により対応点候補をみつける．
/*!
  \param begin0		一方の画像の特徴の先頭
  \param end0		一方の画像の特徴の末尾の次
  \param begin1		もう一方の画像の特徴の先頭
  \param end1		もう一方の画像の特徴の末尾の次
  \return
*/
template <class ITER> FeatureMatch::MatchSet
FeatureMatch::findCandidateMatches(ITER begin0, ITER end0,
				   ITER begin1, ITER end1) const
{
    using namespace	std;

  // [begin0, end0), [begin1, end1) を angle によって分類する．
    const size_t		nbuckets = 360;
    vector<vector<ITER> >	buckets0(nbuckets);
    vector<vector<ITER> >	buckets1(nbuckets);
    for (ITER feature0 = begin0; feature0 != end0; ++feature0)
    {
	int	i = int(fraction(feature0->angle, buckets0.size()));

	buckets0[i].push_back(feature0);
    }
    for (ITER feature1 = begin1; feature1 != end1; ++feature1)
    {
	int	i = int(fraction(feature1->angle, buckets1.size()));

	buckets1[i].push_back(feature1);
    }

  // [begin0, end0)の各特徴について特徴記述子間の距離をもとに対応候補を検出する．
    MatchSet	candidateMatches;
    for (ITER feature0 = begin0; feature0 != end0; ++feature0)
    {
      // feature0 に対する最良の対応を探す．
	ITER	feature_best1;
	if (findBestMatch(feature0, feature_best1, buckets1))
	{
	  // 逆方向もチェックして相思相愛だけを残す
	    ITER	feature_best0;
	    if (findBestMatch(feature_best1, feature_best0, buckets0) &&
		feature0 == feature_best0)
	    {
		candidateMatches.push_back(make_pair(*feature0,
						     *feature_best1));
	    }
	}
    }

    return candidateMatches;
}

//! 指定された特徴に最も良く対応する特徴を指定されたバケット群の中から探索する．
/*!
  \param feature	特徴
  \param feature_best	featureに対する最良の特徴が返される
  \param buckets	対応相手となる特徴を格納したバケット群
  \return		最良のマッチングのスコアが2位のマッチングのスコアの
			separation倍未満ならばtrue, そうでなければfalse
*/
template <class ITER> bool
FeatureMatch::findBestMatch(
		  ITER feature, ITER& feature_best,
		  const std::vector<std::vector<ITER> >& buckets) const
{
    using namespace	std;

    typedef typename iterator_traits<ITER>::value_type	feature_type;
    typedef typename feature_type::value_type		value_type;
    typedef vector<ITER>				bucket_t;
    
  // 該当区間を検索する．
    value_type	sqd_best   = numeric_limits<value_type>::max(),
		sqd_second = numeric_limits<value_type>::max();
    const int	idx	   = int(fraction(feature->angle, buckets.size())),
		range	   = int(ceil(fraction(_params.diffAngleMax,
					       buckets.size())));
    for (int i = idx - range; i <= idx + range; ++i)
    {
	const int	j = (i < 0		 ? i + buckets.size() :
			     i >= buckets.size() ? i - buckets.size() : i);
	const bucket_t&	bucket = buckets[j];
	
	for (typename bucket_t::const_iterator
		 iter = bucket.begin(); iter !=  bucket.end(); ++iter)
	{
	    value_type	sqd = feature->sqdistOfFeature(**iter);

	    if (sqd < sqd_best)
	    {
		sqd_second   = sqd_best;
		sqd_best     = sqd;
		feature_best = *iter;
	    }
	    else if (sqd < sqd_second)
	    {
		sqd_second = sqd;
	    }
	}
    }

    return (sqd_best < _params.separation * _params.separation * sqd_second);
}

/************************************************************************
*  global functions							*
************************************************************************/
//! 入力ストリームから2枚の画像間の点対応を読み込む(ASCII)．
/*!
  \param in	入力ストリーム
  \param match	点対応
  \return	inで指定した入力ストリーム
*/
inline std::istream&
operator >>(std::istream& in, FeatureMatch::Match& match)
{
    char	c;
    return in >> c >> match.first[0]  >> c >> match.first[1]
	      >> c >> match.second[0] >> c >> match.second[1] >> skipl;
}
    
//! 出力ストリームに2枚の画像間の点対応を書き出す(ASCII)．
/*!
  \param out	出力ストリーム
  \param match	点対応
  \return	outで指定した出力ストリーム
*/
inline std::ostream&
operator <<(std::ostream& out, const FeatureMatch::Match& match)
{
    return out << " x" << match.first[0]  << " y" << match.first[1]
	       << " X" << match.second[0] << " Y" << match.second[1]
	       << " t0"
	       << std::endl;
}
    
}
#endif	// !__TU_FEATUREMATCH_H
