/*
 *  $Id: V4L2CameraUtility.h 1588 2014-07-10 20:05:03Z ueshiba $
 */
/*!
  \file		V4L2CameraUtility.h
  \brief	クラス TU::V4L2CameraArray の定義と実装
*/
#ifndef __TU_V4L2CAMERAUTILITY_H
#define __TU_V4L2CAMERAUTILITY_H

#include "TU/V4L2++.h"
#include "TU/Heap.h"
#include <algorithm>	// for std::for_each()
#include <functional>	// for std::bind()

namespace TU
{
/************************************************************************
*  class V4L2CameraArray						*
************************************************************************/
//! Video for Linux v.2カメラの配列を表すクラス
class V4L2CameraArray : public Array<V4L2Camera>
{
  public:
  //! デフォルトのカメラ名
    static constexpr const char*
	DEFAULT_CAMERA_NAME = "/usr/local/etc/cameras/V4L2Camera";

  public:
    explicit		V4L2CameraArray(size_t ncameras=0)		;

    void		restore(const char* name=DEFAULT_CAMERA_NAME)	;
    void		save()					const	;
    const std::string&	name()					const	;
    std::string		configFile()				const	;
    std::string		calibFile()				const	;
    
  private:
    std::string		_name;	//!< カメラのfull path名
};

//! カメラ名を返す.
/*!
  \return	カメラ名
*/
inline const std::string&
V4L2CameraArray::name() const
{
    return _name;
}
    
//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
inline std::string
V4L2CameraArray::configFile() const
{
    return _name + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
inline std::string
V4L2CameraArray::calibFile() const
{
    return _name + ".calib";
}

/************************************************************************
*  global data								*
************************************************************************/
constexpr u_int	V4L2CAMERA_CHOICE	= V4L2Camera::UNKNOWN_FEATURE;
constexpr u_int	V4L2CAMERA_ALL		= V4L2Camera::UNKNOWN_FEATURE + 1;
    
/************************************************************************
*  global functions							*
************************************************************************/
template <class CAMERAS> auto
setFormat(CAMERAS&& cameras, u_int id, int val)
    -> typename std::enable_if<
	  std::is_convertible<
	      typename std::remove_reference<
		  decltype(*std::begin(cameras))>::type, V4L2Camera>::value,
	      bool>::type
{
    const auto	pixelFormat = V4L2Camera::uintToPixelFormat(id);

    if (pixelFormat == V4L2Camera::UNKNOWN_PIXEL_FORMAT)
	return false;

    std::for_each(std::begin(cameras), std::end(cameras),
		  [=](V4L2Camera& camera)
		  {
		      const auto&
			  frameSize = camera.availableFrameSizes(pixelFormat)
					    .first[val];
		      const auto&
			  frameRate = *frameSize.availableFrameRates().first;
		      camera.setFormat(pixelFormat,
				       frameSize.width.max,
				       frameSize.height.max,
				       frameRate.fps_n.min,
				       frameRate.fps_d.max);
		  });

    return true;
}

inline bool
setFormat(V4L2Camera& camera, u_int id, int val)
{
    return setFormat(make_range(&camera, &camera + 1), id, val);
}

template <class CAMERAS> auto
setFeature(CAMERAS&& cameras, u_int id, int val)
    -> typename std::enable_if<
	  std::is_convertible<
	      typename std::remove_reference<
		  decltype(*std::begin(cameras))>::type, V4L2Camera>::value,
	      bool>::type
{
    const auto	feature = V4L2Camera::uintToFeature(id);

    if (feature == V4L2Camera::UNKNOWN_FEATURE)
	return false;

    std::for_each(std::begin(cameras), std::end(cameras),
		  std::bind(&V4L2Camera::setValue,
			    std::placeholders::_1, feature, val));
    return true;
}
     
inline bool
setFeature(V4L2Camera& camera, u_int id, int val)
{
    return setFeature(make_range(&camera, &camera + 1), id, val);
}

inline bool
getFeature(const V4L2Camera& camera, u_int id, int& val)
{
    const auto	feature = V4L2Camera::uintToFeature(id);

    if (feature == V4L2Camera::UNKNOWN_FEATURE)
	return false;

    val = camera.getValue(feature);

    return true;
}

//! 複数のカメラから同期した画像を保持する．
/*!
  \param cameras	カメラの配列
  \param maxSkew	画像間のタイムスタンプの許容ずれ幅(nsec単位)
*/
template <class CAMERAS> auto
syncedSnap(CAMERAS&& cameras, uint64_t maxSkew=1000)
    -> typename std::enable_if<
	  std::is_convertible<
	      typename std::remove_reference<
		  decltype(*std::begin(cameras))>::type,
	      V4L2Camera>::value>::type
{
    typedef decltype(std::begin(cameras))	iterator;
    typedef std::pair<uint64_t, iterator>	timestamp_t;
    
    Heap<timestamp_t,
    std::greater<timestamp_t> >	timestamps(std::size(cameras));

    std::for_each(std::begin(cameras), std::end(cameras),
		  std::bind(&V4L2Camera::snap, std::placeholders::_1));

    timestamp_t	last(0, std::end(cameras));
    for (auto camera = std::begin(cameras); camera != std::end(cameras);
	 ++camera)
    {
	timestamp_t	timestamp(camera->arrivaltime(), camera);
	timestamps.push(timestamp);
	if (timestamp > last)
	    last = timestamp;
    }

    for (timestamp_t top; last.first > (top = timestamps.pop()).first + maxSkew;
	 timestamps.push(last))
	last = {top.second->snap().arrivaltime(), top.second};
}
    
}
#endif	// ! __TU_IEEE1394CAMERAUTILITY_H
