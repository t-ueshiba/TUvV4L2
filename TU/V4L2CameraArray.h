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
			DEFAULT_CAMERA_NAME = "V4L2Camera";
  //! カメラ設定ファイルを収めるデフォルトのディレクトリ名
    static constexpr const char*
			DEFAULT_CONFIG_DIRS = ".:/usr/local/etc/cameras";

  public:
    explicit		V4L2CameraArray(size_t ncameras=0)		;

    void		restore(const char* name=nullptr,
				const char* dirs=nullptr)		;
    void		save()					const	;
    const std::string&	fullName()				const	;
    std::string		configFile()				const	;
    std::string		calibFile()				const	;
    
  private:
    std::string		_fullName;	//!< カメラのfull path名
};

//! カメラのfull path名を返す.
/*!
  \return	カメラのfull path名
*/
inline const std::string&
V4L2CameraArray::fullName() const
{
    return _fullName;
}
    
//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
inline std::string
V4L2CameraArray::configFile() const
{
    return _fullName + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
inline std::string
V4L2CameraArray::calibFile() const
{
    return _fullName + ".calib";
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

}
#endif	// ! __TU_IEEE1394CAMERAUTILITY_H
