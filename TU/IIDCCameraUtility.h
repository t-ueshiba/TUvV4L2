/*
 *  $Id: IIDCCameraUtility.h 1681 2014-10-17 02:16:17Z ueshiba $
 */
/*!
  \file		IIDCCameraUtility.h
  \brief	クラス TU::IIDCCameraArray の定義と実装
*/
#ifndef __TU_IIDCCAMERAUTILITY_H
#define __TU_IIDCCAMERAUTILITY_H

#include "TU/IIDC++.h"
#include "TU/algorithm.h"
#include "TU/Heap.h"
#include <string>

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
//! IIDCデジタルカメラの配列を表すクラス
/*!
  TU::IIDCCameraへのポインタの配列として定義される.
*/
class IIDCCameraArray : public Array<IIDCCamera>
{
  public:
    typedef IIDCCamera	camera_type;

  public:
  //! デフォルトのカメラ名
    static constexpr const char* DEFAULT_CAMERA_NAME = "IIDCCamera";
  //! カメラ設定ファイルを収めるデフォルトのディレクトリ名
    static constexpr const char* DEFAULT_CONFIG_DIRS = ".:/usr/local/etc/cameras";
    
  public:
    IIDCCameraArray(const char* name=DEFAULT_CAMERA_NAME,
		    const char* dirs=DEFAULT_CONFIG_DIRS,
		    IIDCCamera::Speed speed=IIDCCamera::SPD_400M)	;
    ~IIDCCameraArray()							;

    void		restore(const char* name,
				const char* dirs=nullptr,
				IIDCCamera::Speed
				speed=IIDCCamera::SPD_400M)		;
    void		save()					const	;
    const std::string&	fullName()				const	;
    std::string		configFile()				const	;
    std::string		calibFile()				const	;
    
  private:
    std::string		_fullName;	//!< カメラのfull path名
};

/************************************************************************
*  global data								*
************************************************************************/
constexpr u_int	IIDCCAMERA_OFFSET_ONOFF = 0x100;
constexpr u_int	IIDCCAMERA_OFFSET_AUTO  = 0x200;
constexpr u_int	IIDCCAMERA_OFFSET_ABS   = 0x300;
constexpr u_int	IIDCCAMERA_OFFSET_VR    = 0x2;
constexpr u_int	IIDCCAMERA_CHOICE	= IIDCCamera::BRIGHTNESS + 2;
    
/************************************************************************
*  global functions							*
************************************************************************/
template <class ITER>
typename std::enable_if<std::is_convertible<
			    typename std::iterator_traits<ITER>::value_type,
			    IIDCCamera>::value, bool>::type
setFormat(ITER begin, ITER end, u_int id, u_int val)
{
    switch (id)
    {
      case IIDCCamera::YUV444_160x120:
      case IIDCCamera::YUV422_320x240:
      case IIDCCamera::YUV411_640x480:
      case IIDCCamera::YUV422_640x480:
      case IIDCCamera::RGB24_640x480:
      case IIDCCamera::MONO8_640x480:
      case IIDCCamera::MONO16_640x480:
      case IIDCCamera::YUV422_800x600:
      case IIDCCamera::RGB24_800x600:
      case IIDCCamera::MONO8_800x600:
      case IIDCCamera::YUV422_1024x768:
      case IIDCCamera::RGB24_1024x768:
      case IIDCCamera::MONO8_1024x768:
      case IIDCCamera::MONO16_800x600:
      case IIDCCamera::MONO16_1024x768:
      case IIDCCamera::YUV422_1280x960:
      case IIDCCamera::RGB24_1280x960:
      case IIDCCamera::MONO8_1280x960:
      case IIDCCamera::YUV422_1600x1200:
      case IIDCCamera::RGB24_1600x1200:
      case IIDCCamera::MONO8_1600x1200:
      case IIDCCamera::MONO16_1280x960:
      case IIDCCamera::MONO16_1600x1200:
	std::for_each(begin, end, std::bind(&IIDCCamera::setFormatAndFrameRate,
					    std::placeholders::_1,
					    IIDCCamera::uintToFormat(id),
					    IIDCCamera::uintToFrameRate(val)));
	return true;

      default:
	break;
    }

    return false;
}

inline bool
setFormat(IIDCCamera& camera, u_int id, u_int val)
{
    return setFormat(&camera, &camera + 1, id, val);
}

template <class ITER>
typename std::enable_if<std::is_convertible<
			    typename std::iterator_traits<ITER>::value_type,
			    IIDCCamera>::value, bool>::type
setFeature(ITER begin, ITER end, u_int id, u_int val, float fval)
{
    using namespace	std;
    
    switch (id)
    {
      case IIDCCamera::BRIGHTNESS:
      case IIDCCamera::AUTO_EXPOSURE:
      case IIDCCamera::SHARPNESS:
      case IIDCCamera::WHITE_BALANCE:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_VR:
      case IIDCCamera::HUE:
      case IIDCCamera::SATURATION:
      case IIDCCamera::GAMMA:
      case IIDCCamera::SHUTTER:
      case IIDCCamera::GAIN:
      case IIDCCamera::IRIS:
      case IIDCCamera::FOCUS:
      case IIDCCamera::TEMPERATURE:
      case IIDCCamera::TRIGGER_DELAY:
      case IIDCCamera::FRAME_RATE:
      case IIDCCamera::ZOOM:
      case IIDCCamera::PAN:
      case IIDCCamera::TILT:
      {
	const auto	feature = IIDCCamera::uintToFeature(id);

	if (begin->isAbsControl(feature))
	{
	    IIDCCamera& (IIDCCamera::*pf)(IIDCCamera::Feature, float)
		= &IIDCCamera::setValue;
	  
	    for_each(begin, end, bind(pf, placeholders::_1, feature, fval));
	}
	else
	{
	    IIDCCamera& (IIDCCamera::*pf)(IIDCCamera::Feature, u_int)
		= &IIDCCamera::setValue;
	  
	    for_each(begin, end, bind(pf, placeholders::_1, feature, u_int(fval)));
	}
      }
	return true;
      case IIDCCamera::TRIGGER_MODE:
	for_each(begin, end, bind(&IIDCCamera::setTriggerMode, placeholders::_1,
				  IIDCCamera::uintToTriggerMode(val)));
	return true;
      
      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TRIGGER_DELAY + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::FRAME_RATE    + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_ONOFF:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_ONOFF:
      {
	IIDCCamera& (IIDCCamera::*pf)(IIDCCamera::Feature, bool)
	    = &IIDCCamera::setActive;
	for_each(begin, end,
		 bind(pf, placeholders::_1,
		      IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_ONOFF),
		      bool(val)));
      }
	return true;

      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::FRAME_RATE    + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_AUTO:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_AUTO:
	for_each(begin, end,
		 bind(&IIDCCamera::setAuto, placeholders::_1,
		      IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_AUTO),
		      bool(val)));
        return true;

      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_AUTO:
	for_each(begin, end,
		 bind(&IIDCCamera::setTriggerPolarity, placeholders::_1,
		      bool(val)));
	return true;

      case IIDCCamera::BRIGHTNESS    + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::AUTO_EXPOSURE + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::SHARPNESS     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::HUE	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::SATURATION    + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::GAMMA	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::SHUTTER	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::GAIN	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::IRIS	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::FOCUS	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::TEMPERATURE   + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::TRIGGER_DELAY + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::FRAME_RATE    + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::ZOOM	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::PAN	     + IIDCCAMERA_OFFSET_ABS:
      case IIDCCamera::TILT	     + IIDCCAMERA_OFFSET_ABS:
	for_each(begin, end,
		 bind(&IIDCCamera::setAbsControl, placeholders::_1,
		      IIDCCamera::uintToFeature(id - IIDCCAMERA_OFFSET_ABS),
		      bool(val)));
        return true;

      default:
	break;
    }

    return false;
}

inline bool
setFeature(IIDCCamera& camera, u_int id, u_int val, float fval)
{
    return setFeature(&camera, &camera + 1, id, val, fval);
}

//! 複数のカメラから同期した画像を保持する．
/*!
  \param cameras	カメラへのポインタの配列
  \param maxSkew	画像間のタイムスタンプの許容ずれ幅(nsec単位)
*/
template <class ITER> void
syncedSnap(ITER camera, size_t ncameras, uint64_t maxSkew=1000)
{
    typedef std::pair<uint64_t, ITER>	timestamp_t;
    
    Heap<timestamp_t, std::greater<timestamp_t> >	timestamps(ncameras);

    std::for_each_n(camera, ncameras,
		    std::bind(&IIDCCamera::snap, std::placeholders::_1));

    timestamp_t	last(0, nullptr);
    for (; ncameras--; ++camera)
    {
	timestamp_t	timestamp(camera->getTimestamp(), camera);
	timestamps.push(timestamp);
	if (timestamp > last)
	    last = timestamp;
    }

    for (timestamp_t top; last.first - (top = timestamps.pop()).first > maxSkew;
	 timestamps.push(last))
	last = {top.second->snap().getTimestamp(), top.second};
}
    
}
#endif	// ! __TU_IIDCCAMERAUTILITY_H
