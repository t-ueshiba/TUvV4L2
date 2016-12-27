/*
 *  $Id$
 */
#include "TU/IIDCCameraArray.h"
#include "TU/io.h"

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
constexpr const char*	IIDCCameraArray::DEFAULT_CAMERA_NAME;
    
//! IIDCデジタルカメラの配列を生成する.
IIDCCameraArray::IIDCCameraArray(size_t ncameras)
    :Array<IIDCCamera>(ncameras), _name()
{
}
    
//! 設定ファイルを読み込んでIIDCデジタルカメラの配列を初期化する.
/*!
  \param fullName	カメラ名
  \param dirs		カメラ設定ファイルの探索ディレクトリ名の並び
  \param speed		FireWireバスの速度
*/
void
IIDCCameraArray::restore(const char* name, IIDCCamera::Speed speed)
{
    _name = name;

  // 設定ファイルをオープンする.
    std::ifstream	in(configFile().c_str());
    if (!in)
	throw std::runtime_error("IIDCCameraArray::restore(): cannot open " +
				 configFile());
    in >> *this;
    for (auto& camera : *this)
	camera.setSpeed(speed);
}

//! IIDCデジタルカメラの配列の設定を設定ファイルに書き込む.
void
IIDCCameraArray::save() const
{
    std::ofstream	out(configFile().c_str());
    if (!out)
	throw std::runtime_error("IIDCCameraArray::save(): cannot open " +
				 configFile());
    out << *this;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool
getFeature(const IIDCCamera& camera, u_int id, u_int& val, float& fval)
{
    switch (id)
    {
      case IIDCCamera::BRIGHTNESS:
      case IIDCCamera::AUTO_EXPOSURE:
      case IIDCCamera::SHARPNESS:
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
	  
	val  = camera.getValue<u_int>(feature);
	fval = camera.getValue<float>(feature);
      }
	return true;
	
      case IIDCCamera::TRIGGER_MODE:
	val  = camera.getTriggerMode();
	fval = val;
	return true;

      case IIDCCamera::WHITE_BALANCE:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_VR:
      {
	u_int	ub, vr;
	camera.getWhiteBalance(ub, vr);
	float	fub, fvr;
	camera.getWhiteBalance(fub, fvr);

	if (id == IIDCCamera::WHITE_BALANCE)
	{
	    val  = ub;
	    fval = fub;
	}
	else
	{
	    val  = vr;
	    fval = fvr;
	}
      }
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
	val  = camera.isActive(IIDCCamera::uintToFeature(
				   id - IIDCCAMERA_OFFSET_ONOFF));
	fval = val;
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
	val  = camera.isAuto(IIDCCamera::uintToFeature(
				 id - IIDCCAMERA_OFFSET_AUTO));
	fval = val;
	return true;

      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_AUTO:
	val  = camera.getTriggerPolarity();
	fval = val;
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
	val  = camera.isAbsControl(IIDCCamera::uintToFeature(
				       id - IIDCCAMERA_OFFSET_ABS));
	fval = val;
        return true;

      default:
	break;
    }

    return false;
}
    
}
