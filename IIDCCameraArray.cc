/*
 *  $Id$
 */
#include "TU/IIDCCameraArray.h"
#include "TU/io.h"

namespace TU
{
#ifndef TUIIDCPP_CONF_DIR
#  define TUIIDCPP_CONF_DIR	"/usr/local/etc"
#endif
    
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
constexpr const char*	IIDCCameraArray::DEFAULT_CAMERA_NAME;
    
//! IIDCデジタルカメラの配列を生成する.
IIDCCameraArray::IIDCCameraArray(const char* name)
    :Array<IIDCCamera>(), _name(name)
{
}
    
//! 設定ファイルを読み込んでIIDCデジタルカメラの配列を初期化する.
void
IIDCCameraArray::restore()
{
    std::ifstream	in(configFile().c_str());
    if (!in)
	throw std::runtime_error("IIDCCameraArray::restore(): cannot open " +
				 configFile());
    in >> *this;
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

//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
std::string
IIDCCameraArray::configFile() const
{
    return std::string(TUIIDCPP_CONF_DIR) + '/' + _name + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
std::string
IIDCCameraArray::calibFile() const
{
    return std::string(TUIIDCPP_CONF_DIR) + '/' + _name + ".calib";
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

	if (camera.isAbsControl(feature))
	{
	    val  = 0;
	    fval = camera.getValue<float>(feature);
	}
	else
	{
	    val  = camera.getValue<u_int>(feature);
	    fval = val;
	}
      }
	return true;
	
      case IIDCCamera::TRIGGER_MODE:
	val  = camera.getTriggerMode();
	fval = 0;
	return true;

      case IIDCCamera::WHITE_BALANCE:
      case IIDCCamera::WHITE_BALANCE + IIDCCAMERA_OFFSET_VR:
	if (camera.isAbsControl(IIDCCamera::WHITE_BALANCE))
	{
	    float	ub, vr;
	    camera.getWhiteBalance(ub, vr);

	    val  = 0;
	    fval = (id == IIDCCamera::WHITE_BALANCE ? ub : vr);
	}
	else
	{
	    u_int	ub, vr;
	    camera.getWhiteBalance(ub, vr);

	    val  = (id == IIDCCamera::WHITE_BALANCE ? ub : vr);
	    fval = val;
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
	fval = 0;
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
	fval = 0;
	return true;

      case IIDCCamera::TRIGGER_MODE  + IIDCCAMERA_OFFSET_AUTO:
	val  = camera.getTriggerPolarity();
	fval = 0;
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
	fval = 0;
        return true;

      default:
	break;
    }

    return false;
}
    
std::ostream&
operator <<(std::ostream& out, const IIDCCameraArray& cameras)
{
    YAML::Emitter	emitter;
    emitter << YAML::BeginSeq;
    for (const auto& camera : cameras)
	emitter << camera;
    emitter << YAML::EndSeq;

    return out << emitter.c_str() << std::endl;
}

std::istream&
operator >>(std::istream& in, IIDCCameraArray& cameras)
{
    const auto	node = YAML::Load(in);
    cameras.resize(node.size());
    for (size_t i = 0; i < node.size(); ++i)
	node[i] >> cameras[i];

    return in;
}

}
