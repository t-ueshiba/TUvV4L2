/*
 *  $Id$
 */
/*!
  \file		Ieee1394CameraArray.h
  \brief	クラス TU::Ieee1394CameraArray の定義と実装
*/
#ifndef __TU_IEEE1394CAMERAARRAY_H
#define __TU_IEEE1394CAMERAARRAY_H

#include "TU/Ieee1394++.h"

#ifdef HAVE_LIBTUTOOLS__
#  include <string>

//! デフォルトのカメラ名
#  define DEFAULT_CAMERA_NAME	"IEEE1394Camera"
//! カメラ設定ファイルを収めるデフォルトのディレクトリ名
#  define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"

namespace TU
{
/************************************************************************
*  class Ieee1394CameraArray						*
************************************************************************/
//! IEEE1394デジタルカメラの配列を表すクラス
/*!
  TU::Ieee1394Cameraへのポインタの配列として定義される.
*/
class Ieee1394CameraArray : public Array<Ieee1394Camera*>
{
  public:
    typedef Ieee1394Camera	camera_type;
    
  public:
    Ieee1394CameraArray()						;
    Ieee1394CameraArray(const char* name, const char* dirs=0,
			Ieee1394Node::Speed speed=Ieee1394Node::SPD_400M,
			int ncameras=-1)				;
    ~Ieee1394CameraArray()						;

    void	initialize(const char* name, const char* dirs=0,
			   Ieee1394Node::Speed
			       speed=Ieee1394Node::SPD_400M,
			   int ncameras=-1)				;
    const std::string&
		fullName()					const	;
    std::string	configFile()					const	;
    std::string	calibFile()					const	;
    const Ieee1394CameraArray&
		exec(Ieee1394Camera& (Ieee1394Camera::*mf)(),
		     int n=-1)					const	;
    template <class ARG>
    const Ieee1394CameraArray&
		exec(Ieee1394Camera& (Ieee1394Camera::*mf)(ARG),
		     ARG arg, int n=-1)				const	;
    template <class ARG0, class ARG1>
    const Ieee1394CameraArray&
		exec(Ieee1394Camera& (Ieee1394Camera::*mf)(ARG0, ARG1),
		     ARG0 arg0, ARG1 arg1, int n=-1)		const	;
    
  private:
    std::string	_fullName;	//!< カメラのfull path名
};

//! カメラのfull path名を返す.
/*!
  \return	カメラのfull path名
*/
inline const std::string&
Ieee1394CameraArray::fullName() const
{
    return _fullName;
}
    
//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
inline std::string
Ieee1394CameraArray::configFile() const
{
    return _fullName + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
inline std::string
Ieee1394CameraArray::calibFile() const
{
    return _fullName + ".calib";
}

template <class ARG> const Ieee1394CameraArray&
Ieee1394CameraArray::exec(Ieee1394Camera& (Ieee1394Camera::*mf)(ARG),
			  ARG arg, int n) const
{
    if (0 <= n && n < size())
	((*this)[n]->*mf)(arg);
    else
	for (size_t i = 0; i < size(); ++i)
	    ((*this)[i]->*mf)(arg);
    return *this;
}

template <class ARG0, class ARG1> const Ieee1394CameraArray&
Ieee1394CameraArray::exec(Ieee1394Camera& (Ieee1394Camera::*mf)(ARG0, ARG1),
			  ARG0 arg0, ARG1 arg1, int n) const
{
    if (0 <= n && n < size())
	((*this)[n]->*mf)(arg0, arg1);
    else
	for (size_t i = 0; i < size(); ++i)
	    ((*this)[i]->*mf)(arg0, arg1);
    return *this;
}

/************************************************************************
*  global functions							*
************************************************************************/
bool	handleCameraFormats(const Ieee1394CameraArray& cameras,
			    u_int id, u_int val);
bool	handleCameraFeatures(const Ieee1394CameraArray& cameras,
			     u_int id, int val, int new_val, int n=-1);
bool	handleCameraWhiteBalance(const Ieee1394CameraArray& cameras,
				 u_int id, int val, int ub, int vr, int n=-1);
}
#endif	// HAVE_LIBTUTOOLS__
#endif	// ! __TU_IEEE1394CAMERAARRAY_H
