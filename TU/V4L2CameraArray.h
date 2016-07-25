/*
 *  $Id: V4L2CameraArray.h 1588 2014-07-10 20:05:03Z ueshiba $
 */
/*!
  \file		V4L2CameraArray.h
  \brief	クラス TU::V4L2CameraArray の定義と実装
*/
#ifndef __TU_V4L2CAMERAARRAY_H
#define __TU_V4L2CAMERAARRAY_H

#include "TU/V4L2++.h"
#include "TU/io.h"

//! デフォルトのカメラ名
#define TU_V4L2_DEFAULT_CAMERA_NAME	"V4L2Camera"
//! カメラ設定ファイルを収めるデフォルトのディレクトリ名
#define TU_V4L2_DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"

namespace TU
{
/************************************************************************
*  class V4L2CameraArray						*
************************************************************************/
//! IEEE1394デジタルカメラの配列を表すクラス
/*!
  TU::V4L2Cameraへのポインタの配列として定義される.
*/
class V4L2CameraArray : public Array<V4L2Camera*>
{
  public:
    typedef V4L2Camera	camera_type;
    
  public:
    V4L2CameraArray()							;
    V4L2CameraArray(const char* name, const char* dirs=0,
		    int ncameras=-1)					;
    ~V4L2CameraArray()							;

    void		initialize(const char* name, const char* dirs=0,
				   int ncameras=-1)			;
    const std::string&	fullName()				const	;
    std::string		configFile()				const	;
    std::string		calibFile()				const	;
    friend std::istream&
			operator >>(std::istream& in,
				    V4L2CameraArray& cameras)		;
    
  private:
    std::istream&	restore(std::istream& in, int ncameras)		;
    
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
*  global functions							*
************************************************************************/
std::ostream&	operator <<(std::ostream& out, const V4L2CameraArray& cameras);

}
#endif	// ! __TU_IEEE1394CAMERAARRAY_H
