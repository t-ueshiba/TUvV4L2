/*
 *  $Id: IIDCCameraArray.h 1681 2014-10-17 02:16:17Z ueshiba $
 */
/*!
  \file		IIDCCameraArray.h
  \brief	クラス TU::IIDCCameraArray の定義と実装
*/
#ifndef __TU_IIDCCAMERAARRAY_H
#define __TU_IIDCCAMERAARRAY_H

#include "TU/IIDC++.h"

#include <string>

//! デフォルトのカメラ名
#define DEFAULT_CAMERA_NAME	"IIDCCamera"
//! カメラ設定ファイルを収めるデフォルトのディレクトリ名
#define DEFAULT_CONFIG_DIRS	".:/usr/local/etc/cameras"

namespace TU
{
/************************************************************************
*  class IIDCCameraArray						*
************************************************************************/
//! IIDCデジタルカメラの配列を表すクラス
/*!
  TU::IIDCCameraへのポインタの配列として定義される.
*/
class IIDCCameraArray : public Array<IIDCCamera*>
{
  public:
    typedef IIDCCamera	camera_type;
    
  public:
    IIDCCameraArray()							;
    IIDCCameraArray(const char* name, const char* dirs=0,
		    IIDCCamera::Speed speed=IIDCCamera::SPD_400M,
		    int ncameras=-1)					;
    ~IIDCCameraArray()							;

    void		initialize(const char* name, const char* dirs=0,
				   IIDCCamera::Speed
				   speed=IIDCCamera::SPD_400M,
				   int ncameras=-1)			;
    const std::string&	fullName()				const	;
    std::string		configFile()				const	;
    std::string		calibFile()				const	;
    friend std::istream&
			operator >>(std::istream& in,
				    IIDCCameraArray& cameras)		;
    
  private:
    std::istream&	restore(std::istream& in,
				int ncameras, IIDCCamera::Speed speed)	;
    
  private:
    std::string		_fullName;	//!< カメラのfull path名
};

//! カメラのfull path名を返す.
/*!
  \return	カメラのfull path名
*/
inline const std::string&
IIDCCameraArray::fullName() const
{
    return _fullName;
}
    
//! カメラ設定ファイル名を返す.
/*!
  \return	カメラ設定ファイル名
*/
inline std::string
IIDCCameraArray::configFile() const
{
    return _fullName + ".conf";
}
    
//! キャリブレーションファイル名を返す.
/*!
  \return	キャリブレーションファイル名
*/
inline std::string
IIDCCameraArray::calibFile() const
{
    return _fullName + ".calib";
}

/************************************************************************
*  global functions							*
************************************************************************/
std::ostream&
operator <<(std::ostream& out, const IIDCCameraArray& cameras)		;

}
#endif	// ! __TU_IIDCCAMERAARRAY_H
