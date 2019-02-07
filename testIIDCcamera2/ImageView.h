/*!
 *  \file	ImageView.h
 */
#ifndef TU_IMAGEVIEW_H
#define TU_IMAGEVIEW_H

#include <QGraphicsView>
#include <QImage>
#include <QMenu>
#include "TU/Image++.h"

namespace TU
{
/************************************************************************
*  class ImageView							*
************************************************************************/
class ImageView : public QGraphicsView
{
  private:
    template <class T>
    struct Tag	{ using	type = T; };
    
  public:
    ImageView(QWidget* parent, size_t width, size_t height)		;

    template <class CAMERA>
    void	captureAndDisplay(CAMERA& camera)			;
    
  private:
    template <class CAMERA, class T>
    void	captureAndDisplay(CAMERA& camera, Tag<T>)		;
    template <class CAMERA>
    void	captureAndDisplay(CAMERA& camera, Tag<uint8_t>)		;
    template <class CAMERA>
    void	captureAndDisplay(CAMERA& camera, Tag<RGB>)		;
    template <class CAMERA>
    void	captureBayerAndDisplay(CAMERA& camera)			;
    
    void	paintEvent(QPaintEvent* event)				;
    void	setScale(qreal scale)					;
    void	showContextMenu(const QPoint& p)			;

  private:
    QVector<uchar>	_image;
    QVector<RGB>	_rgb;
    QImage		_qimage;
    QVector<QRgb>	_colors;
    QMenu* const	_menu;
    bool		_fit;
    qreal		_scale;
};

template <class CAMERA, class T> inline void
ImageView::captureAndDisplay(CAMERA& camera, Tag<T>)
{
    constexpr auto	N = iterator_value<pixel_iterator<const T*> >::npixels;

    const auto	npixels = camera.width() * camera.height();
    _image.resize(npixels*sizeof(T));
    camera.snap().captureRaw(_image.data());

    _rgb.resize(npixels);
    std::copy_n(make_pixel_iterator(reinterpret_cast<const T*>(_image.data())),
		npixels/N,
		make_pixel_iterator(_rgb.data()));
    _qimage = QImage(reinterpret_cast<const uchar*>(_rgb.data()),
		     camera.width(), camera.height(),
		     camera.width()*sizeof(RGB), QImage::Format_RGB888);
    _qimage.setColorTable(_colors);
}

template <class CAMERA> inline void
ImageView::captureAndDisplay(CAMERA& camera, Tag<uint8_t>)
{
    const auto	npixels = camera.width() * camera.height();
    _image.resize(npixels);
    camera.snap().captureRaw(_image.data());

    _qimage = QImage(_image.data(), camera.width(), camera.height(),
		     camera.width(), QImage::Format_Indexed8);
    _qimage.setColorTable(_colors);
}

template <class CAMERA> inline void
ImageView::captureAndDisplay(CAMERA& camera, Tag<RGB>)
{
    const auto	npixels = camera.width() * camera.height();
    _image.resize(npixels*sizeof(RGB));
    camera.snap().captureRaw(_image.data());

    _qimage = QImage(_image.data(), camera.width(), camera.height(),
		     camera.width()*sizeof(RGB), QImage::Format_RGB888);
    _qimage.setColorTable(_colors);
}

template <class CAMERA> inline void
ImageView::captureBayerAndDisplay(CAMERA& camera)
{
    const auto	npixels = camera.width() * camera.height();
    _image.resize(npixels*sizeof(RGB));
    camera.snap().captureBayerRaw(_image.data());

    _qimage = QImage(_image.data(), camera.width(), camera.height(),
		     camera.width()*sizeof(RGB), QImage::Format_RGB888);
    _qimage.setColorTable(_colors);
}


}	// namespace TU
#endif	// !TU_IMAGEVIEW_H
