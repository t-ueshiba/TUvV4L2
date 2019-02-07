/*!
 *   \file	MainWindow.h
 */
#ifndef TU_MAINWINDOW_H
#define TU_MAINWINDOW_H

#include <QMainWindow>
#include <QListWidget>
#include <QFileDialog>
#include <QErrorMessage>
#include <yaml-cpp/yaml.h>
#include <fstream>
#include "CameraWindow.h"
#include "xpms.h"

namespace TU
{
/************************************************************************
*  class MainWindow<CAMERA>						*
************************************************************************/
template <class CAMERA>
class MainWindow : public QMainWindow
{
  public:
			MainWindow()					;

  private:
    void		addExtraCmds()					{}
    void		up()						;
    void		down()						;
    void		save()					const	;
    
    static CAMERA	createCamera(int n)				;
    static QString	defaultConfigFile()				;
    
  private:
    std::vector<CAMERA>		_cameras;
    QWidget*		const	_central;
    QGridLayout*	const	_layout;
    QListWidget*	const	_list;
    QErrorMessage*	const	_errmsg;
};

template <class CAMERA>
MainWindow<CAMERA>::MainWindow()
    :QMainWindow(),
     _central(new QWidget(this)),
     _layout(new QGridLayout(_central)),
     _list(new QListWidget(_central)),
     _errmsg(new QErrorMessage(_central))
{
    _layout->setHorizontalSpacing(2);
    _layout->setVerticalSpacing(2);

  // 利用可能な全てのカメラを列挙してリストに入れる．
    for (int n = 0; ; ++n)
    {
	try
	{
	    _cameras.emplace_back(createCamera(n));
	    _list->addItem(new QListWidgetItem(QPixmap(camera_xpm),
					       cameraName(_cameras.back()),
					       _list));
	}
	catch (const std::exception& err)
	{
	    break;
	}
    }

    if (_list->count() > 0)
	_list->setCurrentRow(0);

  // リスト中のカメラが選択されたらそれを操作するウィンドウを生成し，
  // リストから隠す．ウィンドウが破壊されたら再びリストに表示する．
    connect(_list, &QListWidget::itemDoubleClicked,
	    [this](QListWidgetItem* item)
	    {
		const auto
		    cameraWindow = new CameraWindow<CAMERA>(
					  this, _cameras[_list->row(item)]);
		connect(cameraWindow, &QObject::destroyed,
			[item](){ item->setHidden(false); });
		item->setHidden(true);
		cameraWindow->show();
	    });
    _layout->addWidget(_list, 0, 0, 2, 1);

  // ボタンを生成する．
    auto	button = new QPushButton(QPixmap(up_xpm), tr(""), _central);
    button->setSizePolicy(QSizePolicy::Preferred,
			  QSizePolicy::MinimumExpanding);
    connect(button, &QPushButton::clicked, this, &MainWindow::up);
    _layout->addWidget(button, 0, 1, 1, 1);

    button = new QPushButton(QPixmap(down_xpm), tr(""), _central);
    button->setSizePolicy(QSizePolicy::Preferred,
			  QSizePolicy::MinimumExpanding);
    connect(button, &QPushButton::clicked, this, &MainWindow::down);
    _layout->addWidget(button, 1, 1, 1, 1);

    button = new QPushButton(QPixmap(save_xpm), tr("Save"), _central);
    connect(button, &QPushButton::clicked, this, &MainWindow::save);
    _layout->addWidget(button, 0, 2, 1, 1);

    button = new QPushButton(QPixmap(exit_xpm), tr("Exit"), _central);
    connect(button, &QPushButton::clicked, this, &QMainWindow::close);
    _layout->addWidget(button, 1, 2, 1, 1);

    addExtraCmds();

    setCentralWidget(_central);

    show();
}

template <class CAMERA> void
MainWindow<CAMERA>::up()
{
    const auto	row = _list->currentRow();

    for (auto i = row; i < _list->count(); ++i)
	if (!_list->item(i)->isHidden())
	{
	    for (auto j = row; --j >= 0; )
		if (!_list->item(j)->isHidden())
		{
		    std::swap(*_list->item(i), *_list->item(j));
		    std::swap(_cameras[i], _cameras[j]);
		    _list->setCurrentRow(j);
		    
		    break;
		}

	    break;
	}
}
    
template <class CAMERA> void
MainWindow<CAMERA>::down()
{
    const auto	row = _list->currentRow();

    for (auto i = row; i >= 0; --i)
	if (!_list->item(i)->isHidden())
	{
	    for (auto j = row; ++j < _list->count(); )
		if (!_list->item(j)->isHidden())
		{
		    std::swap(*_list->item(i), *_list->item(j));
		    std::swap(_cameras[i], _cameras[j]);
		    _list->setCurrentRow(j);

		    break;
		}

	    break;
	}
}
    
template <class CAMERA> void
MainWindow<CAMERA>::save() const
{
    if (_cameras.size() == 0)
	return;

    auto		fileName = QFileDialog::getSaveFileName(
					_central, tr("Save config."),
					defaultConfigFile(),
					tr("camera_name (*.conf)"));
    std::ofstream	out(fileName.toUtf8().data());
    if (!out)
    {
	_errmsg->showMessage(fileName.prepend("Cannot open "));
	return;
    }
    
    YAML::Emitter	emitter;
    emitter << YAML::BeginSeq;
    for (const auto& camera : _cameras)
	emitter << camera;
    emitter << YAML::EndSeq;
    
    out << emitter.c_str() << std::endl;
}
    
}
#endif	// !TU_MAINWINDOW_H
