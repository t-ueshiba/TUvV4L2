/*
 *  $Id$
 */
#include <boost/foreach.hpp>
#include "TU/HRP2++.h"

namespace TU
{
/************************************************************************
*  class HRP2								*
************************************************************************/
HRP2::HRP2(int argc, char* argv[], const char* linkName, u_int capacity)
    :super(), _getRealPose(*this, linkName, capacity), _executeCommand(*this)
{
    using namespace	std;
    
    if (!init(argc, argv))
	throw runtime_error("HRP2Client::init() failed!");
    setup(false);
}

void
HRP2::setup(bool isLeftHand)
{
    using namespace	std;
    
  // 拘束設定
    bool	constrained[] = {true, true, true, true, true, true};
    double	weights[]     = {10.0, 10.0, 10.0, 10.0, 10.0, 10.0};
    if (!SelectTaskDofs(isLeftHand, constrained, weights))
	throw runtime_error("HRP2Client::SelectTaskDofs() failed!!");
    
  // 使用する自由度を設定
    bool	usedDofs[] =
		{
		    true, true, true, true, true, true,		// right leg
		    true, true, true, true, true, true,		// left leg
		    true, false,				// waist
		    false, false,				// neck
		    true, true, true, true, true, true, true,	// right arm
		    false,					// right hand
		    false, false, false, false, false, false, false, // left arm
		    false,					// left hand
		    true, true, true, false, false, false	// base
		};
    if (!SelectUsedDofs(usedDofs))
	throw runtime_error("HRP2Client::SelectUsedDofs() failed!!");
    
  // ベースとなるリンクを設定
    if (!SelectBaseLink("RLEG_JOINT5"))
	throw runtime_error("HRP2Client::SelectBaseLink() failed!!");

  // Laterモードに設定
    if (!SelectExecutionMode(true))
	throw runtime_error("HRP2Client::SelectExecutionMode() failed!!");

  // 反対側の手を強制的にdeselectし，所望の手をselectする．
    DeSelectArm(!isLeftHand);
    SelectArm(isLeftHand);

  // スレッドを起動する．
    _getRealPose.run();
    _executeCommand.run();
}

bool
HRP2::GetRealPose(const char* linkName, Pose& D, Time& t) const
{
    return _getRealPose(D, t);
}
    
bool
HRP2::GetRealPose(const char* linkName, Time time, Pose& D, Time& t) const
{
    return _getRealPose(time, D, t);
}

void
HRP2::PlayMotion(bool blocked)
{
    if (blocked)
	super::PlayMotion();
    else
	_executeCommand(&super::PlayMotion);
}

bool
HRP2::poll() const
{
    return _executeCommand.poll();
}
    
/************************************************************************
*  class HRP2::GetRealPoseThread					*
************************************************************************/
HRP2::GetRealPoseThread::GetRealPoseThread(HRP2Client& hrp2,
					   const char* linkName,
					   u_int capacity)
    :_hrp2(hrp2), _linkName(linkName), _poses(capacity),
     _quit(false), _mutex(), _thread()
     
{
    pthread_mutex_init(&_mutex, NULL);
}

HRP2::GetRealPoseThread::~GetRealPoseThread()
{
    pthread_mutex_lock(&_mutex);
    _quit = true;			// 終了フラグを立てる
    pthread_mutex_unlock(&_mutex);

    pthread_join(_thread, NULL);	// 子スレッドの終了を待つ
    pthread_mutex_destroy(&_mutex);
}

void
HRP2::GetRealPoseThread::run()
{
    pthread_create(&_thread, NULL, threadProc, this);

    usleep(1000);
}
    
bool
HRP2::GetRealPoseThread::operator ()(Time time,
				     Pose& D, Time& t) const
{
  // 与えられた時刻よりも後のポーズが得られるまで待つ．
    ChronoPose	after;		// timeよりも後の時刻で取得されたポーズ
    for (;;)			// を発見するまで待つ．
    {
	pthread_mutex_lock(&_mutex);
	if (!_poses.empty() && (after = _poses.back()).t > time)
	    break;
	pthread_mutex_unlock(&_mutex);
    }

  // リングバッファを過去に遡り，与えられた時刻よりも前のポーズを探す．
    BOOST_REVERSE_FOREACH (const ChronoPose& pose, _poses)
    {
	if (pose.t <= time)	// timeの直前のポーズならば．．．
	{
	  // poseとafterのうち，その時刻がtimeに近い方を返す．
	    if ((time - pose.t) < (after.t - time))
	    {			// timeがafterよりもposeの時刻に近ければ．．．
		D = pose.D;
		t = pose.t;
	    }
	    else		// timeがposeよりもafterの時刻に近ければ．．．
	    {
		D = after.D;
		t = after.t;
	    }
	    pthread_mutex_unlock(&_mutex);

	    return true;	// timeを挟む２つのポーズを発見した
	}
	after = pose;
    }
    pthread_mutex_unlock(&_mutex);

    return false;		// timeの直前のポーズを発見できなかった
}

bool
HRP2::GetRealPoseThread::operator ()(Pose& D, Time& t) const
{
    pthread_mutex_lock(&_mutex);
    if (!_poses.empty())
    {
	D = _poses.back().D;
	t = _poses.back().t;
	pthread_mutex_unlock(&_mutex);

	return true;
    }
    pthread_mutex_unlock(&_mutex);

    return false;
}
    
void
HRP2::GetRealPoseThread::timeSpan(Time& t0, Time& t1) const
{
    for (;;)
    {
	pthread_mutex_lock(&_mutex);
	if (!_poses.empty())
	{
	    t0 = _poses.front().t;
	    t1 = _poses.back().t;
	    pthread_mutex_unlock(&_mutex);
	    break;
	}
	pthread_mutex_unlock(&_mutex);
    }
}
    
void*
HRP2::GetRealPoseThread::mainLoop()
{
    for (;;)
    {
	pthread_mutex_lock(&_mutex);
	bool	quit = _quit;		// 命令を取得
	pthread_mutex_unlock(&_mutex);
	if (quit)			// 終了命令ならば...
	    break;			// 脱出

	Pose	D;
	double	sec, nsec;
	if (_hrp2.GetRealPose(const_cast<char*>(_linkName.c_str()),
			      D.data(), &sec, &nsec))	// ポーズ入力成功？
	{
	    Time	t = usec(sec, nsec);	// micro second
	    
	    if (_poses.empty() || (t != _poses.back().t))
	    {
		pthread_mutex_lock(&_mutex);
		_poses.push_back(ChronoPose(D, t));	// リングバッファに入れる
		pthread_mutex_unlock(&_mutex);
	    }
	}
    }

    return 0;
}
    
void*
HRP2::GetRealPoseThread::threadProc(void* thread)
{
    GetRealPoseThread*	th = static_cast<GetRealPoseThread*>(thread);

    return th->mainLoop();
}

/************************************************************************
*  class HRP2::ExecuteCommandThread					*
************************************************************************/
HRP2::ExecuteCommandThread::ExecuteCommandThread(HRP2Client& hrp2)
    :_hrp2(hrp2), _command(0), _quit(false), _mutex(), _cond(), _thread()
{
    pthread_mutex_init(&_mutex, NULL);
    pthread_cond_init(&_cond, NULL);
}

HRP2::ExecuteCommandThread::~ExecuteCommandThread()
{
    pthread_mutex_lock(&_mutex);
    _quit = true;			// 終了命令をセット
    pthread_cond_signal(&_cond);	// 子スレッドに終了命令を送る
    pthread_mutex_unlock(&_mutex);

    pthread_join(_thread, NULL);	// 子スレッドの終了を待つ
    pthread_cond_destroy(&_cond);
    pthread_mutex_destroy(&_mutex);
}

void
HRP2::ExecuteCommandThread::run()
{
    pthread_create(&_thread, NULL, threadProc, this);

    usleep(1000);
}
    
void
HRP2::ExecuteCommandThread::operator ()(Command command) const
{
    pthread_mutex_lock(&_mutex);
    _command = command;
    pthread_cond_signal(&_cond);	// 子スレッドに実行命令を送る
    pthread_mutex_unlock(&_mutex);
}

void
HRP2::ExecuteCommandThread::wait() const
{
    pthread_mutex_lock(&_mutex);
    while (_command != 0)			// 実行が完了するまで
	pthread_cond_wait(&_cond, &_mutex);	// 待つ
    pthread_mutex_unlock(&_mutex);
}

bool
HRP2::ExecuteCommandThread::poll() const
{
    pthread_mutex_lock(&_mutex);
    Command	command = _command;
    pthread_mutex_unlock(&_mutex);

    return command == 0;
}
    
void*
HRP2::ExecuteCommandThread::mainLoop()
{
    pthread_mutex_lock(&_mutex);
    for (;;)
    {
	pthread_cond_wait(&_cond, &_mutex);	// 親からの命令を待つ
	if (_quit)				// スレッド終了命令ならば...
	    break;				// ループを脱出
	else if (_command != 0)			// 実行命令ならば...
	{
	    pthread_mutex_unlock(&_mutex);
	    (_hrp2.*_command)();		// コマンドを実行
	    pthread_mutex_lock(&_mutex);
	    _command = 0;
	    pthread_cond_signal(&_cond);	// 親に実行が完了したことを通知
	}
    }
    pthread_mutex_unlock(&_mutex);

    return 0;
}

void*
HRP2::ExecuteCommandThread::threadProc(void* thread)
{
    ExecuteCommandThread* th = static_cast<ExecuteCommandThread*>(thread);

    return th->mainLoop();
}
    
}

