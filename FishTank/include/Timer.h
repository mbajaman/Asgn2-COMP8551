#ifndef _TIMER_H
#define _TIMER_H

#include <windows.h>
#include <MMSYSTEM.H>



#pragma comment(lib, "winmm.lib")

struct Timer {
	LARGE_INTEGER m_start, m_stop, m_freq;
	Timer() { QueryPerformanceFrequency(&m_freq); }
	void start() { QueryPerformanceCounter(&m_start); }
	double stop()  { QueryPerformanceCounter(&m_stop);  return dt();}
	double dt() { return double(m_stop.QuadPart - m_start.QuadPart) / double(m_freq.QuadPart); }
}; 

#endif