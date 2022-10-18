#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "CImg.h"
#include <windows.h>
#include <MMSYSTEM.H>
#include <emmintrin.h>
#include "TankDemo.h"

#pragma comment(lib, "winmm.lib")

struct Timer {
	LARGE_INTEGER m_start, m_stop, m_freq;
	Timer() { QueryPerformanceFrequency(&m_freq); }
	void start() { QueryPerformanceCounter(&m_start); }
	double stop() { QueryPerformanceCounter(&m_stop);  return dt(); }
	double dt() { return double(m_stop.QuadPart - m_start.QuadPart) / double(m_freq.QuadPart); }
};

#pragma warning(disable:4018) // signed/unsigned mismatch
using namespace cimg_library;

typedef struct {
	const char *backgroundImageName;
	const char *bubbleImageName;
	const char *attractorImageName;
	SimdMode mode;
	float simStep;
	TankDemo::DisplayFlags flags;
	unsigned int nBubbles;
	DWORD threadId;
} BubbleThreadParams;

unsigned int tankDemoThread(void *data)
{
	BubbleThreadParams *params = (BubbleThreadParams *)data;

	TankDemo tankDemo;

	tankDemo.init(params->nBubbles, params->backgroundImageName, params->bubbleImageName, params->attractorImageName);
	CImgDisplay display(*tankDemo.backgroundImage());


	int frame_counter = 0;
	while (1) {
		tankDemo.frame(&display, params->flags, params->mode, params->simStep);
		switch (params->mode) {
		case SIMD_EMMX_INTRINSICS:
			display.set_title("MODE: %s (%d)", "intrinsics", frame_counter++);
			break;
		case SIMD_EMMX:
			display.set_title("MODE: %s (%d)", "mmx", frame_counter++);
			break;
		default:
			display.set_title("MODE: %s (%d)", "serial", frame_counter++);
			break;
		}

		if (display.is_keyESC()) {
			break;
		}
	}
	return 0;
}

void startTankDemoThreads(const char *background, const char *bubble, const char *attractors)
{
	BubbleThreadParams params[3];
	HANDLE threadHandles[3];

	for (int i = 0; i < 3; i++) {
		params[i].backgroundImageName = background;
		params[i].bubbleImageName = bubble;
		params[i].attractorImageName = attractors;
		params[i].flags = TankDemo::DRAW_BUBBLES;
		params[i].simStep = 0.1f;
		params[i].nBubbles = 500;
		switch (i) {
		case 0:
			params[i].mode = SIMD_EMMX_INTRINSICS;
			break;
		case 1:
			params[i].mode = SIMD_EMMX;
			break;
		default:
			params[i].mode = SIMD_NONE;
			break;
		}
		threadHandles[i] = CreateThread(NULL, 0, LPTHREAD_START_ROUTINE(tankDemoThread), &params[i], 0, &params[i].threadId);
		if (threadHandles[i] == NULL) {
			MessageBox(NULL, "Couldn't start thread", "FATAL", MB_OK);
			exit(1);
		}
	}

	WaitForMultipleObjects(3, threadHandles, FALSE, INFINITE);

	for (int i = 0; i < 3; i++) TerminateThread(threadHandles[i], 0);
}



int main(int argc, char* argv[])
{
	int optarg = 1;
	while (optarg < argc) {
		if ((argc - optarg) < 2) {
			fprintf(stderr, "Usage: %s <background image> <bubble image>\n", argv[0]);
			exit(1);
		}
		else {
			startTankDemoThreads(argv[optarg], argv[optarg + 1], NULL);
			optarg += 2;
		}
	}

	return 0;
}
