#pragma once

#include "defs.h"
#include "Tank.h"


class TankDemo
{
	UCImg *m_backgroundImage;
	UCImg *m_backstoreImage;
	UCImg *m_bubbleImage;
	UCImg *m_attractorImage;
	Tank m_tank;	
public:
	enum DisplayFlags { DRAW_BUBBLES = 0x1, DRAW_ATTRACTORS = 0x2};
	TankDemo(void);
	~TankDemo(void);
	int init(unsigned int nBubbles,
			const char *backgroundImageName,
			const char *bubbgleImageName,
			const char *attractorImageName);
	void frame(cimg_library::CImgDisplay *display, DisplayFlags flags, SimdMode mode, float dt);
	const UCImg *backgroundImage() { return m_backgroundImage; }
};
