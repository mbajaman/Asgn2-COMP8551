#include "TankDemo.h"
#include "ImageOperators.h"

TankDemo::TankDemo(void)
{
	m_backgroundImage = m_backstoreImage = m_bubbleImage = m_attractorImage = NULL;
}

TankDemo::~TankDemo(void)
{
	if (m_backgroundImage != NULL) delete m_backgroundImage;
	if (m_backstoreImage != NULL) delete m_backstoreImage;
	if (m_bubbleImage != NULL) delete m_bubbleImage;
	if (m_attractorImage != NULL) delete m_attractorImage;

}

int TankDemo::init(unsigned int nBubbles, 
				   const char *backgroundImageName, 
				   const char *bubbleImageName, 
				   const char *attractorImageName)
{
	m_backgroundImage = (backgroundImageName != NULL) ? new UCImg(backgroundImageName) : NULL;
	m_backstoreImage = (backgroundImageName != NULL) ? new UCImg(backgroundImageName) : NULL;
	m_bubbleImage = (bubbleImageName != NULL) ? new UCImg(bubbleImageName) : NULL;
	m_attractorImage = (attractorImageName != NULL) ? new UCImg(attractorImageName) : NULL;

	m_tank.init(nBubbles, m_backgroundImage->width(), m_backgroundImage->height());
	return 0;
}

void TankDemo::frame(cimg_library::CImgDisplay *display, DisplayFlags flags, SimdMode mode, float dt)
{
	m_tank.iterate(dt);

	if (m_backgroundImage == NULL) return;

	// clean the original image;
	m_backgroundImage->assign(*m_backstoreImage);


	if ((flags & DRAW_BUBBLES) && m_bubbleImage != NULL) {
		const BubbleVector & bubbles = m_tank.bubbles();

		for (BubbleVector::const_iterator b = bubbles.begin(); b != bubbles.end(); b++) {
			blitBlend(*m_bubbleImage, *m_backgroundImage, (*b)->x(), (*b)->y(), mode); 
		}
	}

	if (flags & DRAW_ATTRACTORS && m_attractorImage != NULL) {
		const AttractorVector & att = m_tank.attractors();

		for (AttractorVector::const_iterator a = att.begin(); a != att.end(); a++) {
			blitBlend(*m_attractorImage, *m_backgroundImage, (*a)->x(), (*a)->y(), mode);
		}
	}

	display->display(*m_backgroundImage);
}