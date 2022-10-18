#include "Tank.h"
#include "SpringAttractor.h"
#include "GravityAttractor.h"
#include "WallAttractor.h"

Tank::Tank()
{
}

void Tank::init(unsigned int nBubbles, int width, int height)
{

	/**
	// 4 springs configuration
	for (int x = 0; x < 2; x++) {
		for (int y = 0; y < 2; y++) {
			SpringAttractor *spring = new SpringAttractor(0.0001);
			spring->setPosition(x * width / 2.0f + width / 4.0f, y * height / 2.0f + height / 4.0f);
			m_attactors.push_back(spring);
		}
	}
	**/
	SpringAttractor *spring = new SpringAttractor(0.0001);
	spring->setPosition(width / 2.0f, height / 2.0f);
	m_attactors.push_back(spring);

	GravityAttractor *g = new GravityAttractor(0.0001);
	g->setPosition(0.0f, 0.0f);
	m_attactors.push_back(g);
/**
	WallAttractor *wall = new WallAttractor(width, height);
	m_attactors.push_back(wall);
**/
	for (unsigned int i = 0; i < nBubbles; i++) {
		Bubble *b = new Bubble;
		b->setRandomPosition(width, height);
		b->setRandomVelocity(5, 5);
		m_bubbles.push_back(b);
	}
}


Tank::~Tank(void)
{
	for (BubbleVector::iterator b = m_bubbles.begin(); b != m_bubbles.end(); b++) {
		delete *(b);
	}

	for (AttractorVector::iterator a = m_attactors.begin(); a != m_attactors.end(); a++) {
		delete *(a);
	}

}

/// calculate the new positions of the bubbles.
/// calculate new positions for the attractors;
void Tank::iterate(float dt)
{
	int n = m_attactors.size();

	for (BubbleVector::iterator b = m_bubbles.begin(); b != m_bubbles.end(); b++) {
		for (AttractorVector::iterator a = m_attactors.begin(); a != m_attactors.end(); a++) {
			float fx, fy;
			(*a)->f(*(*b), &fx, &fy);
			(*b)->applyForce(fx, fy);
			(*b)->move(dt);
		}
	}
}