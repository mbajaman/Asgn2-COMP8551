#pragma once

#include "defs.h"
#include <vector>
#include "Bubble.h"
#include "Attractor.h"

typedef std::vector<Bubble *> BubbleVector;
typedef std::vector<Attractor *> AttractorVector;

class Tank
{
	BubbleVector m_bubbles;	
	AttractorVector m_attactors;

public:
	Tank();
	void init(unsigned int nBubbles, int width, int height);
	~Tank(void);
	void iterate(float dt);

	const AttractorVector & attractors() { return m_attactors; }
	const BubbleVector & bubbles() { return m_bubbles; }
};
