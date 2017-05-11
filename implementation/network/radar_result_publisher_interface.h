#ifndef _RADARRESULTPUBLISHERINTERFACE_H_
#define _RADARRESULTPUBLISHERINTERFACE_H_



#include "haltija_types.h"


class RadarResultPublisherInterface {
public:

    virtual ~RadarResultPublisherInterface() {};

    virtual void publish(const char * prefix, const RadarMessage_t & message) = 0;

};


#endif //_RADARRESULTPUBLISHERINTERFACE_H_


