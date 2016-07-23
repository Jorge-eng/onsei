#ifndef _TINYTENSOR_MEMORY_H_
#define _TINYTENSOR_MEMORY_H_

#ifdef __cplusplus
extern "C" {
#endif

#define DESKTOP
#define UNIT_TESTING
    
#ifdef DESKTOP
#include <memory.h>
#include <string.h>
#include <stdlib.h>

#ifdef UNIT_TESTING

    static inline void __setrand(void * p, const uint32_t n) {
        int i;
        int8_t * p2 = (int8_t *) p;
        for (i = 0; i < n; i++) {
            *p2++ = (int8_t)rand();
        }
    }
    
    static inline void * randalloc(const size_t n) {
        void * p = malloc(n);
        __setrand(p,n);
        return p;
    }
    
    
  #define MALLOC(x) randalloc(x)
#else
  #define MALLOC(x) malloc(x)
#endif


#define FREE(x) free(x)
#define MEMCPY(tgt,src,size) memcpy(tgt,src,size)
#define MEMSET(tgt,val,size) memset(tgt,val,size)
#else
#error "MEMORY IMPLEMENTATION FOR THIS PLATFORM NEEDS TO BE DEFINED"
#endif
    
#ifdef __cplusplus
}
#endif

#endif //_TINYTENSOR_MEMORY_H_
