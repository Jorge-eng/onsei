#include <stdio.h>


#define LOG(...) \
    fprintf (stdout,"%s:%d  ",__FILE__,__LINE__); fprintf (stdout, __VA_ARGS__); fprintf(stdout,"\n")
