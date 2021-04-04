#include <alloca.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include "sales.h"
#include "mytour.h"
#include <omp.h>
#include <xmmintrin.h>
#include <emmintrin.h>
#include <limits.h>
typedef struct {
  int val;
  int index;
} Compare;

int main(int argc, char *argv[])
{
  int theArray[10] = {11,22,33,66,10,2,3,4,9,14};
  int size = 10;
  Compare min = { .val = theArray[0], .index = 0};

  //#pragma omp declare reduction(minimum : struct Compare : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out)
  #pragma omp declare reduction(minimum : Compare :               \
    omp_out = omp_in.val < omp_out.val ? omp_in : omp_out)        \
    initializer(omp_priv = {INT_MAX, 0})
  printf("min val %d\n",min.val);
  #pragma omp parallel for reduction(minimum:min)
  for(int i = 1; i<size; i++) {
      printf("%d : %d\n",theArray[i],min.val);
      if(theArray[i]<min.val) {
          min.val = theArray[i];
          min.index = i;
          printf("min : %d\n",min.val);
      }
  }
  printf("min : %d\n",min.val);
  return 0;
}
/*
int theArray[10] = {11,22,33,66,1,2,3,4,9,14};
int size = 10;
int index = 0;
int min = theArray[0];
#pragma omp parallel
{
    int index_local = index;
    float min_local = min;
    #pragma omp for nowait
    for (int i = 1; i < size; i++) {
        if (theArray[i] < min_local) {
            min_local = theArray[i];
            index_local = i;
        }
    }
    #pragma omp critical
    {
        if (min_local < min) {
            min = min_local;
            index = index_local;
        }
    }
}
printf("min : %d\n",min);
return 0;

*/
