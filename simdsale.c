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
const int DEBUG = 0;

typedef struct {
  float CloseDist;
  int ClosePt;
} Compare;

float sqr(float x)
{
  return x*x;
}

float dist(const point cities[], int i, int j) {
  float x =  sqr(cities[i].x-cities[j].x)+ sqr(cities[i].y-cities[j].y);
  x = sqrt(x);
  return x;
}

__m128 approx_dist(float *cityx,float *cityy, int j, int ThisPt){
   __m128 ix, iy, jxs, jys, xdiff, ydiff, xdiffSqrd, ydiffSqrd, xyAddedDiff, xySqrt;
   ix = _mm_set1_ps(cityx[ThisPt]);
   iy = _mm_set1_ps(cityy[ThisPt]);
   jxs = _mm_load_ps(&cityx[j]);
   jys = _mm_load_ps(&cityy[j]);
   xdiff = _mm_sub_ps(ix, jxs);
   ydiff = _mm_sub_ps(iy, jys);
   xdiffSqrd = _mm_mul_ps(xdiff, xdiff);
   ydiffSqrd = _mm_mul_ps(ydiff, ydiff);
   xyAddedDiff = _mm_add_ps(xdiffSqrd, ydiffSqrd);
   xySqrt = _mm_sqrt_ps(xyAddedDiff);
   return xySqrt;
}

void simple_find_tour_concur(const point cities[], int tour[], int ncities) {
      int i, j;
      int ThisPt, ClosePt = 0;
      float CloseDist;
      int endTour = 0;
      int alignedLength = ncities + (4 - (ncities % 4));
      float *cityx = malloc(sizeof(float) * alignedLength);
      float *cityy = malloc(sizeof(float) * alignedLength);
      float visited[alignedLength];
      __m128 v0 = _mm_set1_ps(0);
      #pragma omp parallel for
      for (i = 0; i < ncities; i++) {
          if (i % 4 == 0) {
              _mm_store_ps(&visited[i], v0);
          }
          cityx[i] = cities[i].x;
          cityy[i] = cities[i].y;
      }
      ThisPt = ncities - 1;
      visited[ncities - 1] = 1;
      tour[endTour++] = ncities - 1;

      // Determine the tour
      __m128 ix, iy, jxs, jys, xdiff, ydiff, xdiffSqrd, ydiffSqrd;
      float *costs = malloc(sizeof(float) * alignedLength);
      #pragma omp parallel for private(j, ix, iy, jxs, jys, xdiff, ydiff, xdiffSqrd, ydiffSqrd)
      for (j = 0; j < alignedLength; j += 4) {
          ix = _mm_set1_ps(cityx[ThisPt]);
          iy = _mm_set1_ps(cityy[ThisPt]);
          jxs = _mm_load_ps(&cityx[j]);
          jys = _mm_load_ps(&cityy[j]);
          xdiff = _mm_sub_ps(ix, jxs);
          ydiff = _mm_sub_ps(iy, jys);
          xdiffSqrd = _mm_mul_ps(xdiff, xdiff);
          ydiffSqrd = _mm_mul_ps(ydiff, ydiff);
          _mm_store_ps(&costs[j], _mm_sqrt_ps((_mm_add_ps(xdiffSqrd, ydiffSqrd))));
      }
      for (i = 1; i < ncities; i++) {
          CloseDist = DBL_MAX;
          for (j = 0; j < ncities - 1; j++) {
              if (!visited[j] && costs[j] < CloseDist) {
                  CloseDist = costs[j];
                  ClosePt = j;
              }
          }
          tour[endTour++] = ClosePt;
          visited[ClosePt] = 1;
          ThisPt = ClosePt;
      }
      free(costs);
}

void simple_find_tour(const point cities[], int tour[], int ncities)
{
  int i,j;
  char *visited = alloca(ncities);
  int ThisPt, ClosePt=0;
  float CloseDist;
  int endtour=0;
  for (i=0; i<ncities; i++) {
    visited[i]=0;
  }
  ThisPt = ncities-1;
  visited[ncities-1] = 1;
  tour[endtour++] = ncities-1;

  for (i=1; i<ncities; i++) {
    CloseDist = DBL_MAX;
    for (j=0; j<ncities-1; j++) {
      if (!visited[j]) {
      	if (dist(cities, ThisPt, j) < CloseDist) {
          CloseDist = dist(cities, ThisPt, j);
          ClosePt = j;
        }
      }
    }
    tour[endtour++] = ClosePt;
    visited[ClosePt] = 1;
    ThisPt = ClosePt;
  }
}
void write_tour(int ncities, point * cities, int * tour)
{
  int i;
  float sumdist = 0.0;
  printf("%d\n", tour[0]);
  for ( i = 1; i < ncities; i++ ) {
    printf("%d\n", tour[i]);
    sumdist += dist(cities, tour[i-1], tour[i]);
  }
  printf("sumdist = %f\n", sumdist);
}

void write_eps_file(int ncities, point *cities, int *tour)
{
  FILE *psfile;
  int i;

  psfile = fopen("sales.eps","w");
  fprintf(psfile, "%%!PS-Adobe-2.0 EPSF-1.2\n%%%%BoundingBox: 0 0 300 300\n");
  fprintf(psfile, "1 setlinejoin\n0 setlinewidth\n");
  fprintf(psfile, "%f %f moveto\n",
	  300.0*cities[tour[0]].x, 300.0*cities[tour[0]].y);
  for (i=1; i<ncities; i++) {
    fprintf(psfile, "%f %f lineto\n",
	    300.0*cities[tour[i]].x, 300.0*cities[tour[i]].y);
  }
  fprintf(psfile,"stroke\n");
}

/* create a random set of cities */
void initialize_cities(point * cities, point * cities_con, int ncities, unsigned seed)
{
  int i;
  point point;
  srandom(seed);
  for (i=0; i<ncities; i++) {
    point.x = ((float)(random()))/(float)(1U<<31);
    point.y = ((float)(random()))/(float)(1U<<31);
    cities[i].x = point.x;
    cities[i].y = point.y;
    cities_con[i].x = point.x;    //making a fresh copy of the city for the concurrent path finder
    cities_con[i].y = point.y;    //making a fresh copy of the city for the concurrent path finder
  }
}

int check_tour(const point *cities, int * tour, int ncities)
{
  int * tour2 = malloc(ncities*sizeof(int));

  int i;
  int result = 1;
  simple_find_tour(cities,tour2,ncities);
  //printf("checking\n");
  for ( i = 0; i < ncities; i++ ) {
    //printf("%d,%d\n",tour[i],tour2[i]);
    if ( tour[i] != tour2[i] ) {
      result = 0;
    }
  }
  free(tour2);
  return result;
}

void call_student_tour(const point *cities, int * tour, int ncities)
{
  my_tour(cities, tour, ncities);
}

int main(int argc, char *argv[])
{
  int i, ncities;
  point *cities, *cities_con;
  int *tour, *tourConcur;
  int seed;
  int tour_okay, tour_okay_concur;
  struct timeval start_time, stop_time;
  long long compute_time;


  if (argc!=2) {
    fprintf(stderr, "usage: %s <ncities>\n", argv[0]);
    exit(1);
  }

  /* initialize random set of cities */
  ncities = atoi(argv[1]);
  cities = malloc(ncities*sizeof(point));
  cities_con = malloc(ncities*sizeof(point));
  tour = malloc(ncities*sizeof(int));
  tourConcur = malloc(ncities*sizeof(int));
  seed = 3656384L % ncities;
  initialize_cities(cities, cities_con, ncities, seed);
  gettimeofday(&start_time, NULL);
  simple_find_tour(cities, tour, ncities);
  gettimeofday(&stop_time, NULL);

  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to find tour without openMP: %lld microseconds\n", compute_time);


  /* ||  find tour through the cities with openPL || */
  gettimeofday(&start_time, NULL);       // udr a new start time for concur
  simple_find_tour_concur(cities_con,tourConcur,ncities); // use a new tour array
  gettimeofday(&stop_time, NULL);        // use a new stop time for concur

  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to find tour with openMP: %lld microseconds\n", compute_time);  /* check that the tour we found is correct */

  /* || find tour through the cities without openPL || */


  tour_okay = check_tour(cities,tour,ncities);
  tour_okay_concur = check_tour(cities_con,tourConcur,ncities);

  //
  // for(int loop = 0; loop < 50; loop++){
  //   printf("%d : %d \n", tour[loop], tourConcur[loop]);
  // }
  if ( !tour_okay || !tour_okay_concur) {
    fprintf(stderr, "FATAL: incorrect tour in either sequential tour generation or concur tour generation\n");
    printf("%d %d\n", tour_okay, tour_okay_concur );
  }
  else {
    printf("passed checks\n");
  }

  /* write out results */
  if ( DEBUG ) {
    write_eps_file(ncities, cities, tour);
    write_tour(ncities, cities, tour);
  }

  free(cities);
  free(tour);
  return 0;
}
