#include <alloca.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include "sales.h"
#include "mytour.h"

const int DEBUG = 0;

float sqr(float x)
{
  return x*x;
}

float dist(const point cities[], int i, int j) {
  return sqrt(sqr(cities[i].x-cities[j].x)+
	      sqr(cities[i].y-cities[j].y));
}

void simple_find_tour(const point cities[], int tour[], int ncities)
{
  int i,j;
  char *visited = alloca(ncities);
  int ThisPt, ClosePt=0;
  float CloseDist;
  int endtour=0;
  
  for (i=0; i<ncities; i++)
    visited[i]=0;
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

/* write the tour out to console */
void write_tour(int ncities, point * cities, int * tour)
{
  int i;
  float sumdist = 0.0;

  /* write out the tour to the screen */
  printf("%d\n", tour[0]);
  for ( i = 1; i < ncities; i++ ) {
    printf("%d\n", tour[i]);
    sumdist += dist(cities, tour[i-1], tour[i]);
  }
  printf("sumdist = %f\n", sumdist);
}

/* write out an encapsulated postscript file of the tour */
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
void initialize_cities(point * cities, int ncities, unsigned seed)
{
  int i;

  srandom(seed);
  for (i=0; i<ncities; i++) {
    cities[i].x = ((float)(random()))/(float)(1U<<31);
    cities[i].y = ((float)(random()))/(float)(1U<<31);
  }
}

int check_tour(const point *cities, int * tour, int ncities)
{
  int * tour2 = malloc(ncities*sizeof(int));
  int i;
  int result = 1;

  simple_find_tour(cities,tour2,ncities);

  for ( i = 0; i < ncities; i++ ) {
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
  point *cities;
  int *tour;
  int seed;
  int tour_okay;
  struct timeval start_time, stop_time;
  long long compute_time;
  

  if (argc!=2) {
    fprintf(stderr, "usage: %s <ncities>\n", argv[0]);
    exit(1);
  }

  /* initialize random set of cities */
  ncities = atoi(argv[1]);
  cities = malloc(ncities*sizeof(point));
  tour = malloc(ncities*sizeof(int));
  seed = 3656384L % ncities;
  initialize_cities(cities, ncities, seed);

  /* find tour through the cities */
  gettimeofday(&start_time, NULL);
  call_student_tour(cities,tour,ncities);
  gettimeofday(&stop_time, NULL);
  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to find tour: %lld microseconds\n", compute_time);

  /* check that the tour we found is correct */
  tour_okay = check_tour(cities,tour,ncities);
  if ( !tour_okay ) {
    fprintf(stderr, "FATAL: incorrect tour\n");
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
