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

const int DEBUG = 0;

float sqr(float x)
{
  return x*x;
}

float dist(const point cities[], int i, int j) {
  return sqrt(sqr(cities[i].x-cities[j].x)+
	      sqr(cities[i].y-cities[j].y));
}


/*
  Approximate distance between two points using floating point bit manupulation in 
  place of square root for the solution to hypotenuse.

  result = sqrt( )

*/
float approx_dist(const point cities[], int i, int j){
  
  float n = (((cities[i].x - cities[j].x) * (cities[i].x - cities[j].x)) + ((cities[i].y - cities[j].y) * (cities[i].y - cities[j].y)));
  const int result = 0x1fbb4000 + (*(int*)&n >> 1);
    return *(float*)&result; 
}

// sequential code without OpenMP
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
// CODE TO EDIT //
// TEST CODE ONE PART AT A TIME, AT TIME TO EXECUTE ABOVE THE PARALLEL INSTRUCTION
/* this is the sample code but with openMP concurrent tools added */
void simple_find_tour_concur(const point cities[], int tour[], int ncities)
{
  
  int i,j;
  char *visited = (char*)calloc(ncities, sizeof(char));
  int ThisPt, ClosePt=0;
  float CloseDist;
  int endtour=0;
  /* ||  find tour through the cities with openPL || */
  
  ThisPt = ncities-1;
  visited[ncities-1] = 1;
  tour[endtour++] = ncities-1;

  struct timeval start_time, stop_time;
  long long compute_time;
  int done = 0;
  float * distances = calloc(12, sizeof(float));
  
  float costs[ncities][ncities];
  
  for (i=1; i<ncities; i++) {
    CloseDist = DBL_MAX;
    
      for (j=0; j<ncities-1; j += 12) {
        distances[0] = DBL_MAX;
        distances[1] = DBL_MAX;
        distances[2] = DBL_MAX;
        distances[3] = DBL_MAX;
        distances[4] = DBL_MAX;
        distances[5] = DBL_MAX;
        distances[6] = DBL_MAX;
        distances[7] = DBL_MAX;
        distances[8] = DBL_MAX;
        distances[9] = DBL_MAX;
        distances[10] = DBL_MAX;
        distances[11] = DBL_MAX;
        
        if( i==2 && j == 0)gettimeofday(&start_time, NULL);
        #pragma omp parallel sections firstprivate(cities, visited)
          { 
            #pragma omp section
            {
              
              if(j < ncities-1){
                if (!visited[j]) {
                  distances[0] = approx_dist(cities, ThisPt, j);
                  
                }
              }

            }
            #pragma omp section
            {
              if(j+1 < ncities-1){
                if (!visited[j+1]) {
                  distances[1] = approx_dist(cities, ThisPt, j+1);
                  
                }
              }
            }
            #pragma omp section
            {
              if(j+2 < ncities-1){
                if (!visited[j+2]) {
                  distances[2] = approx_dist(cities, ThisPt, j+2);
                  
                }
              }
            }
            #pragma omp section
            {
              if(j+3 < ncities-1){
                if (!visited[j+3]) {
                  distances[3] = approx_dist(cities, ThisPt, j+3);

                }
              }
            }
            #pragma omp section
            {
              if(j+4 < ncities-1){
                if (!visited[j+4]) {
                  distances[4] = approx_dist(cities, ThisPt, j+4);

                }
              }
            }
            #pragma omp section
            {
              if(j+5 < ncities-1){
                if (!visited[j+5]) {
                  distances[5] = approx_dist(cities, ThisPt, j+5);

                }
              }
            }
            #pragma omp section
            {
              if(j+6 < ncities-1){
                if (!visited[j+6]) {
                  distances[6] = approx_dist(cities, ThisPt, j+6);

                }
              }
            }
            #pragma omp section
            {
              if(j+7 < ncities-1){
                if (!visited[j+7]) {
                  distances[7] = approx_dist(cities, ThisPt, j+7);

                }
              }
            }
            #pragma omp section
            {
              if(j+8 < ncities-1){
                if (!visited[j+8]) {
                  distances[8] = approx_dist(cities, ThisPt, j+8);

                }
              }
            }
            #pragma omp section
            {
              if(j+9 < ncities-1){
                if (!visited[j+9]) {
                  distances[9] = approx_dist(cities, ThisPt, j+9);

                }
              }
            }
            #pragma omp section
            {
              if(j+10 < ncities-1){
                if (!visited[j+10]) {
                  distances[10] = approx_dist(cities, ThisPt, j+10);

                }
              }
            }
            #pragma omp section
            {
              if(j+11 < ncities-1){
                if (!visited[j+11]) {
                  distances[11] = approx_dist(cities, ThisPt, j+11);

                }
              }
            }
        }
        
        if(i==2 && j ==0)gettimeofday(&stop_time, NULL);
        #pragma omp single
        {
          float min = DBL_MAX;
          int min_index = 0;
          for (int i = 0; i<12; i++)
          {
            if(min > distances[i]){
              min_index = i;
              min = distances[i];
            }
          }

          if (min < CloseDist) 
          {
            CloseDist = min;
            ClosePt = j+min_index;
          }
        }
        
      }
    tour[endtour++] = ClosePt;
    visited[ClosePt] = 1;
    ThisPt = ClosePt;
  }
  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to execute last parallel section: %lld microseconds\n", compute_time);
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
    // cities_con[i].x = point.x;    //making a fresh copy of the city for the concurrent path finder
    // cities_con[i].y = point.y;    //making a fresh copy of the city for the concurrent path finder
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
  simple_find_tour_concur(cities,tourConcur,ncities); // use a new tour array
  gettimeofday(&stop_time, NULL);        // use a new stop time for concur

  compute_time = (stop_time.tv_sec - start_time.tv_sec) * 1000000L +
    (stop_time.tv_usec - start_time.tv_usec);
  printf("Time to find tour with openMP: %lld microseconds\n", compute_time);  /* check that the tour we found is correct */

  /* || find tour through the cities without openPL || */


  tour_okay = check_tour(cities,tour,ncities);
  tour_okay_concur = check_tour(cities,tourConcur,ncities);
  if ( !tour_okay || !tour_okay_concur) {
    fprintf(stderr, "FATAL: incorrect tour in either sequential tour generation or concur tour generation\n");
    printf("%d %d\n", tour_okay, tour_okay_concur );
  }

  /* write out results */
  if ( DEBUG ) {
    //write_eps_file(ncities, cities, tour);
    write_tour(ncities, cities, tour);
    write_tour(ncities, cities, tourConcur);
  }

  free(cities);
  free(tour);
  return 0;
}
