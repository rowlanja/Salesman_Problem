#include "mytour.h"
#include <stdio.h>
#include <alloca.h>
#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <sys/time.h>
#include "sales.h"
#include <omp.h>
#include <xmmintrin.h>
#include <limits.h>
void my_tour(const point cities[], int tour[], int ncities)
{

	int i, j;
	char *visited = (char *)calloc(ncities, sizeof(char));
	int ThisPt, ClosePt = 0;
	float CloseDist;
	int endtour = 0;
	struct timeval start_time, stop_time;
	long long compute_time;

	/* ||  find tour through the cities with openPL || */
	float(*costs)[ncities] = malloc(sizeof(float[ncities][ncities]));
	int omp_toggle = 0;
	int omp_switch = 1; // set to 0 for OMP = OFF, set to 1 for OMP = ON
	if (ncities > 2000)
		omp_toggle = omp_switch & 1;
		// Spawns threads
		// this section calculates the distance
		// OMP for directive instructs the compiler to distribute loop iterations within the team of threads
		// The omp simd directive is applied to a loop to indicate that multiple iterations of the loop can be executed concurrently by using SIMD instructions.
		// faster calculation of distance using a more approximate yet accurate calculation
	#pragma omp parallel if (omp_toggle)
	{
		#pragma omp for
		for (int i = 0; i < ncities; i++)
		{
			#pragma omp simd
			for (int j = 0; j < ncities; j++)
			{
				float n = (((cities[i].x - cities[j].x) * (cities[i].x - cities[j].x)) +
						   ((cities[i].y - cities[j].y) * (cities[i].y - cities[j].y)));
				const int result = 0x1fbb4000 + (*(int *)&n >> 1);
				costs[i][j] = *(float *)&result;
			}
		}
	}
	visited[ncities - 1] = 1;
	tour[0] = ncities - 1;
	int source = ncities - 1;
	int min_index = 0;
	float min_cost = DBL_MAX;
	float cost = DBL_MAX;
	//now generate tour
	//tour generation begins at ncities-1 as the source
	//list of costs associated with source extracted from costs[source][]
	//index (city number) of minimum cost is saved, source of next step in tour begins from new saved index
	//set city to be visited
	//begin new step of tour generation
	for (i = 1; i < ncities; i++)
	{
		for (j = 0; j < ncities; j++)
		{
			cost = costs[source][j];
			if (cost < min_cost && (visited[j] != 1))
			{
				min_index = j;
				min_cost = cost;
			}
		}
		min_cost = DBL_MAX;
		tour[i] = min_index;
		source = min_index;
		visited[min_index] = 1;
	}
	free(costs);
}
