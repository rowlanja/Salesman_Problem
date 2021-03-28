#include "mytour.h"
#include <stdio.h>

void my_tour(const point cities[], int tour[], int ncities)
{
	simple_find_tour(cities, tour, ncities);
	/*
	if(ncities < 10){
		printf("[*] executing sequential pathfinder");
		simple_find_tour(cities, tour, ncities);
	}
	else{
		printf("[*] executing concurrent pathfinder");
		concur_find_tour(cities, tour, ncities);
	}
  */
}
void my_tour_concur(const point cities[], int tour[], int ncities)
{
	simple_find_tour_concur(cities, tour, ncities);
	/*
	if(ncities < 10){
		printf("[*] executing sequential pathfinder");
		simple_find_tour(cities, tour, ncities);
	}
	else{
		printf("[*] executing concurrent pathfinder");
		concur_find_tour(cities, tour, ncities);
	}
  */
}
