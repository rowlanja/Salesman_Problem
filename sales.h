// sales.h
// header file for travelling salesperson problem

#ifndef SALES_H
#define SALES_H

typedef struct {
  float x;
  float y;
} point;

void simple_find_tour(const point cities[], int tour[], int ncities);

#endif // SALES_H
