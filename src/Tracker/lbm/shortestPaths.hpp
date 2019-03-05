#ifndef __SHORTEST_PATHS_H__
#define __SHORTEST_PATHS_H__

#include "global.hpp"

struct Edge {
  unsigned src;
  unsigned dst;
  float w;

  Edge();
  Edge(unsigned src, unsigned dst, float w);
  void print();
};

int BellmanFord(std::vector<Edge> &E, unsigned nn_size, unsigned src,
                unsigned dst,
                std::pair<std::vector<uint>, float> &shortest_path);

// TODO: valgrind shows memory issues with YenKShortestPath code so fix them
//      maybe just isolate this function away from the whole application and
//      test it
//      individually
void YenKShortestPaths(
    std::vector<Edge> &E, unsigned nn_size, unsigned src, unsigned dst,
    unsigned K,
    std::vector<std::pair<std::vector<uint>, float>> &shortest_paths);

void EppsteinKShortestPaths(
    std::vector<Edge> &E, unsigned nn_size, unsigned src, unsigned dst,
    unsigned K,
    std::vector<std::pair<std::vector<uint>, float>> &shortest_paths);

#endif
