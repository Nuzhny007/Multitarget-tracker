#include "shortestPaths.hpp"
#include "graehl/graph.h"

Edge::Edge() : src(0), dst(0), w(0) {}

Edge::Edge(unsigned src, unsigned dst, float w) : src(src), dst(dst), w(w) {}

void Edge::print() {
  std::cout << "src:" << src << "; dst:" << dst << "; w:" << w << std::endl;
}

int BellmanFord(std::vector<Edge> &E, unsigned nn_size, unsigned src,
                unsigned dst,
                std::pair<std::vector<uint>, float> &shortest_path) {
  assert(!shortest_path.first.size());
  std::vector<float> dist(nn_size);
  std::vector<unsigned> parent(nn_size);
  for (auto &v : dist) {
    v = INF;
  }
  dist[src] = 0;

  for (unsigned i = 0; i < nn_size - 1; i++) {
    for (auto &e : E) {
      if (dist[e.src] + e.w < dist[e.dst]) {
        dist[e.dst] = dist[e.src] + e.w;
        parent[e.dst] = e.src;
      }
    }
  }

  // check for negative cycle
  for (auto &e : E) {
    if (dist[e.src] + e.w < dist[e.dst]) {
      return -1;
    }
  }

  if (dist[dst] == INF || !E.size()) {
    return 0;
  }

  shortest_path.second = dist[dst];
  shortest_path.first.emplace_back(dst);
  while (1) {
    if (shortest_path.first[shortest_path.first.size() - 1] == src) {
      break;
    } else {
      shortest_path.first.emplace_back(
          parent[shortest_path.first[shortest_path.first.size() - 1]]);
    }
  }
  std::reverse(shortest_path.first.begin(), shortest_path.first.end());

  return 0;
}

void YenKShortestPaths(
    std::vector<Edge> &E, unsigned nn_size, unsigned src, unsigned dst,
    unsigned K,
    std::vector<std::pair<std::vector<uint>, float>> &shortest_paths) {
  // determine the shortest path from src to dst
  std::pair<std::vector<uint>, float> the_shortest_path;

  int status = BellmanFord(E, nn_size, src, dst, the_shortest_path);
  if (status == -1) {
    // negative weight cycle
    assert(0);
  } else if (the_shortest_path.first.size()) {
    // shortest path exist
    shortest_paths.emplace_back(the_shortest_path);
    if (K == 1) {
      return;
    }

    // finding the next K-1 shortest paths
    std::vector<std::pair<std::vector<uint>, float>> potential_shortest_paths;
    while (1) {
      std::vector<uint> current_shortest_path =
          (shortest_paths.end() - 1)->first;
      for (size_t i = 0; i < current_shortest_path.size() - 1; i++) {
        // current_shortest_path[i] is the spur node for this iteration

        // creating root_path from src uptill this spur node
        std::pair<std::vector<uint>, float> root_path;
        root_path.first.insert(root_path.first.begin(),
                               current_shortest_path.begin(),
                               current_shortest_path.begin() + i + 1);
        root_path.second = 0;
        for (int i = 0; i < (int)(root_path.first.size() - 1); i++) {
          for (auto &e : E) {
            if (e.src == root_path.first[i] &&
                e.dst == root_path.first[i + 1]) {
              root_path.second += e.w;
            }
          }
        }
        // creating a new node network for finding spur path
        std::vector<Edge> E_tmp = E;
        // checking previous shortest paths if they share this rootpath
        for (auto &sp : shortest_paths) {
          std::vector<uint> tmp(sp.first.begin(), sp.first.begin() + i + 1);
          if (root_path.first == tmp) {
            // if they share then removing their 1st spur edge from node
            // network
            for (auto &e : E_tmp) {
              if (e.src == current_shortest_path[i] &&
                  e.dst == sp.first[i + 1]) {
                e.w = INF;
                break;
              }
            }
          }
        }
        // also checking previous potential_shortest_paths as we dont need
        // to recompute them if there is atleast one path already computed
        // on that path
        for (auto &sp : potential_shortest_paths) {
          std::vector<uint> tmp(sp.first.begin(), sp.first.begin() + i + 1);
          if (root_path.first == tmp) {
            // if they share then removing their 1st spur edge from node
            // network
            for (auto &e : E_tmp) {
              if (e.src == current_shortest_path[i] &&
                  e.dst == sp.first[i + 1]) {
                e.w = INF;
                break;
              }
            }
          }
        }
        E_tmp.erase(std::remove_if(E_tmp.begin(), E_tmp.end(),
                                   [](const Edge &e) { return (e.w == INF); }),
                    E_tmp.end());
        // removing rootpath nodes from node network except this spur node
        for (auto &n : root_path.first) {
          E_tmp.erase(
              std::remove_if(E_tmp.begin(), E_tmp.end(),
                             [&n, &current_shortest_path, i](const Edge &e) {
                return (e.src == n && n != current_shortest_path[i]);
              }),
              E_tmp.end());
        }
        std::pair<std::vector<uint>, float> this_spur_path;
        status = BellmanFord(E_tmp, nn_size, current_shortest_path[i], dst,
                             this_spur_path);
        if (!status) {
          if (this_spur_path.first.size()) {
            // concatenating root_path to make complete path
            this_spur_path.first.insert(this_spur_path.first.begin(),
                                        root_path.first.begin(),
                                        root_path.first.end() - 1);
            this_spur_path.second += root_path.second;

            potential_shortest_paths.emplace_back(this_spur_path);
          }
        } else {
          // negative weight cycle
          assert(0);
        }
      }

      // got all the potential shortest paths upto now..now selecting the next
      // shortest path from here
      if (!potential_shortest_paths.size()) {
        break;
      } else {
        float min_cost = INF;
        size_t min_cost_id = 0;
        // finding the least cost path from potential_shortest_paths
        for (size_t i = 0; i < potential_shortest_paths.size(); i++) {
          if (min_cost > potential_shortest_paths[i].second) {
            min_cost = potential_shortest_paths[i].second;
            min_cost_id = i;
          }
        }

        shortest_paths.emplace_back(potential_shortest_paths[min_cost_id]);
        potential_shortest_paths.erase(potential_shortest_paths.begin() +
                                       min_cost_id);

        if (shortest_paths.size() == K) {
          break;
        }
      }
    }
  }
}

void EppsteinKShortestPaths(
    std::vector<Edge> &E, unsigned nn_size, unsigned src, unsigned dst,
    unsigned K,
    std::vector<std::pair<std::vector<uint>, float>> &shortest_paths) {
  Graph g;
  g.nStates = nn_size;

  g.states = new GraphState[nn_size];
  GraphArc tmp;
  tmp.data = NULL;
  for (auto &e : E) {
    tmp.source = e.src;
    tmp.dest = e.dst;
    tmp.weight = e.w;

    g.states[e.src].arcs.push(tmp);
  }

  List<List<GraphArc *>> *paths = bestPaths(g, src, dst, K);

  for (ListIter<List<GraphArc *>> pathIter(*paths); pathIter; ++pathIter) {
    std::pair<std::vector<uint>, float> sp;
    sp.second = 0;
    GraphArc *a = NULL;
    for (ListIter<GraphArc *> arcIter(pathIter.data()); arcIter; ++arcIter) {
      a = arcIter.data();
      sp.first.emplace_back((uint)a->source);
      sp.second += a->weight;
    }
    sp.first.emplace_back((uint)a->dest);
    shortest_paths.emplace_back(sp);
  }
}
