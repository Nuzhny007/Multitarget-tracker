#pragma once

#include "Node.h"
#include "Sink.h"
#include <stack>
#include <vector>
#include <queue>
#include <limits.h>
#include <unordered_map>
#include <iostream>
#include <algorithm>    // std::find
#include <cstring>
#include <fstream>
#include <memory>
#include "defines.h"

///
/// \brief The Graph class
///
class Graph
{
public:
        int num_nodes_ = 0;
        int num_edges_ = 0;
        int src_id_ = 0;
        int sink_id_ = 0;
        track_t en_weight_ = 0;
        track_t ex_weight_ = 0;

        std::vector<Node> V_; // save all nodes in the graph (precursor/successor/edge idx)
        std::vector<track_t> edge_weights;
        static inline size_t node_key(int i,int j)
        {
            return (size_t) i << 32 | (unsigned int) j;
        }
        std::unordered_map<size_t, int> node_id2edge_id;
        std::vector<int> shortest_path;
        std::unique_ptr<Sink> sink_info;
        track_t precursor_queue_top_val = 0;
        // for data validation
        std::vector<std::pair<int, int>> edge_tail_head;
        std::vector<track_t> edge_org_weights;
		size_t upt_node_num = 0;
        //
        std::vector<bool> node_visited, edge_visited;
        std::vector<int> node_in_visited;
        std::vector<int> parent_node_id;
        std::vector<int> ancestor_node_id;
        std::vector<track_t> distance2src;
        std::vector<track_t> ancestor_ssd; //shortest distance for a single ancestor
        std::vector<std::vector<int>> ancestors_descendants; // for each ancestor, its following nodes
        std::multimap<track_t, int> node_upt_waitinglist;
        std::stack<int> edge_upt_waitinglist;
        track_t node_upt_shift = 0;
        track_t cur_upt_shift = 0;
        std::vector<track_t> nodewise_upt_shift;
        std::queue<int> tplog_queue, tmp_queue;
        std::vector<int> tplog_vec;

        track_t cur_path_max_cost = 0;
        track_t cur_remain_max_distance = 0;

        Graph() = default;
        Graph(int num_nodes, int num_edges, int src_id, int sink_id, track_t en_weight, track_t ex_weight);
        ~Graph();

        Node &get_node(int pos);

        void add_edge(int tail_id, int head_id, int edge_id, track_t weight);

        void invalid_edge_rm();
        void shortest_path_dag();
        // A function used by shortest_path_dag
        void topologicalSortUtil(int v, std::vector<bool>& visited, std::stack<int> &Stack);

        void extract_shortest_path();
        // flip shortest path
        void flip_path();

        // update edge weights
        void update_allgraph_weights();

        /**All nodes**/
        void find_node_set4update(std::vector<int> &update_node_id);
        void update_subgraph_weights(std::vector<int> &update_node_id);

        void update_shortest_path_tree_recursive(std::vector<int> &update_node_id);
        void recursive_update_successors_distance(int curr_node_id, track_t curr_dist, int curr_ancestor, std::vector<int> &update_node_id4edges);
        void topologicalSort_counter_order(int v);
        void update_sink_info(std::vector<int> update_node_id);
};
