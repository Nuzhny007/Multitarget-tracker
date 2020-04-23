#pragma once

#include <vector>
#include "Node.h"
#include <algorithm>       /* fabs */
#include <map>
#include <limits>
#include "defines.h"

constexpr track_t FINF = std::numeric_limits<track_t>::max();
constexpr track_t FINFHALF = FINF / 2.0f;

///
/// \brief The Sink class
///
class Sink
{
public:
    // use set to save sink's precursors' distances
    std::multimap<track_t, int> sink_precursors;
    std::vector<track_t> sink_precursor_weights;
    track_t sink_cost_ = 0; // this can be a vector, in our framework, it it a scaler
    track_t sink_weight_shift = 0;

    Sink() = default;
    Sink(int n, track_t sink_cost);

    void sink_update_all(std::vector<Node> &V, std::vector<track_t> &distance2src, int sink_id, int n);

    void sink_update_all_weight(std::vector<Node> &V, std::vector<track_t> &distance2src, int sink_id, int n);


    void sink_build_precursormap(std::vector<track_t> &ancestor_ssd, std::vector<int> &ancestor_node_id, std::vector<int> &parent_node_id, int n);


    void sink_update_all_half(std::vector<track_t> distance2src, int sink_id, int n);
    void sink_update_subgraph(std::vector<int> update_node_id, std::vector<track_t> distance2src, int sink_id, int n);
};
