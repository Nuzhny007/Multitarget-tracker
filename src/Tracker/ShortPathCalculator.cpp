#include "ShortPathCalculator.h"

#include <GTL/GTL.h>
#include "mygraph.h"
#include "mwbmatching.h"
#include "tokenise.h"

#include "muSSP/Graph.h"

///
/// \brief SPBipart::Solve
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void SPBipart::Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t maxCost)
{
    MyGraph G;
    G.make_directed();

    std::vector<node> nodes(N + M);

    for (size_t i = 0; i < nodes.size(); ++i)
    {
        nodes[i] = G.new_node();
    }

    edge_map<int> weights(G, 100);
    for (size_t i = 0; i < N; i++)
    {
        bool hasZeroEdge = false;

        for (size_t j = 0; j < M; j++)
        {
            track_t currCost = costMatrix[i + j * N];

            edge e = G.new_edge(nodes[i], nodes[N + j]);

            if (currCost < m_settings.m_distThres)
            {
                int weight = static_cast<int>(maxCost - currCost + 1);
                G.set_edge_weight(e, weight);
                weights[e] = weight;
            }
            else
            {
                if (!hasZeroEdge)
                {
                    G.set_edge_weight(e, 0);
                    weights[e] = 0;
                }
                hasZeroEdge = true;
            }
        }
    }

    edges_t L = MAX_WEIGHT_BIPARTITE_MATCHING(G, weights);
    for (edges_t::iterator it = L.begin(); it != L.end(); ++it)
    {
        node a = it->source();
        node b = it->target();
        assignment[b.id()] = static_cast<assignments_t::value_type>(a.id() - N);
    }
}

///
/// \brief FindSP
/// \param orgGraph
///
void FindSP(Graph& orgGraph, std::vector<track_t>& pathCost, std::vector<std::vector<int>>& pathSet)
{
	std::cout << "1: remove dummy edges" << std::endl;
    orgGraph.invalid_edge_rm();

    int path_num = 0;

	std::cout << "2: initialize shortest path tree from the DAG" << std::endl;
    orgGraph.shortest_path_dag();

	if (orgGraph.distance2src.size() <= orgGraph.sink_id_)
	{
		std::cout << "Error: orgGraph.distance2src.size() <= orgGraph.sink_id_" << std::endl;
	}
	else
	{
		pathCost.push_back(orgGraph.distance2src[orgGraph.sink_id_]);
		orgGraph.cur_path_max_cost = -orgGraph.distance2src[orgGraph.sink_id_]; // the largest cost we can accept
	}

	std::cout << "3: convert edge cost (make all weights positive)" << std::endl;
    orgGraph.update_allgraph_weights();

	std::cout << "8: extract shortest path" << std::endl;
    orgGraph.extract_shortest_path();

    pathSet.push_back(orgGraph.shortest_path);
    path_num++;

    std::vector<size_t> update_node_num;

	std::cout << "4: find nodes for updating based on branch node" << std::endl;
    std::vector<int> node_id4updating;
    orgGraph.find_node_set4update(node_id4updating);

	std::cout << "10: rebuild residual graph by flipping paths" << std::endl;
    orgGraph.flip_path(); // also erase the top sinker
    for (;;)
    {
		std::cout << "6: update shortest path tree based on the selected sub-graph" << std::endl;
        orgGraph.update_shortest_path_tree_recursive(node_id4updating);
		std::cout << "Iteration " << path_num << ", updated node number " << orgGraph.upt_node_num << std::endl;

		std::cout << "7: update sink node (heap)" << std::endl;
        orgGraph.update_sink_info(node_id4updating);
        update_node_num.push_back(node_id4updating.size());

		std::cout << "8: extract shortest path" << std::endl;
        orgGraph.extract_shortest_path();

		std::cout << "test if stop" << std::endl;
		track_t cur_path_cost = pathCost[path_num - 1] + orgGraph.distance2src[orgGraph.sink_id_];
        if (cur_path_cost > -0.0000001f)
            break;

        pathCost.push_back(cur_path_cost);
        orgGraph.cur_path_max_cost = -cur_path_cost;
        pathSet.push_back(orgGraph.shortest_path);
        path_num++;

		std::cout << "9: update weights" << std::endl;
        orgGraph.update_subgraph_weights(node_id4updating);

		std::cout << "4: find nodes for updating" << std::endl;
        orgGraph.find_node_set4update(node_id4updating);

		std::cout << "10: rebuild the graph" << std::endl;
        orgGraph.flip_path();
    }
}

///
/// \brief SPmuSSP::Solve
/// \param costMatrix
/// \param N
/// \param M
/// \param assignment
/// \param maxCost
///
void SPmuSSP::Solve(const distMatrix_t& costMatrix, size_t N, size_t M, assignments_t& assignment, track_t /*maxCost*/)
{
	std::cout << "SPmuSSP::Solve: tracks = " << N << ", M = " << M << std::endl;
	std::cout << "m_detects init: size = " << m_detects.size() << ", config:" << std::endl;
	for (size_t i = 0; i < m_detects.size(); ++i)
	{
		std::cout << "layer[" << i << "]: nodes = " << m_detects[i].Size() << ", arcs = " << m_detects[i].m_arcsCount << std::endl;
	}

    // Add new "layer" to the graph
    if (m_detects.size() < 2)
    {
        m_detects.resize(2);
        m_detects[0].Resize(N);
        m_detects[1].Resize(M);
    }
    else
    {
        assert(m_detects.back().Size() == N);
        m_detects.push_back(Layer());
        m_detects.back().Resize(M);

        if (m_detects.size() > m_settings.m_maxHistory)
            m_detects.pop_front();
    }

    auto layer = m_detects.rbegin() + 1;
    for (size_t i = 0; i < N; ++i)
    {
        Node& node = (*layer)[i];

        for (size_t j = 0; j < M; ++j)
        {
            track_t currCost = costMatrix[i + j * N];
            if (currCost < m_settings.m_distThres)
            {
                node.Add(j, currCost);
                layer->m_arcsCount++;
            }
        }
    }

	std::cout << "m_detects updated: size = " << m_detects.size() << ", config:" << std::endl;
	for (size_t i = 0; i < m_detects.size(); ++i)
	{
		std::cout << "layer[" << i << "]: nodes = " << m_detects[i].Size() << ", arcs = " << m_detects[i].m_arcsCount << std::endl;
	}

    // Calc number of nodes and arcs
    size_t nNodes = 0; // no of nodes
	size_t nArcs = 0;  // no of arcs
    for (const auto& llayer : m_detects)
    {
        nNodes += llayer.Size();
        nArcs += llayer.m_arcsCount;
    }

	std::cout << "Create Graph: nodes " << nNodes << ", arcs " << nArcs << std::endl;
    Graph orgGraph(static_cast<int>(nNodes), static_cast<int>(nArcs), 0, static_cast<int>(nNodes) - 1, 0, 0);
    int edgeID = 0;
    size_t edgesSum = 0;
    size_t nodesSum = 0;
    for (const auto& llayer : m_detects)
    {
        for (size_t j = 0; j < llayer.m_nodes.size(); ++j)
        {
            const auto& node = llayer.m_nodes[j];
            for (size_t i = 0; i < node.m_arcs.size(); ++i)
            {
                const auto& arc = node.m_arcs[i];
                size_t tail = nodesSum + j;
				size_t head = nodesSum + llayer.m_nodes.size() + arc.first;
                orgGraph.add_edge(static_cast<int>(tail), static_cast<int>(head), edgeID, arc.second);
                ++edgeID;
            }
        }
        edgesSum += llayer.m_arcsCount;
        nodesSum += llayer.m_nodes.size();
    }

	std::cout << "Find paths" << std::endl;
	std::vector<track_t> pathCost;
    std::vector<std::vector<int>> pathSet;
    FindSP(orgGraph, pathCost, pathSet);

    // track_t costSum = 0;
    // for (auto &&i : pathCost)
    // {
    //    costSum += i;
    // }
    // printf("The number of paths: %ld, total cost is %.7f, final path cost is: %.7f.\n", path_cost.size(), cost_sum, path_cost[path_cost.size() - 1]);
    // print_solution(org_graph.get(), path_set, "output.txt");//"output_edge_rm.txt"

    auto GetRowIndFromID = [&](int id)
    {
        int res = -1;
        size_t nodesSum = 0;
        for (const auto& layer : m_detects)
        {
            if (nodesSum + layer.m_nodes.size() > static_cast<size_t>(id))
            {
                res = id - static_cast<int>(nodesSum);
            }
            nodesSum += layer.m_nodes.size();
        }
        return res;
    };
    auto GetRegionIndFromID = [&](int id)
    {
        int res = -1;
        size_t nodesSum = 0;
        for (size_t i = 0; i < m_detects.size(); ++i)
        {
            const auto& layer = m_detects[i];
            if (nodesSum + layer.m_nodes.size() > static_cast<size_t>(id))
            {
                if (i + 1 == m_detects.size())
                    res = id - static_cast<int>(nodesSum);
                break;
            }
            nodesSum += layer.m_nodes.size();
        }
        return res;
    };

	std::cout << "Get result" << std::endl;
    for (size_t i = 0; i < pathSet.size(); ++i)
    {
        const auto& path = pathSet[i];

        if (path.size() > 1)
        {
            std::map<int, size_t> freq;

            for (size_t j = 0; j < path.size(); ++j)
            {
                int row = GetRowIndFromID(path[j]);
                assert(row >= 0);
                freq[row]++;
            }
            int maxRow = -1;
            size_t maxvals = 0;
            for (auto it : freq)
            {
                if (maxvals < it.second)
                {
                    maxvals = it.second;
                    maxRow = it.first;
                }
            }
            assert(maxRow >= 0);
            assignment[maxRow] = GetRegionIndFromID(static_cast<int>(path.size()) - 1);
        }
    }
}

///
/// \brief SPmuSSP::UpdateDetects
/// \param deletedTracks
/// \param newTracks
/// \param reg2tracks
///
void SPmuSSP::UpdateDetects(const std::vector<size_t>& deletedTracks, size_t newTracks, const std::vector<size_t>& reg2track)
{
	std::cout << "SPmuSSP::UpdateDetects: deletedTracks = " << deletedTracks.size() << ", newTracks = " << newTracks << ", reg2track = " << reg2track.size() << std::endl;
	std::cout << "m_detects before: size = " << m_detects.size() << ", config:" << std::endl;
	for (size_t i = 0; i < m_detects.size(); ++i)
	{
		std::cout << "layer[" << i << "]: nodes = " << m_detects[i].Size() << ", arcs = " << m_detects[i].m_arcsCount << std::endl;
	}

	if (m_detects.empty())
		return;

	m_detects.pop_back();
	auto currLayer = m_detects.rbegin();

	if (m_detects.size() > 1)
	{
		auto prevLayer = m_detects.rbegin() + 1;
		for (auto ind : deletedTracks)
		{
			// Delete all arcs to this node from the previous layer
			for (size_t i = 0; i < prevLayer->m_nodes.size(); ++i)
			{
				Node& node = (*prevLayer)[i];

				for (auto it = std::begin(node.m_arcs); it != std::end(node.m_arcs);)
				{
					if (it->first == ind)
					{
						it = node.m_arcs.erase(it);
						prevLayer->m_arcsCount--;
					}
					else
					{
						++it;
					}
				}
			}
			// Delete node
			currLayer->m_nodes.erase(std::begin(currLayer->m_nodes) + ind);
		}
	}

	size_t newSize = currLayer->Size() + newTracks;
	m_detects.push_back(Layer());
	m_detects.back().Resize(newSize);

	auto layer = m_detects.rbegin() + 1;
	for (size_t i = 0; i < layer->Size(); ++i)
	{
		Node& node = (*layer)[i];

		for (auto& arc : node.m_arcs)
		{
			arc.first = reg2track[arc.first];
		}
	}

	std::cout << "m_detects after: size = " << m_detects.size() << ", config:" << std::endl;
	for (size_t i = 0; i < m_detects.size(); ++i)
	{
		std::cout << "layer[" << i << "]: nodes = " << m_detects[i].Size() << ", arcs = " << m_detects[i].m_arcsCount << std::endl;
	}
}
