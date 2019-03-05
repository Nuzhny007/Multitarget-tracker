/*
 * simulate.hpp
 *
 *  Created on: May 15, 2016
 *      Author: uzleo
 */

#ifndef SIMULATE_HPP_
#define SIMULATE_HPP_

#include "global.hpp"
#include <armadillo>

#ifdef DEBUG
#define NOISELESS
#endif

// this module generates a simulation of the 2D scene

// the objects move with constant velocity

// this module generates the measurements that would have been recevied from the
// sensor
// based on the object motion and clutter in the scene

class Simulate {
  // sub-structures/classes
private:
  // to represent a specific object in the scene
  class simulateTarget {
    // sub-structures/classes
  public:
    enum state { pre_birth, alive, dead };
    // member variables
  private:
    // object's varying position
    arma::fvec m_position;
    // object's constant velocity
    arma::fvec m_velocity;
    float m_birth_time;
    float m_death_time;

    Simulate *m_owner;

  public:
    state m_state;
    // member functions
  public:
    simulateTarget(Simulate *owner, arma::fvec pos, arma::fvec vel,
                   std::pair<float, float> life_span);
    void updatePosition();
    void printPosition();
    arma::fvec getPosition();
    void updateState();
  };

  // member variables
private:
  // dynamic number of objects in the scene
  std::vector<simulateTarget> m_targets;
  // measurements container upto current time
  std::vector<arma::fvec> m_measurements;
  std::vector<arma::fvec> m_true_pos;

  float scene_xmax, scene_xmin, scene_ymax, scene_ymin;
  float noise_strength;
  float detection_prob;
  int clutter_per_scan;
  bool remove_dead_targets;

public:
  float m_current_time;
  float m_delta_t;
  // member functions
public:
  Simulate();
  void Initialize();
  void runSimulation();
  std::vector<arma::fvec> getTrueObservations();
  std::vector<arma::fvec> getSensorMeasurements();
};

#endif /* SIMULATE_HPP_ */

// TODO: documentation
