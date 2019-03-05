/*
 * simulate.cpp
 *
 *  Created on: May 14, 2016
 *      Author: uzleo
 */

#include <cstdio>
#include <assert.h>
#include <time.h>

#include "simulate.hpp"

Simulate::Simulate() { Initialize(); }

// constructor to explicitly set object's initial position and velocity
Simulate::simulateTarget::simulateTarget(Simulate *owner, arma::fvec position,
                                         arma::fvec velocity,
                                         std::pair<float, float> life_span) {
  m_state = state::pre_birth;
  m_position = position;
  m_velocity = velocity;
  m_birth_time = life_span.first;
  m_death_time = life_span.second;
  m_owner = owner;
}

// print current object position
void Simulate::simulateTarget::printPosition() { m_position.print("position"); }

// update object position based on its constant velocity motion
void Simulate::simulateTarget::updatePosition() {
  assert(m_state == alive);
  m_position(0) += (m_velocity(0) * m_owner->m_delta_t);
  m_position(1) += (m_velocity(1) * m_owner->m_delta_t);
}

//
void Simulate::simulateTarget::updateState() {
  if (m_owner->m_current_time > m_death_time) {
    m_state = dead;
  } else if (m_owner->m_current_time >= m_birth_time) {
    m_state = alive;
  }
}

// get the current position of the object
arma::fvec Simulate::simulateTarget::getPosition() {
  assert(m_state == alive);
  return m_position;
}

// initialize the simulation scenario at time-step 0
void Simulate::Initialize() {
#ifndef DEBUG
  srand(time(NULL));
#else
  srand(1);
#endif

  m_delta_t = 1.f;
  scene_xmin = -1000.f;
  scene_xmax = 1000.f;
  scene_ymin = -1000.f;
  scene_ymax = 1000.f;
  noise_strength = 1.4f;
  detection_prob = 0.9f;
  clutter_per_scan = 5;
  m_current_time = 0.f;
  remove_dead_targets = false;

  // creating targets
  m_targets.push_back(
      simulateTarget(this, arma::fvec({0.f, 0.f}), arma::fvec({10.f, 10.f}),
                     std::make_pair(10 * m_delta_t, 70 * m_delta_t)));

  m_targets.push_back(
      simulateTarget(this, arma::fvec({-330.f, 220.f}), arma::fvec({2.f, -6.f}),
                     std::make_pair(30 * m_delta_t, 90 * m_delta_t)));
}

//
void Simulate::runSimulation() {
  m_measurements.clear();
  m_true_pos.clear();
  m_current_time += m_delta_t;

  // update each of object's position
  for (auto &tar : m_targets) {
    tar.updateState();
    if (tar.m_state == simulateTarget::dead) {
      remove_dead_targets = true;
    } else if (tar.m_state == simulateTarget::alive) {
      tar.updatePosition();
      // get measurement from this object
      arma::fvec pos = tar.getPosition();
      // only getting the measurement if the target is within the scene
      if ((pos(0) <= scene_xmax && pos(0) >= scene_xmin) &&
          (pos(1) <= scene_ymax && pos(1) >= scene_ymin)) {
        m_true_pos.push_back(pos);
#ifdef NOISELESS
        m_measurements.push_back(pos);
#else
        if (((float)rand() / (float)RAND_MAX) < detection_prob) {
          // adding white gaussian noise with 0 mean and unit variance
          pos += noise_strength * arma::randn<arma::fvec>(pos.size());
          m_measurements.push_back(pos);
        }
#endif
      } else {
        // target is out of the scene so killing it
        tar.m_state = simulateTarget::dead;
        remove_dead_targets = true;
      }
    }
  }

  // adding clutter
  for (int cl = 0; cl < clutter_per_scan; cl++) {
    arma::fvec tmp(2);
    tmp(0) = (float)(scene_xmin + rand() % (int)(scene_xmax - scene_xmin));
    tmp(1) = (float)(scene_ymin + rand() % (int)(scene_ymax - scene_ymin));
    m_measurements.push_back(tmp);
  }

  // removing dead targets if they are present
  if (remove_dead_targets) {
    remove_dead_targets = false;
    m_targets.erase(std::remove_if(m_targets.begin(), m_targets.end(),
                                   [](simulateTarget &tar) {
                      return (tar.m_state == simulateTarget::dead);
                    }),
                    m_targets.end());
  }
}

//
std::vector<arma::fvec> Simulate::getTrueObservations() { return m_true_pos; }

//
std::vector<arma::fvec> Simulate::getSensorMeasurements() {
  return m_measurements;
}

// TODO: documentation
