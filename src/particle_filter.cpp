/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles=150;
	
	// Noise
	default_random_engine generator;
	normal_distribution<double> noise_x(0, std[0]);
	normal_distribution<double> noise_y(0, std[1]);
	normal_distribution<double> noise_theta(0, std[2]);
	
	for (int i = 0; i < num_particles; i++){
		Particle part;
		part.id = i;
		part.x = x;
		part.y = y;
		part.theta = theta;
		part.weight = 1;
		
		part.x += noise_x(generator);
		part.y += noise_y(generator);
		part.theta += noise_theta(generator);
		
		particles.push_back(part);
	}
	
	is_initialized = true;	
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine generator;
	normal_distribution<double> noise_x(0, std_pos[0]);
	normal_distribution<double> noise_y(0, std_pos[1]);
	normal_distribution<double> noise_theta(0, std_pos[2]);
	
	for (int i = 0; i < num_particles; i++){
		double dt_v = delta_t * velocity;
		double v_yawrate = velocity / yaw_rate;
		double cos_theta = cos(particles[i].theta);
		double sin_theta = sin(particles[i].theta);
		double theta_yaw_dt = particles[i].theta + yaw_rate * delta_t;
		
		// motion model if yaw rate is zero
		if (fabs(yaw_rate) < 0.0001){ 
			particles[i].x += dt_v * cos_theta;
			particles[i].y += dt_v * sin_theta;
		}
		else {
			particles[i].x += v_yawrate * (sin(theta_yaw_dt) - sin_theta);
			particles[i].y += v_yawrate * (cos_theta - cos(theta_yaw_dt));
			particles[i].theta += yaw_rate * delta_t;
		}
		
		// Noise
		particles[i].x += noise_x(generator);
		particles[i].y += noise_y(generator);
		particles[i].theta += noise_theta(generator);	 	  
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i = 0; i < observations.size(); i++){
		LandmarkObs current_obs = observations[i];
		double min_dist = 1e300; //large distance
		int id_tracker = 0; //tracker for closest distance
		
		for (int j = 0; j < predicted.size(); j++){
			LandmarkObs current_pred = predicted[j];
			double current_dist = dist(current_obs.x, current_obs.y, current_pred.x, current_pred.y);
			
			if (current_dist < min_dist){
				id_tracker = current_pred.id;
				min_dist = current_dist;
			}
		}
		observations[i].id = id_tracker;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	for (int i = 0; i < num_particles; i++){
		// vector to contain predicted landmark positions
		vector<LandmarkObs> preds;
		// vector to contain transformed observations
		vector<LandmarkObs> trans;
		
		// current particle
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta;
		
		// transform observations to map coordinates
		for (int j = 0; j < observations.size(); j++){
			int id_obs = observations[j].x;
			double x_obs = observations[j].x;
			double y_obs = observations[j].y;					
			double cos_theta = cos(theta);
			double sin_theta = sin(theta);
			
			double x_trans = x + cos_theta * x_obs - sin_theta * y_obs;
			double y_trans = y + sin_theta * x_obs + cos_theta * y_obs;
			
			trans.push_back(LandmarkObs{id_obs, x_trans, y_trans});
		}
		
		// look for landmarks in sensor range
		for (int k = 0; k < map_landmarks.landmark_list.size(); k++){
			int land_id = map_landmarks.landmark_list[k].id_i;
			double land_x = map_landmarks.landmark_list[k].x_f;
			double land_y = map_landmarks.landmark_list[k].y_f;
			
			double current_dist = dist(x, y, land_x, land_y);
			
			// add landmark feature to predicted vector if within range
			if (current_dist < sensor_range){
				preds.push_back(LandmarkObs{land_id, land_x, land_y});
			}
		}
		
		dataAssociation(preds, trans);
		particles[i].weight = 1;
		
		// cycle through each transformed observation and update weight
		for (int l = 0; l < trans.size(); l++){
			int id = trans[l].id;
			double x_obs = trans[l].x;
			double y_obs = trans[l].y;
			double x_pred, y_pred;
			
			for (int m = 0; m < preds.size(); m++){
				if (preds[m].id == id){
					x_pred = preds[m].x;
					y_pred = preds[m].y;
				}
			}
			
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			
			double weight = 1/(2*M_PI*std_x*std_y)*exp(-(pow(x_obs-x_pred,2)/(2*std_x*std_x)+pow(y_obs-y_pred,2)/(2*std_y*std_y)));
			
			particles[i].weight *= weight;
		}
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// vector of new particles
	vector<Particle> new_parts;
	
	// vector of weights
	vector<double> weights;
	for(int i = 0; i < num_particles; i++){
		weights.push_back(particles[i].weight);
	}
	
	// starting index
	default_random_engine generator;
	uniform_int_distribution<int> uni_int(0, num_particles-1);
	
	int index = uni_int(generator);
	double beta = 0;
	double mw = *max_element(weights.begin(), weights.end());
	uniform_real_distribution<double> uni_real(0.0, mw);
	
	for (int i = 0; i < num_particles; i++){
		beta += uni_real(generator) * 2 * mw;
		while (beta > weights[index]){
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
	new_parts.push_back(particles[index]);
	}
	
	particles = new_parts;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
