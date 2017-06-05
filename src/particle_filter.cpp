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
    default_random_engine gen;
    normal_distribution<double> nd_x(x, std[0]);
    normal_distribution<double> nd_y(y, std[1]);
    normal_distribution<double> nd_theta(theta, std[2]);

    num_particles = 20;
    weights = vector<double>(num_particles);
    particles = vector<Particle>(num_particles);
    
    for (int i=0; i < num_particles; i++) {
        particles[i].id = i;
        particles[i].x = nd_x(gen);
        particles[i].y = nd_y(gen);
        particles[i].theta = nd_theta(gen);
        particles[i].weight = 1.0;
        weights[i] = 1.0;
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    // Predict new particle state after delta_t
    default_random_engine gen;
    for (int i=0; i < num_particles; i++) {
        
        // Use motion model to predict the next state of the particles after delta_t
        double theta = particles[i].theta;
        if (fabs(yaw_rate) >= 0.001) {
            particles[i].x += velocity/yaw_rate * (sin(theta + yaw_rate * delta_t)-sin(theta));
            particles[i].y += velocity/yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
            particles[i].theta += yaw_rate*delta_t;
        } else {
            particles[i].x += velocity * delta_t * cos(theta);
            particles[i].y += velocity * delta_t * sin(theta);
            
        }
        
        // Add gaussian noise to the predicted particle state
        std:: normal_distribution<double> ndx(particles[i].x, std_pos[0]);
        std:: normal_distribution<double> ndy(particles[i].y, std_pos[1]);
        std:: normal_distribution<double> ndt(particles[i].theta, std_pos[2]);
        particles[i].x = ndx(gen);	
        particles[i].y = ndy(gen);	
        particles[i].theta = ndt(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Iterate over all observations and assign the closest observation to the predicted landmark (not sure why it is called predicted)
    for (auto& obs : observations) {
        double min_distance = 10000000.0;
        for (auto pred : predicted) {
            auto dist_val = dist(obs.x, obs.y, pred.x,pred.y);
            if (dist_val < min_distance) {
                obs.id = pred.id;
                min_distance = dist_val;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
    // Update particle filter weights
    // Iterate over all particles
    for (int i=0; i < num_particles; i++) {
        // Transform car coordinates into map coordinates and store them in a vector
        std::vector<LandmarkObs> transf_observations = std::vector<LandmarkObs>();
        for (const auto obs : observations) {
            double transf_obs_x = cos(particles[i].theta) * obs.x - sin(particles[i].theta) * obs.y + particles[i].x;
            double transf_obs_y = sin(particles[i].theta) * obs.x + cos(particles[i].theta) * obs.y + particles[i].y;
            transf_observations.push_back(LandmarkObs{obs.id, transf_obs_x, transf_obs_y});
        }
        
        // Filter the landmarks available to each particle using sensor_range in order to enhance algorithm speed
        std::vector<LandmarkObs> particle_landmarks;
        for (const auto& m_landmark : map_landmarks.landmark_list){
            double distance = dist(m_landmark.x_f, m_landmark.y_f, particles[i].x, particles[i].y);
            // If the distance between the landmark and the particle is less than sensor_range, then store landmark in  vector (particle_landmarks) and store the pair <id,map landmark> in a dictionary for later use
            if (distance <= sensor_range){
                particle_landmarks.push_back(LandmarkObs{m_landmark.id_i,m_landmark.x_f,m_landmark.y_f});
            }
        }
        
        // If nearby landmarks exist
        if (particle_landmarks.size() > 0) {
            // Associate the particle_landmarks to the transformed sensor observations
            dataAssociation(particle_landmarks, transf_observations);
            particles[i].weight = 1;
            for (const auto transf_obs : transf_observations) {
                // Retrieve the landmark with the minimum distance to the transformed sensor observation w.r.t to ith particle
                std::vector<LandmarkObs>::iterator it = std::find_if(particle_landmarks.begin(), particle_landmarks.end(), [&](const LandmarkObs& l){
                    return l.id==transf_obs.id;
                });
                
                double mmt_x = it->x;
                double mmt_y = it->y;
                
                double predicted_mmt_x = transf_obs.x;
                double predicted_mmt_y = transf_obs.y;
                
                // Update wieghts using a multivariate gaussian distribution without correlation between x and y
                double x_delta = pow(predicted_mmt_x-mmt_x,2)/(2*pow(std_landmark[0],2));
                double y_delta = pow(predicted_mmt_y-mmt_y,2)/(2*pow(std_landmark[1],2));
                
                // Multiple individual space (x,y) weights to get the new particle weight
                particles[i].weight *= 1/(2*M_PI*std_landmark[0]*std_landmark[1]) * exp(-(x_delta + y_delta));
            }
            
            weights[i] = particles[i].weight;
            
        } else {
            // If no nearby landmarks were found for that particle, then set weight to 0.
            weights[i] = 0.0;
        }
    }
}

void ParticleFilter::resample() {
    // Resample particles using a discrete distribution that returns an integer [0,n) with a probability proportional to the size of the index weight
    default_random_engine gen;
    vector<Particle> new_particles =  std::vector<Particle>(num_particles);
    std::discrete_distribution<> d(weights.begin(), weights.end());
    for(int i=0; i<num_particles; ++i) {
        new_particles[i]=particles[d(gen)];
    }
    particles=new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
