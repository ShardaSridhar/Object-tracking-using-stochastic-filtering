# Object-tracking-using-stochastic-filtering
Object tracking using Unscented Kalman filter and Particle filter. 
The basic principle of stochastic filtering- Predict(using system update equation) and update using current observation.
UKF assumes a Gaussian prior but Particle filter assumes a non-Gaussian prior. 
The code Unscented_Kalman_Filter.m and pf.m correspond to the filters. likelihood.m and plotstep.m are functions that are needed for the main code.  
