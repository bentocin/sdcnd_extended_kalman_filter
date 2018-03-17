#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */

  // Measurement model for the laser
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  // Measurement model for the radar (non-linear --> Jacobian)
  Hj_ << 0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0, 0;

    // Initialize covariance matrix
    ekf_.P_ = MatrixXd(4,4);
    ekf_.P_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1000, 0,
              0, 0, 0, 1000;
    
    // Setup start for state transmission matrix
    ekf_.F_ = MatrixXd(4,4);
    ekf_.F_ << 1, 0, 0, 0,
              0, 1, 0, 0,
              0, 0, 1, 0,
              0, 0, 0, 1;

  // Initialize noise of the process (9.0 mentioned in task description)
  noise_ax = 9.0;
  noise_ay = 9.0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {


  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // First measurement
    ekf_.x_ = VectorXd(4);

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // Convert radar from polar to cartesian coordinates and initialize state
      double x = cos(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[0];
      double y = sin(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[0];
      // double v_x = cos(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[2];
      // double v_y = sin(measurement_pack.raw_measurements_[1]) * measurement_pack.raw_measurements_[2];
      ekf_.x_ << x, y, 0, 0;
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // Initialize state
      ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }

    if (ekf_.x_[0] < 0.0001)
    {
      ekf_.x_[0] = 0.0001;
    }

    if (ekf_.x_[1] < 0.0001)
    {
      ekf_.x_[1] = 0.0001;
    }

    // Save timestamp
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // Time elapsed between the current and previous measurements in seconds
  float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;

  // Update the reference timestamp
  previous_timestamp_ = measurement_pack.timestamp_;

  // Set time dependent elements of state transmission matrix
  ekf_.F_(0,2) = dt;
  ekf_.F_(1, 3) = dt;

  ekf_.Q_ = MatrixXd(4,4);
  ekf_.Q_ << pow(dt, 4)/4.0*noise_ax, 0, pow(dt, 3)/2.0*noise_ax, 0,
            0, pow(dt, 4)/4.0*noise_ay, 0, pow(dt, 3)/2.0*noise_ay,
            pow(dt, 3)/2.0*noise_ax, 0, dt*dt*noise_ax, 0,
            0, pow(dt, 3)/2.0*noise_ay, 0, dt*dt*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  /**
   TODO:
     * Use the sensor type to perform the update step.
     * Update the state and covariance matrices.
   */

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
    ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
    ekf_.R_ = R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
    ekf_.H_ = H_laser_;
    ekf_.R_ = R_laser_;
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
