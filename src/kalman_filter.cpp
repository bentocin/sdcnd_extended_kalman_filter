#include "kalman_filter.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {

  // Linear prediction model is the same for Laser and Radar
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {

  // Linear update model
  // Calculate the difference between sensor measurement and prediction
  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  // Kalman filter gain
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  x_ = x_ + K * y;

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {

  // Non-linear update model
  // Convert cartesian to polar
  VectorXd h_x = VectorXd(3);
  h_x(0) = sqrt(x_(0) * x_(0) + x_(1) * x_(1));
  h_x(1) = atan2(x_(1), x_(0));
  h_x(2) = (x_(0) * x_(2) + x_(1) * x_(3)) / h_x(0);
  
  // Calculate the difference between sensor measurement and prediction
  VectorXd y = z - h_x;

  // Normalize the angle
  while (y(1) > M_PI || y(1) < - M_PI)
  {
    if (y(1) > M_PI) 
    {
      y(1) -= M_PI;
    } else {
      y(1) += M_PI;
    }
  }

  MatrixXd S = H_ * P_ * H_.transpose() + R_;
  // Kalman filter gain
  MatrixXd K = P_ * H_.transpose() * S.inverse();
  x_ = x_ + K * y;

  MatrixXd I = MatrixXd::Identity(x_.size(), x_.size());
  P_ = (I - K * H_) * P_;
}
