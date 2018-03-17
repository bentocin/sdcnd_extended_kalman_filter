#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  // Initialize the root mean squared error vector
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // Check that prerequisites are fullfilled
  if (estimations.size() != ground_truth.size() || estimations.size() == 0)
  {
    cout << "Invalid estimation or ground truth data" << endl;
    return rmse;
  }

  // Loop over all the data points and sum the squared errors
  for (int i = 0; i < estimations.size(); i++)
  {
    VectorXd residual = estimations[i] - ground_truth[i];

    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Take the average and calculate the root
  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  cout << "RMSE = " << rmse << endl;
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

  // Initialize Jacobian for radar measurements and 4 dimensional state
  MatrixXd H_j(3,4);

  double p_x = x_state(0);
  double p_y = x_state(1);
  double v_x = x_state(2);
  double v_y = x_state(3);
  double c1 = p_x * p_x + p_y * p_y;
  double c2 = sqrt(c1);

  if (fabs(c1) < 0.0001)
  {
    c1 = 0.0001;
  }

  H_j <<  p_x/(c2), p_y/(c2), 0, 0,
    -p_y/(c1), p_x/(c1), 0, 0,
    (p_y * (v_x * p_y - v_y * p_x))/pow(c1, 1.5), (p_x * (v_y * p_x - v_x * p_y))/pow(c1, 1.5), p_x/c2, p_y/c2;

  return H_j;
}
