#ifndef SOFTDTREE_ADAM_HPP_
#define SOFTDTREE_ADAM_HPP_

#include <cmath>
#include <cstdint>

#include <Eigen/Dense>

class Adam {
public:
  Adam(const double eta = 0.001, const double beta1 = 0.9, const double beta2 = 0.999, const double epsilon = 1e-16):
    eta_(eta), beta1_(beta1), beta2_(beta2), epsilon_(epsilon), iter_(0), moment1_scl_(0), moment2_scl_(0) { }

  void update(Eigen::VectorXd& weight, const Eigen::VectorXd& gradient) {
    if (iter_ == 0) {
      const uint32_t n_elements = weight.size();
      moment1_vec_ = Eigen::VectorXd::Zero(n_elements);
      moment2_vec_ = Eigen::VectorXd::Zero(n_elements);
    }
    iter_ += 1;
    moment1_vec_ *= beta1_;
    moment1_vec_ += (1.0 - beta1_) * gradient;
    moment2_vec_ *= beta2_;
    moment2_vec_ += ((1.0 - beta2_) * gradient.array().square()).matrix();
    const Eigen::VectorXd normalized_moment1 = moment1_vec_ / (1.0 - std::pow(beta1_, iter_));
    const Eigen::VectorXd normalized_moment2 = moment2_vec_ / (1.0 - std::pow(beta2_, iter_));
    weight -= eta_ * (normalized_moment1.array() / (normalized_moment2.array().sqrt() + epsilon_).array()).matrix();
  }

  void update(double& weight, const double& gradient) {
    iter_ += 1;
    moment1_scl_ *= beta1_;
    moment1_scl_ += (1.0 - beta1_) * gradient;
    moment2_scl_ *= beta2_;
    moment2_scl_ += (1.0 - beta2_) * (gradient * gradient);
    const double normalized_moment1 = moment1_scl_ / (1.0 - std::pow(beta1_, iter_));
    const double normalized_moment2 = moment2_scl_ / (1.0 - std::pow(beta2_, iter_));
    weight -= eta_ * (normalized_moment1 / (std::sqrt(normalized_moment2) + epsilon_));
  }

private:
  const double eta_;
  const double beta1_;
  const double beta2_;
  const double epsilon_;
  uint32_t iter_;
  Eigen::VectorXd moment1_vec_;
  Eigen::VectorXd moment2_vec_;
  double moment1_scl_;
  double moment2_scl_;
};

#endif // SOFTDTREE_ADAM_HPP_
