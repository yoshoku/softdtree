#ifndef SOFTDTREE_CANDIDATE_HPP_
#define SOFTDTREE_CANDIDATE_HPP_

#include <memory>

#include <Eigen/Dense>

#include "node.hpp"

struct Candidate {
  Candidate(): split(false) {}
  bool split;
  Eigen::MatrixXd weight;
  double bias;
  std::shared_ptr<Node> left;
  std::shared_ptr<Node> right;
};

#endif // SOFTDTREE_CANDIDATE_HPP_
