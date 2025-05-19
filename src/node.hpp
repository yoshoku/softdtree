#ifndef SOFTDTREE_NODE_HPP_
#define SOFTDTREE_NODE_HPP_

#include <cstdint>
#include <vector>

#include <Eigen/Dense>

struct Node {
  int32_t id;
  int32_t parent_id;
  int32_t left_child_id;
  int32_t right_child_id;
  uint32_t depth;
  bool is_leaf;
  bool is_left;
  Eigen::VectorXd weight;
  double bias;
  Eigen::VectorXd response;
  std::vector<uint32_t> feature_ids;

  Node(): id(-1), parent_id(-1), left_child_id(-1), right_child_id(-1),
    depth(0), is_leaf(true), is_left(true), bias(0) {}

  bool has_parent() {
    return parent_id > -1;
  }
};

#endif // SOFTDTREE_NODE_HPP_
