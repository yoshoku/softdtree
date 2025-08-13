#ifndef SOFTDTREE_SOFT_DECISION_TREE_HPP_
#define SOFTDTREE_SOFT_DECISION_TREE_HPP_

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _OPENMP
#pragma omp declare reduction (+: Eigen::VectorXd: omp_out=omp_out+omp_in) \
  initializer(omp_priv=Eigen::VectorXd::Zero(omp_orig.size()))
#endif

#include "adam.hpp"
#include "candidate.hpp"
#include "node.hpp"

class BaseSoftDecisionTree {
public:
  BaseSoftDecisionTree(
    const uint32_t max_depth = 8, const double max_features = 1.0, const uint32_t max_epoch = 50, const uint32_t batch_size = 1,
    const double eta = 0.01, const double beta1 = 0.9, const double beta2 = 0.999, const double epsilon = 1e-8,
    const double tol = 1e-4,
    const int32_t verbose = 0,
    const int32_t random_seed = -1
  ): max_depth_(max_depth), max_features_(max_features), max_epoch_(max_epoch), batch_size_(batch_size),
    eta_(eta), beta1_(beta1), beta2_(beta2), epsilon_(epsilon),
    tol_(tol),
    verbose_(verbose),
    random_seed_([random_seed]() { std::random_device seedg; return (random_seed < 0 ? seedg() : random_seed); }()),
    n_outputs_(0) {}

  void fit(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
    const uint32_t n_features = x.cols();
    n_components_ = std::max(static_cast<uint32_t>(1), std::min(n_features, static_cast<uint32_t>(std::floor(n_features * max_features_))));
    n_outputs_ = y.cols();
    fit_(x, y);
  }

  Eigen::MatrixXd decision_function(const Eigen::MatrixXd& x) {
    const uint32_t n_samples = x.rows();
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(n_samples, n_outputs_);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (uint32_t i = 0; i < n_samples; i++) {
      const std::vector<double> gating_values = calc_gate_functions_(x.row(i), nodes_);
      result.row(i) = evaluate_(x.row(i), nodes_, gating_values);
    }
    return result;
  }

protected:
  const uint32_t max_depth_;
  const double max_features_;
  const uint32_t max_epoch_;
  const uint32_t batch_size_;
  const double eta_;
  const double beta1_;
  const double beta2_;
  const double epsilon_;
  const double tol_;
  const int32_t verbose_;
  const uint32_t random_seed_;
  uint32_t n_outputs_;
  uint32_t n_components_;
  std::vector<std::shared_ptr<Node>> nodes_;
  std::mt19937 rng_;

  void fit_(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y) {
    rng_.seed(random_seed_);
    std::shared_ptr<Node> root = std::make_shared<Node>();
    root->id = 0;
    root->response = y.colwise().mean();
    nodes_.clear();
    nodes_.push_back(root);

    std::vector<std::shared_ptr<Node>> curr_candidates;
    curr_candidates.push_back(root);
    for (uint32_t d = 0; d < max_depth_; d++) {
      std::vector<std::shared_ptr<Node>> next_candidates;
      for (std::shared_ptr<Node>& node : curr_candidates) {
        Candidate candidate = find_split_candidates(x, y, node);
        if (candidate.split) {
          candidate.left->parent_id = node->id;
          candidate.right->parent_id = node->id;
          candidate.left->id = nodes_.size();
          nodes_.push_back(candidate.left);
          candidate.right->id = nodes_.size();
          nodes_.push_back(candidate.right);
          node->is_leaf = false;
          node->weight = candidate.weight;
          node->bias = candidate.bias;
          node->left_child_id = candidate.left->id;
          node->right_child_id = candidate.right->id;
          next_candidates.push_back(candidate.left);
          next_candidates.push_back(candidate.right);
        }
      }
      curr_candidates = next_candidates;
    }
  }

  Candidate find_split_candidates(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::shared_ptr<Node>& node) {
    Candidate candidate;

    // early stop
    if (node->depth > max_depth_) return candidate;

    // initialization
    const double base_error = error_(x, y, nodes_);
    std::vector<std::shared_ptr<Node>> tmp_nodes(nodes_.size());
    std::copy(nodes_.begin(), nodes_.end(), tmp_nodes.begin());
    // parent node
    const uint32_t n_features = x.cols();
    const uint32_t n_outputs = y.cols();
    std::uniform_real_distribution<double> dist(-0.005, 0.005);
    node->weight = Eigen::VectorXd::Zero(n_components_).unaryExpr([&](double disuse_) { return dist(rng_); });
    node->bias = dist(rng_);
    node->is_leaf = false;
    std::vector<uint32_t> seq(n_features);
    std::iota(seq.begin(), seq.end(), 0);
    if (n_components_ < n_features) {
      std::shuffle(seq.begin(), seq.end(), rng_);
      seq.resize(n_components_);
    }
    node->feature_ids = seq;
    // left node
    std::shared_ptr<Node> left = std::make_shared<Node>();
    left->depth = node->depth + 1;
    left->is_left = true;
    left->parent_id = node->id;
    left->response = Eigen::VectorXd::Zero(n_outputs).unaryExpr([&](double disuse_) { return dist(rng_); });
    left->id = tmp_nodes.size();
    tmp_nodes.push_back(left);
    node->left_child_id = left->id;
    // right node
    std::shared_ptr<Node> right = std::make_shared<Node>();
    right->depth = node->depth + 1;
    right->is_left = false;
    right->parent_id = node->id;
    right->response = Eigen::VectorXd::Zero(n_outputs).unaryExpr([&](double disuse_) { return dist(rng_); });
    right->id = tmp_nodes.size();
    tmp_nodes.push_back(right);
    node->right_child_id = right->id;

    // optimization
    learn_parameters_(x, y, node, tmp_nodes);

    // calculate error on splited node.
    const double new_error = error_(x, y, tmp_nodes);
    if (verbose_ > 0) {
      if (node->has_parent()) {
        std::cout << (node->is_left ? "left  " : "right ");
      } else {
        std::cout << "---" << std::endl << "root  ";
      }
      std::cout << "(" << node->depth << "), ";
      std::cout << "base_error: " << base_error << ", new_error: " << new_error << std::endl;
    }

    // Do not split if the new_error is larger than base_error or if the change is small
    if (base_error - new_error < tol_) {
      // restoration to original condition.
      node->is_leaf = true;
      node->feature_ids.clear();
      node->left_child_id = -1;
      node->right_child_id = -1;
      return candidate;
    }

    candidate.split = true;
    candidate.weight = node->weight;
    candidate.bias = node->bias;
    candidate.left = left;
    candidate.right = right;

    return candidate;
  }

  void learn_parameters_(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::shared_ptr<Node>& node, std::vector<std::shared_ptr<Node>>& nodes) {
    const uint32_t n_samples = x.rows();
    const uint32_t n_features = x.cols();
    const uint32_t n_outputs = y.cols();
    std::vector<uint32_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    Adam adam_weight = Adam(eta_, beta1_, beta2_, epsilon_);
    Adam adam_bias = Adam(eta_, beta1_, beta2_, epsilon_);
    Adam adam_l_response = Adam(eta_, beta1_, beta2_, epsilon_);
    Adam adam_r_response = Adam(eta_, beta1_, beta2_, epsilon_);

    for (uint32_t t = 0; t < max_epoch_; t++) {
      std::shuffle(indices.begin(), indices.end(), rng_);
      for (uint32_t n = 0; n < n_samples; n += batch_size_) {
        // prepare mini-batch and gradients.
        uint32_t end = n + batch_size_ < n_samples ? n + batch_size_ : n_samples;
        std::vector<uint32_t> batch_ids(indices.begin() + n, indices.begin() + end);
        const uint32_t sz_batch = batch_ids.size();
        Eigen::VectorXd grad_weight = Eigen::VectorXd::Zero(n_components_);
        double grad_bias = 0.0;
        Eigen::VectorXd grad_l_response = Eigen::VectorXd::Zero(n_outputs);
        Eigen::VectorXd grad_r_response = Eigen::VectorXd::Zero(n_outputs);
        // calculate gradients with mini-batch.
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:grad_weight,grad_bias,grad_l_response,grad_r_response)
        #endif
        for (uint32_t i = 0; i < sz_batch; i++) {
          const Eigen::VectorXd x_vec = x.row(batch_ids[i]);
          const Eigen::VectorXd y_vec = y.row(batch_ids[i]);
          const std::vector<double> gating_values = calc_gate_functions_(x_vec, nodes);
          const Eigen::VectorXd y_pred = evaluate_(x_vec, nodes, gating_values);
          Eigen::VectorXd delta = y_pred - y_vec;
          std::shared_ptr<Node> m = node;
          while (m->has_parent()) {
            if (m->is_left) {
              delta *= gating_values[m->parent_id];
            } else {
              delta *= 1.0 - gating_values[m->parent_id];
            }
            m = nodes[m->parent_id];
          }
          const double g = gating_values[node->id];
          const Eigen::VectorXd alpha = (nodes[node->left_child_id]->response - nodes[node->right_child_id]->response) * (g * (1.0 - g));
          const double coeff = delta.dot(alpha);
          grad_weight += coeff * x_vec(node->feature_ids);
          grad_bias += coeff;
          grad_l_response += delta * g;
          grad_r_response += delta * (1.0 - g);
        }
        grad_weight /= sz_batch;
        grad_bias /= sz_batch;
        grad_l_response /= sz_batch;
        grad_r_response /= sz_batch;
        // update parameters.
        adam_weight.update(node->weight, grad_weight);
        adam_bias.update(node->bias, grad_bias);
        adam_l_response.update(nodes[node->left_child_id]->response, grad_l_response);
        adam_r_response.update(nodes[node->right_child_id]->response, grad_r_response);
      }
    }
  }

  double error_(const Eigen::MatrixXd& x, const Eigen::MatrixXd& y, const std::vector<std::shared_ptr<Node>>& nodes) {
    const uint32_t n_samples = x.rows();
    double error = 0.0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+:error)
    #endif
    for (uint32_t i = 0; i < n_samples; i++) {
      const std::vector<double> gating_values = calc_gate_functions_(x.row(i), nodes);
      const Eigen::VectorXd y_pred = evaluate_(x.row(i), nodes, gating_values);
      const Eigen::VectorXd y_true = y.row(i);
      const Eigen::VectorXd diff = y_pred - y_true;
      error += diff.dot(diff);
    }
    return error;
  }

  std::vector<double> calc_gate_functions_(const Eigen::VectorXd& x, const std::vector<std::shared_ptr<Node>>& nodes) {
    uint32_t n_nodes = nodes.size();
    std::vector<double> gating_values(n_nodes, 0.0);
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (uint32_t i = 0; i < n_nodes; i++) {
      if (!nodes[i]->is_leaf) {
        const double v = x(nodes[i]->feature_ids).dot(nodes[i]->weight) + nodes[i]->bias;
        const double g = 0.5 * (std::tanh(0.5 * v) + 1.0);
        gating_values[nodes[i]->id] = g;
      }
    }
    return gating_values;
  }

  Eigen::VectorXd calc_response_(const Eigen::VectorXd& x, const std::vector<std::shared_ptr<Node>>& nodes, const std::vector<double>& gating_values, const uint32_t id) {
    const std::shared_ptr<Node> n = nodes[id];
    if (n->is_leaf) return n->response;
    const double g = gating_values[id];
    return g * calc_response_(x, nodes, gating_values, n->left_child_id) + (1.0 - g) * calc_response_(x, nodes, gating_values, n->right_child_id);
  }

  virtual Eigen::VectorXd evaluate_(
      const Eigen::VectorXd& x,
      const std::vector<std::shared_ptr<Node>>& nodes,
      const std::vector<double>& gating_values) { return Eigen::VectorXd::Zero(1); }
};

class BaseSoftDecisionTreeClassifier : public BaseSoftDecisionTree {
public:
  BaseSoftDecisionTreeClassifier(
    const uint32_t max_depth = 8, const double max_features = 1.0, const uint32_t max_epoch = 50, const uint32_t batch_size = 1,
    const double eta = 0.01, const double beta1 = 0.9, const double beta2 = 0.999, const double epsilon = 1e-8,
    const double tol = 1e-4,
    const int32_t verbose = 0,
    const int32_t random_seed = -1
  ): BaseSoftDecisionTree(max_depth, max_features, max_epoch, batch_size,
    eta, beta1, beta2, epsilon, tol, verbose, random_seed) {}

protected:
  Eigen::VectorXd evaluate_(const Eigen::VectorXd& x, const std::vector<std::shared_ptr<Node>>& nodes, const std::vector<double>& gating_values) override {
    Eigen::VectorXd r = calc_response_(x, nodes, gating_values, 0);
    return (r.size() > 1 ? softmax_(r) : sigmoid_(r));
  }

  Eigen::VectorXd softmax_(const Eigen::VectorXd& v) {
    const double max_v = v.maxCoeff();
    const Eigen::VectorXd exp_v = (v.array() - max_v).exp();
    return exp_v / exp_v.sum();
  }

  Eigen::VectorXd sigmoid_(const Eigen::VectorXd& v) {
    const double tmp = 0.5 * (std::tanh(0.5 * v(0)) + 1.0);
    return Eigen::VectorXd::Constant(1, tmp);
  }
};

class BaseSoftDecisionTreeRegressor : public BaseSoftDecisionTree {
public:
  BaseSoftDecisionTreeRegressor(
    const uint32_t max_depth = 8, const double max_features = 1.0, const uint32_t max_epoch = 50, const uint32_t batch_size = 1,
    const double eta = 0.01, const double beta1 = 0.9, const double beta2 = 0.999, const double epsilon = 1e-8,
    const double tol = 1e-4,
    const int32_t verbose = 0,
    const int32_t random_seed = -1
  ): BaseSoftDecisionTree(max_depth, max_features, max_epoch, batch_size,
    eta, beta1, beta2, epsilon, tol, verbose, random_seed) {}

protected:
  Eigen::VectorXd evaluate_(const Eigen::VectorXd& x, const std::vector<std::shared_ptr<Node>>& nodes, const std::vector<double>& gating_values) override {
    return calc_response_(x, nodes, gating_values, 0);
  }
};

#endif // SOFTDTREE_SOFT_DECISION_TREE_HPP_
