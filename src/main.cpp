#include "soft_decision_tree.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

NB_MODULE(_softdtree, m) {
  nb::class_<Node>(m, "Node")
    .def(nb::init<>())
    .def_prop_rw("id",
        [](Node& n) { return n.id; },
        [](Node& n, const uint32_t id) { n.id = id; })
    .def_prop_rw("parent_id",
        [](Node& n) { return n.parent_id; },
        [](Node& n, const uint32_t parent_id) { n.parent_id = parent_id; })
    .def_prop_rw("left_child_id",
        [](Node& n) { return n.left_child_id; },
        [](Node& n, const uint32_t left_child_id) { n.left_child_id = left_child_id; })
    .def_prop_rw("right_child_id",
        [](Node& n) { return n.right_child_id; },
        [](Node& n, const uint32_t right_child_id) { n.right_child_id = right_child_id; })
    .def_prop_rw("depth",
        [](Node& n) { return n.depth; },
        [](Node& n, const uint32_t depth) { n.depth = depth; })
    .def_prop_rw("is_leaf",
        [](Node& n) { return n.is_leaf; },
        [](Node& n, const bool is_leaf) { n.is_leaf = is_leaf; })
    .def_prop_rw("is_left",
        [](Node& n) { return n.is_left; },
        [](Node& n, const bool is_left) { n.is_left = is_left; })
    .def_prop_rw("weight",
        [](Node& n) { return n.weight; },
        [](Node& n, const nb::DRef<Eigen::VectorXd>& weight) { n.weight = weight; })
    .def_prop_rw("bias",
        [](Node& n) { return n.bias; },
        [](Node& n, const double bias) { n.bias = bias; })
    .def_prop_rw("response",
        [](Node& n) { return n.response; },
        [](Node& n, const nb::DRef<Eigen::VectorXd>& response) { n.response = response; })
    .def_prop_ro("feature_ids", [](Node& n) { return n.feature_ids; })
    .def_prop_ro("has_parent", &Node::has_parent)
  ;
  nb::class_<BaseSoftDecisionTree>(m, "BaseSoftDecisionTree")
    .def(nb::init<uint32_t, double, uint32_t, uint32_t, double, double, double, double, double, int32_t, int32_t>(),
         nb::arg("max_depth") = 8, nb::arg("max_features") = 1.0, nb::arg("max_epoch") = 50, nb::arg("batch_size") = 1,
         nb::arg("eta") = 0.01, nb::arg("beta1") = 0.9, nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8,
         nb::arg("tol") = 1e-4,
         nb::arg("verbose") = 0,
         nb::arg("random_seed") = -1)
    .def("decision_function",
        [](BaseSoftDecisionTree& t, const nb::DRef<Eigen::MatrixXd>& x) {
          return t.decision_function(x);
        },
        nb::sig("def decision_function(self, x: numpy.typing.NDArray[numpy.float64]) -> numpy.typing.NDArray[numpy.float64]"))
  ;
  nb::class_<BaseSoftDecisionTreeClassifier, BaseSoftDecisionTree>(m, "BaseSoftDecisionTreeClassifier")
    .def(nb::init<uint32_t, double, uint32_t, uint32_t, double, double, double, double, double, int32_t, int32_t>(),
         nb::arg("max_depth") = 8, nb::arg("max_features") = 1.0, nb::arg("max_epoch") = 50, nb::arg("batch_size") = 1,
         nb::arg("eta") = 0.01, nb::arg("beta1") = 0.9, nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8,
         nb::arg("tol") = 1e-4,
         nb::arg("verbose") = 0,
         nb::arg("random_seed") = -1)
    .def("fit",
        [](BaseSoftDecisionTreeClassifier& t, const nb::DRef<Eigen::MatrixXd>& x, const nb::DRef<Eigen::MatrixXd>& y) {
          t.fit(x, y);
        })
  ;
  nb::class_<BaseSoftDecisionTreeRegressor, BaseSoftDecisionTree>(m, "BaseSoftDecisionTreeRegressor")
    .def(nb::init<uint32_t, double, uint32_t, uint32_t, double, double, double, double, double, int32_t, int32_t>(),
         nb::arg("max_depth") = 8, nb::arg("max_features") = 1.0, nb::arg("max_epoch") = 50, nb::arg("batch_size") = 1,
         nb::arg("eta") = 0.01, nb::arg("beta1") = 0.9, nb::arg("beta2") = 0.999, nb::arg("epsilon") = 1e-8,
         nb::arg("tol") = 1e-4,
         nb::arg("verbose") = 0,
         nb::arg("random_seed") = -1)
    .def("fit",
        [](BaseSoftDecisionTreeRegressor& t, const nb::DRef<Eigen::MatrixXd>& x, const nb::DRef<Eigen::MatrixXd>& y) {
          t.fit(x, y);
        })
  ;
}
