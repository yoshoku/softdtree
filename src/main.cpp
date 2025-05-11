#include <nanobind/nanobind.h>

namespace nb = nanobind;

int add(int a, int b) {
  return a + b;
}

NB_MODULE(_core, m) {
  m.def("add", &add);
}
