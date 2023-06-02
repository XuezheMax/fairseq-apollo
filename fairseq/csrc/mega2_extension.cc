#include <torch/torch.h>

#include "ops/timestep_norm.h"
#include "utils.h"

namespace mega2 {

PYBIND11_MODULE(mega2_extension, m) {
  m.doc() = "Mega2 Cpp Extensions.";
  py::module m_ops = m.def_submodule("ops", "Submodule for custom ops.");
  ops::DefineTimestepNormOp(m_ops);
}

}  // namespace mega2
