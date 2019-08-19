// ------------------------------------------------------------------
// Clinically Applicable Deep Learning Framework for Organs at Risk Delineation in CT images
// Copyright (c) 2019 Deep Voxel and University of California Irvine
// Licensed under License CC BY-NC-SA 4.0 [see UaNet/LICENSE for details]
// Written by Hao Tang
// ------------------------------------------------------------------


#include <torch/extension.h>
#include <iostream>
#include <math.h>
#include "nms.cpp"
#include "overlap.cpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cpu_overlap", &cpu_overlap, "box cpu_overlap");
    m.def("cpu_nms", &cpu_nms, "box cpu_nms");
}
