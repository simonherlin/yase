#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace {

bool read_box(PyObject* object, double values[4]) {
    PyObject* sequence = PySequence_Fast(object, "box must be a sequence");
    if (sequence == nullptr) {
        return false;
    }
    const Py_ssize_t size = PySequence_Fast_GET_SIZE(sequence);
    if (size < 4) {
        PyErr_SetString(PyExc_ValueError, "box must contain at least four values");
        Py_DECREF(sequence);
        return false;
    }
    for (int index = 0; index < 4; ++index) {
        values[index] = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(sequence, index));
        if (PyErr_Occurred()) {
            Py_DECREF(sequence);
            return false;
        }
    }
    Py_DECREF(sequence);
    return true;
}

double intersection_over_union(const double left[4], const double right[4]) {
    const double x1 = std::max(left[0], right[0]);
    const double y1 = std::max(left[1], right[1]);
    const double x2 = std::min(left[2], right[2]);
    const double y2 = std::min(left[3], right[3]);
    const double intersection = std::max(0.0, x2 - x1) * std::max(0.0, y2 - y1);
    const double left_area = std::max(0.0, left[2] - left[0]) *
                             std::max(0.0, left[3] - left[1]);
    const double right_area = std::max(0.0, right[2] - right[0]) *
                              std::max(0.0, right[3] - right[1]);
    const double union_area = left_area + right_area - intersection;
    return union_area > 0.0 ? intersection / union_area : 0.0;
}

PyObject* iou_matrix(PyObject*, PyObject* arguments) {
    PyObject* left_objects = nullptr;
    PyObject* right_objects = nullptr;
    if (!PyArg_ParseTuple(arguments, "OO:iou_matrix", &left_objects, &right_objects)) {
        return nullptr;
    }
    PyObject* left_sequence = PySequence_Fast(left_objects, "left must be a sequence");
    if (left_sequence == nullptr) {
        return nullptr;
    }
    PyObject* right_sequence = PySequence_Fast(right_objects, "right must be a sequence");
    if (right_sequence == nullptr) {
        Py_DECREF(left_sequence);
        return nullptr;
    }

    std::vector<std::array<double, 4>> left;
    std::vector<std::array<double, 4>> right;
    left.reserve(PySequence_Fast_GET_SIZE(left_sequence));
    right.reserve(PySequence_Fast_GET_SIZE(right_sequence));
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(left_sequence); ++index) {
        double values[4];
        if (!read_box(PySequence_Fast_GET_ITEM(left_sequence, index), values)) {
            Py_DECREF(left_sequence);
            Py_DECREF(right_sequence);
            return nullptr;
        }
        left.push_back({values[0], values[1], values[2], values[3]});
    }
    for (Py_ssize_t index = 0; index < PySequence_Fast_GET_SIZE(right_sequence); ++index) {
        double values[4];
        if (!read_box(PySequence_Fast_GET_ITEM(right_sequence, index), values)) {
            Py_DECREF(left_sequence);
            Py_DECREF(right_sequence);
            return nullptr;
        }
        right.push_back({values[0], values[1], values[2], values[3]});
    }
    Py_DECREF(left_sequence);
    Py_DECREF(right_sequence);

    PyObject* result = PyList_New(static_cast<Py_ssize_t>(left.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (size_t row = 0; row < left.size(); ++row) {
        PyObject* values = PyList_New(static_cast<Py_ssize_t>(right.size()));
        if (values == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        for (size_t column = 0; column < right.size(); ++column) {
            const double score = intersection_over_union(left[row].data(), right[column].data());
            PyObject* value = PyFloat_FromDouble(score);
            if (value == nullptr) {
                Py_DECREF(values);
                Py_DECREF(result);
                return nullptr;
            }
            PyList_SET_ITEM(values, static_cast<Py_ssize_t>(column), value);
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(row), values);
    }
    return result;
}

PyMethodDef methods[] = {
    {"iou_matrix", iou_matrix, METH_VARARGS, "Compute an IoU matrix in native code."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Optional native acceleration for Yase.",
    -1,
    methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__native() {
    return PyModule_Create(&module);
}
