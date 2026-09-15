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

    std::vector<double> scores(left.size() * right.size(), 0.0);
    Py_BEGIN_ALLOW_THREADS
    for (size_t row = 0; row < left.size(); ++row) {
        for (size_t column = 0; column < right.size(); ++column) {
            scores[row * right.size() + column] =
                intersection_over_union(left[row].data(), right[column].data());
        }
    }
    Py_END_ALLOW_THREADS

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
            PyObject* value = PyFloat_FromDouble(scores[row * right.size() + column]);
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

PyObject* nms_indices(PyObject*, PyObject* arguments) {
    PyObject* box_objects = nullptr;
    PyObject* score_objects = nullptr;
    double threshold = 0.0;
    PyObject* class_objects = Py_None;
    if (!PyArg_ParseTuple(
            arguments, "OOd|O:nms_indices", &box_objects, &score_objects, &threshold,
            &class_objects)) {
        return nullptr;
    }
    if (threshold < 0.0 || threshold > 1.0) {
        PyErr_SetString(PyExc_ValueError, "iou_threshold must be in [0, 1]");
        return nullptr;
    }

    PyObject* boxes_sequence = PySequence_Fast(box_objects, "boxes must be a sequence");
    if (boxes_sequence == nullptr) {
        return nullptr;
    }
    PyObject* scores_sequence = PySequence_Fast(score_objects, "scores must be a sequence");
    if (scores_sequence == nullptr) {
        Py_DECREF(boxes_sequence);
        return nullptr;
    }
    const Py_ssize_t box_count = PySequence_Fast_GET_SIZE(boxes_sequence);
    if (box_count != PySequence_Fast_GET_SIZE(scores_sequence)) {
        PyErr_SetString(PyExc_ValueError, "boxes and scores must have the same length");
        Py_DECREF(boxes_sequence);
        Py_DECREF(scores_sequence);
        return nullptr;
    }

    PyObject* classes_sequence = nullptr;
    if (class_objects != Py_None) {
        classes_sequence = PySequence_Fast(class_objects, "class_ids must be a sequence");
        if (classes_sequence == nullptr) {
            Py_DECREF(boxes_sequence);
            Py_DECREF(scores_sequence);
            return nullptr;
        }
        if (box_count != PySequence_Fast_GET_SIZE(classes_sequence)) {
            PyErr_SetString(PyExc_ValueError, "boxes and class_ids must have the same length");
            Py_DECREF(boxes_sequence);
            Py_DECREF(scores_sequence);
            Py_DECREF(classes_sequence);
            return nullptr;
        }
    }

    std::vector<std::array<double, 4>> boxes;
    std::vector<double> scores;
    std::vector<long> classes;
    boxes.reserve(box_count);
    scores.reserve(box_count);
    if (classes_sequence != nullptr) {
        classes.reserve(box_count);
    }
    for (Py_ssize_t index = 0; index < box_count; ++index) {
        double values[4];
        if (!read_box(PySequence_Fast_GET_ITEM(boxes_sequence, index), values)) {
            Py_DECREF(boxes_sequence);
            Py_DECREF(scores_sequence);
            Py_XDECREF(classes_sequence);
            return nullptr;
        }
        const double score = PyFloat_AsDouble(PySequence_Fast_GET_ITEM(scores_sequence, index));
        if (PyErr_Occurred()) {
            Py_DECREF(boxes_sequence);
            Py_DECREF(scores_sequence);
            Py_XDECREF(classes_sequence);
            return nullptr;
        }
        if (!std::isfinite(score)) {
            PyErr_SetString(PyExc_ValueError, "scores must be finite");
            Py_DECREF(boxes_sequence);
            Py_DECREF(scores_sequence);
            Py_XDECREF(classes_sequence);
            return nullptr;
        }
        boxes.push_back({values[0], values[1], values[2], values[3]});
        scores.push_back(score);
        if (classes_sequence != nullptr) {
            const long class_id = PyLong_AsLong(PySequence_Fast_GET_ITEM(classes_sequence, index));
            if (PyErr_Occurred()) {
                Py_DECREF(boxes_sequence);
                Py_DECREF(scores_sequence);
                Py_DECREF(classes_sequence);
                return nullptr;
            }
            classes.push_back(class_id);
        }
    }
    Py_DECREF(boxes_sequence);
    Py_DECREF(scores_sequence);
    Py_XDECREF(classes_sequence);

    std::vector<size_t> order(boxes.size());
    for (size_t index = 0; index < order.size(); ++index) {
        order[index] = index;
    }
    std::stable_sort(order.begin(), order.end(), [&scores](size_t left, size_t right) {
        return scores[left] > scores[right];
    });
    std::vector<size_t> kept;
    std::vector<bool> suppressed(boxes.size(), false);
    Py_BEGIN_ALLOW_THREADS
    for (size_t position = 0; position < order.size(); ++position) {
        const size_t current = order[position];
        if (suppressed[current]) {
            continue;
        }
        kept.push_back(current);
        for (size_t next_position = position + 1; next_position < order.size(); ++next_position) {
            const size_t next = order[next_position];
            if (suppressed[next]) {
                continue;
            }
            if (!classes.empty() && classes[current] != classes[next]) {
                continue;
            }
            if (intersection_over_union(boxes[current].data(), boxes[next].data()) > threshold) {
                suppressed[next] = true;
            }
        }
    }
    Py_END_ALLOW_THREADS

    PyObject* result = PyList_New(static_cast<Py_ssize_t>(kept.size()));
    if (result == nullptr) {
        return nullptr;
    }
    for (size_t position = 0; position < kept.size(); ++position) {
        PyObject* value = PyLong_FromSize_t(kept[position]);
        if (value == nullptr) {
            Py_DECREF(result);
            return nullptr;
        }
        PyList_SET_ITEM(result, static_cast<Py_ssize_t>(position), value);
    }
    return result;
}

PyMethodDef methods[] = {
    {"iou_matrix", iou_matrix, METH_VARARGS, "Compute an IoU matrix in native code."},
    {"nms_indices", nms_indices, METH_VARARGS, "Compute greedy NMS indices in native code."},
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
