#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "qdiag_kernel.h"

static int qdiag_same_shape(PyArrayObject *a, PyArrayObject *b)
{
    const int ndim = PyArray_NDIM(a);
    int axis;

    if (ndim != PyArray_NDIM(b)) {
        return 0;
    }

    for (axis = 0; axis < ndim; ++axis) {
        if (PyArray_DIM(a, axis) != PyArray_DIM(b, axis)) {
            return 0;
        }
    }

    return 1;
}

static int qdiag_shape_append(
    PyArrayObject *source,
    npy_intp *dims,
    int extra_dims
)
{
    const int ndim = PyArray_NDIM(source);
    int axis;

    if (ndim + extra_dims > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "input has too many dimensions");
        return 0;
    }

    for (axis = 0; axis < ndim; ++axis) {
        dims[axis] = PyArray_DIM(source, axis);
    }

    return 1;
}

static PyObject *qdiag_py_eigh_q(PyObject *self, PyObject *args)
{
    PyObject *obj_qxx;
    PyObject *obj_qyy;
    PyObject *obj_qxy;
    PyObject *obj_qxz;
    PyObject *obj_qyz;

    PyArrayObject *arr_qxx = NULL;
    PyArrayObject *arr_qyy = NULL;
    PyArrayObject *arr_qxy = NULL;
    PyArrayObject *arr_qxz = NULL;
    PyArrayObject *arr_qyz = NULL;
    PyArrayObject *out_w = NULL;
    PyArrayObject *out_v = NULL;

    npy_intp dims_w[NPY_MAXDIMS];
    npy_intp dims_v[NPY_MAXDIMS];

    if (!PyArg_ParseTuple(
        args,
        "OOOOO:eigh_q",
        &obj_qxx,
        &obj_qyy,
        &obj_qxy,
        &obj_qxz,
        &obj_qyz
    )) {
        return NULL;
    }

    /*
     * Convert to aligned, C-contiguous float64 arrays.  If the caller already
     * supplies that format these are zero-copy views/references.
     */
    arr_qxx = (PyArrayObject *)PyArray_FROM_OTF(
        obj_qxx, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    arr_qyy = (PyArrayObject *)PyArray_FROM_OTF(
        obj_qyy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    arr_qxy = (PyArrayObject *)PyArray_FROM_OTF(
        obj_qxy, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    arr_qxz = (PyArrayObject *)PyArray_FROM_OTF(
        obj_qxz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );
    arr_qyz = (PyArrayObject *)PyArray_FROM_OTF(
        obj_qyz, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY
    );

    if (
        arr_qxx == NULL ||
        arr_qyy == NULL ||
        arr_qxy == NULL ||
        arr_qxz == NULL ||
        arr_qyz == NULL
    ) {
        goto fail;
    }

    if (
        !qdiag_same_shape(arr_qxx, arr_qyy) ||
        !qdiag_same_shape(arr_qxx, arr_qxy) ||
        !qdiag_same_shape(arr_qxx, arr_qxz) ||
        !qdiag_same_shape(arr_qxx, arr_qyz)
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "all five Q component arrays must have exactly the same shape"
        );
        goto fail;
    }

    const int ndim = PyArray_NDIM(arr_qxx);

    if (!qdiag_shape_append(arr_qxx, dims_w, 1)) {
        goto fail;
    }
    if (!qdiag_shape_append(arr_qxx, dims_v, 2)) {
        goto fail;
    }

    dims_w[ndim] = 3;
    dims_v[ndim] = 3;
    dims_v[ndim + 1] = 3;

    out_w = (PyArrayObject *)PyArray_SimpleNew(
        ndim + 1, dims_w, NPY_DOUBLE
    );
    out_v = (PyArrayObject *)PyArray_SimpleNew(
        ndim + 2, dims_v, NPY_DOUBLE
    );

    if (out_w == NULL || out_v == NULL) {
        goto fail;
    }

    const npy_intp count = PyArray_SIZE(arr_qxx);

    const double *qxx = (const double *)PyArray_DATA(arr_qxx);
    const double *qyy = (const double *)PyArray_DATA(arr_qyy);
    const double *qxy = (const double *)PyArray_DATA(arr_qxy);
    const double *qxz = (const double *)PyArray_DATA(arr_qxz);
    const double *qyz = (const double *)PyArray_DATA(arr_qyz);

    double *w = (double *)PyArray_DATA(out_w);
    double *v = (double *)PyArray_DATA(out_v);

    /*
     * Release the GIL while the pure-C batch loop runs.  The kernel is
     * single-threaded in v1 by design.
     */
    Py_BEGIN_ALLOW_THREADS

    for (npy_intp index = 0; index < count; ++index) {
        qdiag_solve_q3(
            qxx[index],
            qyy[index],
            qxy[index],
            qxz[index],
            qyz[index],
            w + 3*index,
            v + 9*index
        );
    }

    Py_END_ALLOW_THREADS

    Py_DECREF(arr_qxx);
    Py_DECREF(arr_qyy);
    Py_DECREF(arr_qxy);
    Py_DECREF(arr_qxz);
    Py_DECREF(arr_qyz);

    return Py_BuildValue("NN", (PyObject *)out_w, (PyObject *)out_v);

fail:
    Py_XDECREF(arr_qxx);
    Py_XDECREF(arr_qyy);
    Py_XDECREF(arr_qxy);
    Py_XDECREF(arr_qxz);
    Py_XDECREF(arr_qyz);
    Py_XDECREF(out_w);
    Py_XDECREF(out_v);
    return NULL;
}

static PyObject *qdiag_py_qfield5(PyObject *self, PyObject *args, int dominant)
{
    PyObject *input;
    PyArrayObject *qfield = NULL;
    PyArrayObject *out_w = NULL;
    PyArrayObject *out_v = NULL;
    npy_intp dims_w[NPY_MAXDIMS];
    npy_intp dims_v[NPY_MAXDIMS];

    if (!PyArg_ParseTuple(args, "O", &input)) {
        return NULL;
    }

    qfield = (PyArrayObject *)PyArray_FROM_OTF(
        input, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY
    );
    if (qfield == NULL) {
        return NULL;
    }

    const int ndim = PyArray_NDIM(qfield);
    const int type = PyArray_TYPE(qfield);
    if (ndim < 1 || PyArray_DIM(qfield, ndim - 1) != 5) {
        PyErr_SetString(PyExc_ValueError, "QField5 input must have shape (..., 5)");
        goto fail;
    }
    if (type != NPY_FLOAT && type != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "QField5 input dtype must be float32 or float64");
        goto fail;
    }
    if (ndim + (dominant ? 0 : 1) > NPY_MAXDIMS) {
        PyErr_SetString(PyExc_ValueError, "input has too many dimensions");
        goto fail;
    }

    for (int axis = 0; axis < ndim - 1; ++axis) {
        dims_w[axis] = PyArray_DIM(qfield, axis);
        dims_v[axis] = PyArray_DIM(qfield, axis);
    }
    if (dominant) {
        dims_v[ndim - 1] = 3;
        out_w = (PyArrayObject *)PyArray_SimpleNew(ndim - 1, dims_w, NPY_DOUBLE);
        out_v = (PyArrayObject *)PyArray_SimpleNew(ndim, dims_v, NPY_DOUBLE);
    } else {
        dims_w[ndim - 1] = 3;
        dims_v[ndim - 1] = 3;
        dims_v[ndim] = 3;
        out_w = (PyArrayObject *)PyArray_SimpleNew(ndim, dims_w, NPY_DOUBLE);
        out_v = (PyArrayObject *)PyArray_SimpleNew(ndim + 1, dims_v, NPY_DOUBLE);
    }
    if (out_w == NULL || out_v == NULL) {
        goto fail;
    }

    const npy_intp count = PyArray_SIZE(qfield) / 5;
    double *w = (double *)PyArray_DATA(out_w);
    double *v = (double *)PyArray_DATA(out_v);

    Py_BEGIN_ALLOW_THREADS
    if (type == NPY_FLOAT) {
        const float *q = (const float *)PyArray_DATA(qfield);
        for (npy_intp index = 0; index < count; ++index, q += 5) {
            if (dominant) {
                qdiag_dominant_q3(q[0], q[3], q[1], q[2], q[4], w + index, v + 3*index);
            } else {
                qdiag_solve_q3(q[0], q[3], q[1], q[2], q[4], w + 3*index, v + 9*index);
            }
        }
    } else {
        const double *q = (const double *)PyArray_DATA(qfield);
        for (npy_intp index = 0; index < count; ++index, q += 5) {
            if (dominant) {
                qdiag_dominant_q3(q[0], q[3], q[1], q[2], q[4], w + index, v + 3*index);
            } else {
                qdiag_solve_q3(q[0], q[3], q[1], q[2], q[4], w + 3*index, v + 9*index);
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(qfield);
    return Py_BuildValue("NN", (PyObject *)out_w, (PyObject *)out_v);

fail:
    Py_XDECREF(qfield);
    Py_XDECREF(out_w);
    Py_XDECREF(out_v);
    return NULL;
}

static PyObject *qdiag_py_eigh_qfield5(PyObject *self, PyObject *args)
{
    return qdiag_py_qfield5(self, args, 0);
}

static PyObject *qdiag_py_dominant_qfield5(PyObject *self, PyObject *args)
{
    return qdiag_py_qfield5(self, args, 1);
}

static PyObject *qdiag_py_qfield5_into(PyObject *self, PyObject *args, int dominant)
{
    PyObject *input;
    PyObject *output_w;
    PyObject *output_v;
    PyArrayObject *qfield = NULL;

    if (!PyArg_ParseTuple(args, "OOO", &input, &output_w, &output_v)) {
        return NULL;
    }
    qfield = (PyArrayObject *)PyArray_FROM_OTF(
        input, NPY_NOTYPE, NPY_ARRAY_IN_ARRAY
    );
    if (qfield == NULL) {
        return NULL;
    }

    const int ndim = PyArray_NDIM(qfield);
    const int type = PyArray_TYPE(qfield);
    if (ndim != 2 || PyArray_DIM(qfield, 1) != 5) {
        PyErr_SetString(PyExc_ValueError, "input must have shape (N, 5)");
        goto fail;
    }
    if (type != NPY_FLOAT && type != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "input dtype must be float32 or float64");
        goto fail;
    }
    if (!PyArray_Check(output_w) || !PyArray_Check(output_v)) {
        PyErr_SetString(PyExc_TypeError, "outputs must be NumPy arrays");
        goto fail;
    }

    PyArrayObject *out_w = (PyArrayObject *)output_w;
    PyArrayObject *out_v = (PyArrayObject *)output_v;
    const npy_intp count = PyArray_DIM(qfield, 0);
    const npy_intp expected_w = dominant ? count : 3*count;
    const npy_intp expected_v = dominant ? 3*count : 9*count;
    if (
        PyArray_TYPE(out_w) != NPY_DOUBLE ||
        PyArray_TYPE(out_v) != NPY_DOUBLE ||
        !PyArray_ISCARRAY(out_w) ||
        !PyArray_ISCARRAY(out_v) ||
        PyArray_SIZE(out_w) != expected_w ||
        PyArray_SIZE(out_v) != expected_v
    ) {
        PyErr_SetString(
            PyExc_ValueError,
            "outputs must be writable C-contiguous float64 arrays of the expected size"
        );
        goto fail;
    }

    double *w = (double *)PyArray_DATA(out_w);
    double *v = (double *)PyArray_DATA(out_v);
    Py_BEGIN_ALLOW_THREADS
    if (type == NPY_FLOAT) {
        const float *q = (const float *)PyArray_DATA(qfield);
        for (npy_intp index = 0; index < count; ++index, q += 5) {
            if (dominant) {
                qdiag_dominant_q3(q[0], q[3], q[1], q[2], q[4], w + index, v + 3*index);
            } else {
                qdiag_solve_q3(q[0], q[3], q[1], q[2], q[4], w + 3*index, v + 9*index);
            }
        }
    } else {
        const double *q = (const double *)PyArray_DATA(qfield);
        for (npy_intp index = 0; index < count; ++index, q += 5) {
            if (dominant) {
                qdiag_dominant_q3(q[0], q[3], q[1], q[2], q[4], w + index, v + 3*index);
            } else {
                qdiag_solve_q3(q[0], q[3], q[1], q[2], q[4], w + 3*index, v + 9*index);
            }
        }
    }
    Py_END_ALLOW_THREADS

    Py_DECREF(qfield);
    Py_RETURN_NONE;

fail:
    Py_XDECREF(qfield);
    return NULL;
}

static PyObject *qdiag_py_eigh_qfield5_into(PyObject *self, PyObject *args)
{
    return qdiag_py_qfield5_into(self, args, 0);
}

static PyObject *qdiag_py_dominant_qfield5_into(PyObject *self, PyObject *args)
{
    return qdiag_py_qfield5_into(self, args, 1);
}

static PyMethodDef qdiag_methods[] = {
    {
        "eigh_q",
        qdiag_py_eigh_q,
        METH_VARARGS,
        PyDoc_STR(
            "eigh_q(qxx, qyy, qxy, qxz, qyz) -> (eigenvalues, eigenvectors)\n"
            "\n"
            "Diagonalize batches of real symmetric traceless 3x3 Q tensors.\n"
            "Eigenvalues are ascending; eigenvectors are stored in columns.\n"
        )
    },
    {
        "eigh_qfield5",
        qdiag_py_eigh_qfield5,
        METH_VARARGS,
        PyDoc_STR("eigh_qfield5(q) -> (eigenvalues, eigenvectors)\n")
    },
    {
        "dominant_qfield5",
        qdiag_py_dominant_qfield5,
        METH_VARARGS,
        PyDoc_STR("dominant_qfield5(q) -> (largest_eigenvalue, eigenvector)\n")
    },
    {
        "eigh_qfield5_into",
        qdiag_py_eigh_qfield5_into,
        METH_VARARGS,
        PyDoc_STR("eigh_qfield5_into(q, eigenvalues, eigenvectors) -> None\n")
    },
    {
        "dominant_qfield5_into",
        qdiag_py_dominant_qfield5_into,
        METH_VARARGS,
        PyDoc_STR("dominant_qfield5_into(q, eigenvalue, eigenvector) -> None\n")
    },
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef qdiag_module = {
    PyModuleDef_HEAD_INIT,
    "_core",
    "Portable C backend for Nematics3D Q diagonalization.",
    -1,
    qdiag_methods
};

PyMODINIT_FUNC PyInit__core(void)
{
    import_array();
    return PyModule_Create(&qdiag_module);
}
