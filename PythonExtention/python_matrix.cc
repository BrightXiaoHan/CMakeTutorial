#include <Python.h>
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;

typedef struct
{
    PyObject_HEAD
        MatrixXd matrix;
} PyMatrixObject;

static PyObject *
PyMatrix_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    PyMatrixObject *self;
    self = (PyMatrixObject *)type->tp_alloc(type, 0);

    char *kwlist[] = {"width", "height", NULL};
    int width = 0;
    int height = 0;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "ii", kwlist,
                                     &width, &height))
    {
        Py_DECREF(self);
        return NULL;
    }
    if (width <= 0 or height <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "The height and width must be greater than 0.");
        return NULL;
    }

    MatrixXd mat(width, height);
    self->matrix = mat;
    return (PyObject *)self;
}

#define PARSE_MATRIX(m) ((PyMatrixObject *)m)->matrix

#define RETURN_MATRIX(m, t)                              \
    PyMatrixObject *c = PyObject_NEW(PyMatrixObject, t); \
    c->matrix = m;                                       \
    return (PyObject *)c

static PyObject *
PyMatrix_add(PyObject *a, PyObject *b)
{
    MatrixXd matrix_a = PARSE_MATRIX(a);
    MatrixXd matrix_b = PARSE_MATRIX(b);

    if (matrix_a.cols() != matrix_b.cols() or matrix_a.rows() != matrix_b.rows()){
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    MatrixXd matrix_c = matrix_a + matrix_b;
    RETURN_MATRIX(matrix_c, a->ob_type);
}

static PyObject *
PyMatrix_minus(PyObject *a, PyObject *b)
{
    MatrixXd matrix_a = PARSE_MATRIX(a);
    MatrixXd matrix_b = PARSE_MATRIX(b);

    if (matrix_a.cols() != matrix_b.cols() or matrix_a.rows() != matrix_b.rows()){
        PyErr_SetString(PyExc_ValueError, "The input matrix must be the same shape.");
        return NULL;
    }

    MatrixXd matrix_c = matrix_a - matrix_b;
    RETURN_MATRIX(matrix_c, a->ob_type);
}

static PyObject *
PyMatrix_multiply(PyObject *a, PyObject *b)
{
    MatrixXd matrix_a = PARSE_MATRIX(a);
    MatrixXd matrix_b = PARSE_MATRIX(b);

    if (matrix_a.cols() != matrix_b.rows()){
        PyErr_SetString(PyExc_ValueError, "The colonm rank of matrix A must be the same as the row rank of matrix B.");
        return NULL;
    }
    MatrixXd matrix_c = matrix_a * matrix_b;
    RETURN_MATRIX(matrix_c, a->ob_type);
}

static PyObject *PyMatrix_str(PyObject *a)
{
    MatrixXd matrix = PARSE_MATRIX(a);
    std::stringstream ss;
    ss << matrix;
    return Py_BuildValue("s", ss.str().c_str());
}

static PyNumberMethods numberMethods = {
    PyMatrix_add,      //nb_add
    PyMatrix_minus,    //nb_subtract;
    PyMatrix_multiply, //nb_multiply
    nullptr,           //nb_remainder;
    nullptr,           //nb_divmod;
    nullptr,           // nb_power;
    nullptr,           // nb_negative;
    nullptr,           // nb_positive;
    nullptr,           // nb_absolute;
    nullptr,           // nb_bool;
    nullptr,           // nb_invert;
    nullptr,           // nb_lshift;
    nullptr,           // nb_rshift;
    nullptr,           // nb_and;
    nullptr,           // nb_xor;
    nullptr,           // nb_or;
    nullptr,           // nb_int;
    nullptr,           // nb_reserved;
    nullptr,           // nb_float;

    nullptr, // nb_inplace_add;
    nullptr, // nb_inplace_subtract;
    nullptr, // nb_inplace_multiply;
    nullptr, // nb_inplace_remainder;
    nullptr, // nb_inplace_power;
    nullptr, // nb_inplace_lshift;
    nullptr, // nb_inplace_rshift;
    nullptr, // nb_inplace_and;
    nullptr, // nb_inplace_xor;
    nullptr, // nb_inplace_or;

    nullptr, // nb_floor_divide;
    nullptr, // nb_true_divide;
    nullptr, // nb_inplace_floor_divide;
    nullptr, // nb_inplace_true_divide;

    nullptr, // nb_index;

    nullptr, //nb_matrix_multiply;
    nullptr  //nb_inplace_matrix_multiply;

};

PyObject *PyMatrix_data(PyObject *self, void *closure)
{

    PyMatrixObject *obj = (PyMatrixObject *)self;
    Py_ssize_t width = obj->matrix.cols();
    Py_ssize_t height = obj->matrix.rows();

    PyObject *list = PyList_New(height);
    for (int i = 0; i < height; i++)
    {
        PyObject *internal = PyList_New(width);

        for (int j = 0; j < width; j++)
        {
            PyObject *value = PyFloat_FromDouble(obj->matrix(i, j));
            PyList_SetItem(internal, j, value);
        }

        PyList_SetItem(list, i, internal);
    }
    return list;
}

PyObject *PyMatrix_rows(PyObject *self, void *closure)
{
    PyMatrixObject *obj = (PyMatrixObject *)self;
    return Py_BuildValue("i", obj->matrix.rows());
}

PyObject *PyMatrix_cols(PyObject *self, void *closure)
{
    PyMatrixObject *obj = (PyMatrixObject *)self;
    return Py_BuildValue("i", obj->matrix.cols());
}

static PyGetSetDef MatrixGetSet[] = {
    {"data", (getter)PyMatrix_data, nullptr, nullptr},
    {"row", (getter)PyMatrix_rows, nullptr, nullptr},
    {"colunm", (getter)PyMatrix_cols, nullptr, nullptr},
    {nullptr}};

PyObject *PyMatrix_tolist(PyObject *self, PyObject *args)
{
    return PyMatrix_data(self, nullptr);
}

static PyMethodDef MatrixMethods[] = {
    {"to_list", (PyCFunction)PyMatrix_tolist, METH_VARARGS, "Return the matrix data to a list object."},
    {nullptr}};

static PyTypeObject MatrixType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "matrix.Matrix", /* tp_name */
    sizeof(PyMatrixObject),                            /* tp_basicsize */
    0,                                                 /* tp_itemsize */
    nullptr,                                           /* tp_dealloc */
    nullptr,                                           /* tp_print */
    nullptr,                                           /* tp_getattr */
    nullptr,                                           /* tp_setattr */
    nullptr,                                           /* tp_reserved */
    nullptr,                                           /* tp_repr */
    &numberMethods,                                    /* tp_as_number */
    nullptr,                                           /* tp_as_sequence */
    nullptr,                                           /* tp_as_mapping */
    nullptr,                                           /* tp_hash  */
    nullptr,                                           /* tp_call */
    PyMatrix_str,                                      /* tp_str */
    nullptr,                                           /* tp_getattro */
    nullptr,                                           /* tp_setattro */
    nullptr,                                           /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,          /* tp_flags */
    "Coustom matrix class.",                           /* tp_doc */
    nullptr,                                           /* tp_traverse */
    nullptr,                                           /* tp_clear */
    nullptr,                                           /* tp_richcompare */
    0,                                                 /* tp_weaklistoffset */
    nullptr,                                           /* tp_iter */
    nullptr,                                           /* tp_iternext */
    MatrixMethods,                                     /* tp_methods */
    nullptr,                                           /* tp_members */
    MatrixGetSet,                                      /* tp_getset */
    nullptr,                                           /* tp_base */
    nullptr,                                           /* tp_dict */
    nullptr,                                           /* tp_descr_get */
    nullptr,                                           /* tp_descr_set */
    0,                                                 /* tp_dictoffset */
    nullptr,                                           /* tp_init */
    nullptr,                                           /* tp_alloc */
    PyMatrix_new                                       /* tp_new */
};

static PyObject *PyMatrix_ones(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyMatrixObject *m = (PyMatrixObject *)PyMatrix_new(&MatrixType, args, kwargs);
    m->matrix.setOnes();
    return (PyObject *)m;
}

static PyObject *PyMatrix_zeros(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyMatrixObject *m = (PyMatrixObject *)PyMatrix_new(&MatrixType, args, kwargs);
    m->matrix.setZero();
    return (PyObject *)m;
}

static PyObject *PyMatrix_random(PyObject *self, PyObject *args, PyObject *kwargs)
{
    PyMatrixObject *m = (PyMatrixObject *)PyMatrix_new(&MatrixType, args, kwargs);
    m->matrix.setRandom();
    return (PyObject *)m;
}

static PyObject *PyMatrix_matrix(PyObject *self, PyObject *args)
{
    PyObject *data = nullptr;
    if (!PyArg_ParseTuple(args, "O", &data))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 1");
        return nullptr;
    }
    if (!PyList_Check(data))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 2");
        return nullptr;
    }
    int height = PyList_GET_SIZE(data);
    if (height <= 0)
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 2");
        return nullptr;
    }
    PyObject *list = PyList_GET_ITEM(data, 0);
    if (!PyList_Check(list))
    {
        PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 3");
        return nullptr;
    }
    int width = PyList_GET_SIZE(list);
    MatrixXd p_mat(width, height);
    for (int i = 0; i < height; i++)
    {
        PyObject *list = PyList_GET_ITEM(data, i);
        if (!PyList_Check(list))
        {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. 3");
            return nullptr;
        }
        int tmp = PyList_GET_SIZE(list);
        if (width != tmp)
        {
            PyErr_SetString(PyExc_ValueError, "Please pass a 2 dimensions list object. Each elements of it must be the same length.");
            return nullptr;
        }
        width = tmp;

        for (int j = 0; j < width; j++)
        {
            PyObject *num = PyList_GET_ITEM(list, j);
            if (!PyFloat_Check(num))
            {
                PyErr_SetString(PyExc_ValueError, "Every elements of the matrix must float.");
                return nullptr;
            }
            p_mat(i, j) = ((PyFloatObject *)num)->ob_fval;
        }
    }

    RETURN_MATRIX(p_mat, &MatrixType);
}

static PyMethodDef matrixMethods[] = {
    {"ones", (PyCFunction)PyMatrix_ones, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values one."},
    {"zeros", (PyCFunction)PyMatrix_zeros, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with initial values zero."},
    {"random", (PyCFunction)PyMatrix_random, METH_VARARGS | METH_KEYWORDS, "Return a new matrix with random values"},
    {"matrix", (PyCFunction)PyMatrix_matrix, METH_VARARGS, "Return a new matrix with given values"},
    {nullptr}};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "matrix",
    "Python interface for Matrix calculation",
    -1,
    matrixMethods};

PyObject *initModule(void)
{
    PyObject *m;
    if (PyType_Ready(&MatrixType) < 0)
        return NULL;

    m = PyModule_Create(&module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MatrixType);
    if (PyModule_AddObject(m, "Matrix", (PyObject *)&MatrixType) < 0)
    {
        Py_DECREF(&MatrixType);
        Py_DECREF(m);
        return NULL;
    }

    return m;
}