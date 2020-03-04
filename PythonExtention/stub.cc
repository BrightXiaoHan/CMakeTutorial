#define PY_SSIZE_T_CLEAN
#include <Python.h>

extern PyObject* initModule();

PyMODINIT_FUNC
PyInit_matrix(void)
{
    return initModule();
}