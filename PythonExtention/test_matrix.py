import matrix

shape = 10

mat_a = matrix.random(shape, shape)
mat_b = matrix.random(shape, shape)

print(mat_a * mat_b)
print(mat_a + mat_b)
print(mat_a - mat_b)

print(mat_a.to_list())
print(mat_a.data)
print(mat_a.row)
print(mat_a.colunm)



