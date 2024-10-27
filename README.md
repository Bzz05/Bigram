chain rule:
dx/dy = dx/dz * dz/dy

during back propagation, global derivative * 'local derivative'
just like:
da/db = da/dz * dz/dc * dc/dt * dt/db

with da/dz, i guess being the node before the output, and it iterates back to the input

with + you kind of just pass back the gradient from ith layer to i-1th layer, since they are linear operation, so any change in d is equivalent to change in both a and b
take this for example:
d =  a + b, both dd/da and dd/db is both just dd * 1 = dd (with dd as gradient of d)
for substraction, just make the one with minus sign as (-1)*dd

for multiplication, such as:
d = ab
dd/da = b * dd and dd/db = a * dd, so grad of a = dd * b and grad of b = a * dd

this is expected to not make a complete word, because bigram is kind of a weak model.