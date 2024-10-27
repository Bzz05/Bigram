in backpropagation, we use the chain rule to calculate derivatives as we move backward through layers. 

chain rule:

dx/dz = dx/dy * dy/dz

during backpropagation it went something like this:

da/db = da/dz * dz/dc * dc/dt * dt/db

with da/dz being the node before the output, and it iterates back to layers before it.


for addition, you simply pass back the gradient from the \(i\)th layer to the \(i-1\)th layer, since they are linear operations. this means any change in d is equivalent to the change in both a and b. take this for example:

d = a + b

both dd/da and dd/db are simply:

dd/da = dd/db = 1
so:

grad(a) = grad(d) * 1 and grad(b) = grad(d) * 1

for subtraction, make the gradient with respect to the term with a minus sign negative:

d = a - b

so:

dd/da = 1 and dd/db = -1

which gives:

grad(a) = grad(d) * 1 and grad(b) = grad(d) * -1

for multiplication:

d = a * b

we have:
dd/da = b, dd/db = a

so:

grad(a) = grad(d) * b and grad(b) = grad(d) * a

---
## broadcasting
at last, another concept called broadcasting. more clearly see the code where we try to make the code more efficient by making P = N.float() then P /= P.sum(1, keepdims = True) to normalize it. it could be seen that there is 27 x 27 and 27 x 1 division. 

normally, we couldn't do this. so what pytorch do is kind of remake 27 x 1 to 27 x 27 by adding more columns (i like to think that it as being stretched). also, pytorch did it from right to left, and they could be adjusted by torch only if the element wise is of the same size, one of them is 1, or one of them doesnt exist. and we need to be becareful with 27 x 27 / 27 x 1 and 27 x 27 / 1 x 27. this could possibly lead to bug in our case. if we do keepdims = True, the shape would be 27 x 1 which is a 2D tensor and if we don't, it will just be 27, a 1D tensor. so 27 x 27 / 27, and with broadcasting rule, pytorch will change it to 1 x 27 and the equation becomes 27 x 27 / 1 x 27, which then it stretches (it will make it 27 x 27) in which we are actually normalizing the columns, instead of the rows. you can try running P[0].sum() without using keepdims = True where it should've return the sum as 1.0(because of probability), but it actually doesn't.

---

this is expected to not make a complete word, because bigram is kind of a weak model.


## Credits

This work is made with the teachings and materials from Andrej Karpathy.
