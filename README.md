in backpropagation, we use the chain rule to calculate derivatives as we move backward through layers. The chain rule states:

![Chain Rule](https://latex.codecogs.com/png.latex?%5Cfrac%7Bdx%7D%7Bdy%7D%20%3D%20%5Cfrac%7Bdx%7D%7Bdz%7D%20%5Ccdot%20%5Cfrac%7Bdz%7D%7Bdy%7D)

during backpropagation, global derivative * local derivative:

![Global Derivative](https://latex.codecogs.com/png.latex?%5Cfrac%7Bda%7D%7Bdb%7D%20%3D%20%5Cfrac%7Bda%7D%7Bdz%7D%20%5Ccdot%20%5Cfrac%7Bdz%7D%7Bdc%7D%20%5Ccdot%20%5Cfrac%7Bdc%7D%7Bdt%7D%20%5Ccdot%20%5Cfrac%7Bdt%7D%7Bdb%7D)

with ![\frac{da}{dz}](https://latex.codecogs.com/png.latex?%5Cfrac%7Bda%7D%7Bdz%7D) being the node before the output, and it iterates back to layers before it.


for addition, you simply pass back the gradient from the \(i\)th layer to the \(i-1\)th layer, since they are linear operations. this means any change in \(d\) is equivalent to the change in both \(a\) and \(b\). take this for example:

![d = a + b](https://latex.codecogs.com/png.latex?d%20%3D%20a%20%2B%20b)

both ![\frac{\partial d}{\partial a}](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20a%7D) and ![\frac{\partial d}{\partial b}](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20b%7D) are simply:

![Derivatives](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20a%7D%20%3D%201%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20b%7D%20%3D%201)

so:

![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_a%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%201%20%3D%20%5Ctext%7Bgrad%7D_d)
![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_b%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%201%20%3D%20%5Ctext%7Bgrad%7D_d)

for subtraction, make the gradient with respect to the term with a minus sign negative:

![d = a - b](https://latex.codecogs.com/png.latex?d%20%3D%20a%20-%20b)

so:

![Derivatives](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20a%7D%20%3D%201%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20b%7D%20%3D%20-1)

which gives:

![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_a%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%201%20%3D%20%5Ctext%7Bgrad%7D_d)
![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_b%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%20(-1)%20%3D%20-%5Ctext%7Bgrad%7D_d)

for multiplication:

![d = a \cdot b](https://latex.codecogs.com/png.latex?d%20%3D%20a%20%5Ccdot%20b)

we have:

![Derivatives](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20a%7D%20%3D%20b%20%5Cquad%20%5Ctext%7Band%7D%20%5Cquad%20%5Cfrac%7B%5Cpartial%20d%7D%7B%5Cpartial%20b%7D%20%3D%20a)

so:

![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_a%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%20b) 
![Gradients](https://latex.codecogs.com/png.latex?%5Ctext%7Bgrad%7D_b%20%3D%20%5Ctext%7Bgrad%7D_d%20%5Ccdot%20a)

---
## important
and at last, another concept called broadcasting. more clearly see the code where we try to make the code more efficient by making P = N.float() then P /= P.sum(1, keepdims = True) to normalize it. it could be seen that there is 27 x 27 and 27 x 1 division. 

normally, we couldn't do this. so what pytorch do is kind of remake 27 x 1 to 27 x 27 by adding more columns (i like to think that it as being stretched). also, pytorch did it from right to left, and they could be adjusted by torch only if the element wise is of the same size, one of them is 1, or one of them doesnt exist. and we need to be becareful with 27 x 27 / 27 x 1 and 27 x 27 / 1 x 27. this could possibly lead to bug in our case. if we do keepdims = True, the shape would be 27 x 1 which is a 2D tensor and if we don't, it will just be 27, a 1D tensor. so 27 x 27 / 27, and with broadcasting rule, pytorch will change it to 1 x 27 and the equation becomes 27 x 27 / 1 x 27, which then it stretches (it will make it 27 x 27) in which we are actually normalizing the columns, instead of the rows. you can try running P[0].sum() without using keepdims = True where it should've return the sum as 1.0(because of probability), but it actually doesn't.

---

this is expected to not make a complete word, because bigram is kind of a weak model.


## Credits

This work is made with the teachings and materials from Andrej Karpathy.
