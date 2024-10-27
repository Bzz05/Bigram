In backpropagation, we use the chain rule to calculate derivatives as we move backward through layers. The chain rule states:

\[
\frac{dx}{dy} = \frac{dx}{dz} \cdot \frac{dz}{dy}
\]

During back propagation, global derivative * local derivative:

\[
\frac{da}{db} = \frac{da}{dz} \cdot \frac{dz}{dc} \cdot \frac{dc}{dt} \cdot \frac{dt}{db}
\]

with \(\frac{da}{dz}\), being the node before the output, and it iterates back to layers before it.

## Gradient Flow for Basic Operations

for addition, you simply pass back the gradient from \(i\)th layer to \(i-1\)th layer, since they are linear operations. this means any change in \(d\) is equivalent to the change in both \(a\) and \(b\). Take this for example:

\[
d = a + b
\]

Both \(\frac{\partial d}{\partial a}\) and \(\frac{\partial d}{\partial b}\) are simply:

\[
\frac{\partial d}{\partial a} = 1 \quad \text{and} \quad \frac{\partial d}{\partial b} = 1
\]

so:

\[
\text{grad}_a = \text{grad}_d \cdot 1 = \text{grad}_d
\]
\[\text{grad}_b = \text{grad}_d \cdot 1 = \text{grad}_d\]


for subtraction, you make the gradient with respect to the term with a minus sign negative:

\[
d = a - b
\]

so:

\[
\frac{\partial d}{\partial a} = 1 \quad \text{and} \quad \frac{\partial d}{\partial b} = -1
\]

which gives:

\[
\text{grad}_a = \text{grad}_d \cdot 1 = \text{grad}_d
\]
\[\text{grad}_b = \text{grad}_d \cdot (-1) = -\text{grad}_d\]

for multiplication:

\[
d = a \cdot b
\]

we have:

\[
\frac{\partial d}{\partial a} = b \quad \text{and} \quad \frac{\partial d}{\partial b} = a
\]

so:

\[
\text{grad}_a = \text{grad}_d \cdot b
\]
\[\text{grad}_b = \text{grad}_d \cdot a\]

---

this is expected to not make a complete word, because bigram is kind of a weak model.
