# Symbolic Regression via Student-Teacher Learning
A novel attempt at interpretable function-learning via Student-Teacher Neural Networks.

---

**Table of Contents**
<div id="user-content-toc">
    <ul>
        <li><a href="#1-background">1. Background</a></li>
        <li><a href="#2-this-work">2. This Work</a></li>
    </ul>
</div>

---

## 1. Background
<a href="https://en.wikipedia.org/wiki/Symbolic_regression#:~:text=Symbolic%20regression%20(SR)%20is%20a,terms%20of%20accuracy%20and%20simplicity">Symbolic Regression</a> is a regression analysis method that aims to search for a model that not only fits a particular dataset, but one that is also easily readable as a symbolic regression (such as a mathematical function). Many other works have been completed in the literature that either propose novel solvers to accomplish SR, or those that incoporate novel techniques or toolkits to suppliment SR. In most cases, these solvers incorporate deep learning or otherwise employ neural networks as <a href="https://en.wikipedia.org/wiki/Universal_approximation_theorem">Universal Function Approximators</a>, then attempt to cleverly extract knowledge from the trained NN into a symbolic form. To the knowledge of the authors of this work, no such solvers exist that employ the use of <a href="https://arxiv.org/abs/1503.02531">Student-Teacher</a> networks for knowledge distillation into interpretable networks, which is what this work aims to accomplish.
## 2. This Work
We aim to first construct a "best teacher" for a given dataset that learns a mystery function with sufficiently high accuracy/low loss. Then, we attempt to distill the knowledge of the teacher at each layer to a smaller student network comprised of interpretable modules that coordinate to known operations, or layers of interpretable operations. Ideally, the student network performs as well as the teacher, but provides a simpler (and readable) representation of the teacher network, accomplishing the symbolic extraction of the regression.

This project is ongoing, and should be completed by December of 2022. Until then, active work is being done to generate a successful outcome.