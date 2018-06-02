import random, math
import matplotlib.pyplot as pp
import numpy as np

def target_func(x): return math.cos(x / 2) + math.sin(5 / (math.fabs(x) + 0.2)) - 0.1 * x
def fermi(x): return (1+math.exp(-x))**(-1)
def fermi_dx(x): return fermi(x) * (1 - fermi(x))

func_x = np.arange(-10, 10, 0.02)
func_y = np.empty(func_x.shape)
for i in range(len(func_x)): func_y[i] = target_func(func_x[i])

weight_input = [random.uniform(-0.5, 0.5) for i in range(10)]
weight_bias = [random.uniform(-0.5, 0.5) for i in range(10)]
weight_output = [random.uniform(-0.5, 0.5) for i in range(10)]
weight_outbias = random.uniform(-0.5, 0.5)

while True:
    for i in range(100):
        for j in range(5000):
            x = random.choice(func_x)
            y = target_func(x)
            output = [fermi(wb + x * wi) for wb, wi in zip(weight_bias,weight_input)]
            output_dx = [fermi_dx(wb + x * wi) for wb, wi in zip(weight_bias,weight_input)]
            m = np.dot(output, weight_output)+weight_outbias

            for n in range(10):
                weight_input[n]  = weight_input[n]  - 0.01 * (m - y) * weight_output[n] * output_dx[n] * x
                weight_bias[n]   = weight_bias[n]   - 0.01 * (m - y) * weight_output[n] * output_dx[n] * 1
                weight_output[n] = weight_output[n] - 0.01 * (m - y) * output[n]
            weight_outbias = weight_outbias - 0.01 * (m - y)
        sq_error = 0
        func_mlp = []
        for x in func_x:
            output = [fermi(wb + x * wi) for wb, wi in zip(weight_bias,weight_input)]
            m = np.dot(output, weight_output)+weight_outbias
            sq_error += (m - target_func(x)) ** 2
            func_mlp.append(m)
        print(str(i)+":"+str(sq_error / 1001), end="\r")
    pp.plot(func_x, func_y)
    pp.plot(func_x, func_mlp)
    pp.show()
    input("Press Enter to continue...")
