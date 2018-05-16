import numpy
import matplotlib.pyplot as plt
import scipy.optimize as optimization

# Generated data: y = x plus some noise
input_data = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
output_data = numpy.array([0.1, 0.9, 2.2, 2.8, 3.9, 5.1])



# Initial guess.
intial_guess = numpy.array([0.0, 0.0, 0.0])


def get_output(input_data, a, b, c):
    return a + b * input_data + c * input_data * input_data

optimized_params, cov_matrix = optimization.curve_fit(get_output, input_data, output_data, intial_guess)

resulting_data = get_output(input_data, *optimized_params)

plt.figure()
plt.scatter(input_data, output_data)
plt.plot(input_data, resulting_data)
plt.show()
