import pandas
import math
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from scipy import stats as sp


def import_data():
    # import the data into a pandas dataframe
    data = pandas.read_csv('cricket_data.csv')

    # set x to the chirps per second column
    x = data[['chirps_per_second']]

    # set y to the temperature column
    y = data['temperature']

    return x, y


def plot_data(intercept, coefficient):
    # graph original data as a scatter-plot
    plt.scatter(x, y)

    # set axis labels
    plt.xlabel('chirps_per_second')
    plt.ylabel('temperature')

    # get min and max values
    x_min = math.floor(x.min())
    x_max = math.ceil(x.max()) + 1

    # get y values for best fit line
    best_fit_line = [coefficient * i + intercept for i in range(x_min, x_max)]

    # plot the original x values and the best fit y values
    plt.plot(range(x_min, x_max), best_fit_line, 'r')
    plt.show()


def get_pearson_stats(x,y):
    # convert dataframes to lists
    x_list = list(x.values.flatten())
    y_list = list(y.values.flatten())

    # get pearson results
    pearson_results = sp.pearsonr(x_list, y_list)

    # extract correlation coefficient
    cor_coef = pearson_results[0]

    # extract p-value
    p_value = pearson_results[1]

    return cor_coef, p_value


def print_results():
    # print r^2 score
    print("R^2: " + r2_score.__str__())

    # print coefficient
    print("Coefficient: " + coefficient[0].__str__())

    # print intercept
    print("Intercept: " + intercept.__str__())

    # print correlation oefficient
    print("Correlation Coefficient: " + cor_coef.__str__())

    # print p-value
    print("P-value: " + p_value.__str__())


if __name__ == '__main__':

    # get x and y dataframes
    x, y = import_data()

    # create our linear regression model
    classifier = lm.LinearRegression()

    # fit our model
    results = classifier.fit(x, y)

    # store coefficient
    coefficient = results.coef_

    # store intercept
    intercept = results.intercept_

    # get r^2 score
    r2_score = classifier.score(x, y)

    # get pearson correlation coefficient and p_value
    cor_coef, p_value = get_pearson_stats(x, y)

    # Plot the data and best fit line
    plot_data(intercept, coefficient)

    print_results()