
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt

def showData(dataset):
    # Load dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    #dataset = pandas.read_csv(url, names=names)

    # shape
    print("\nshape")
    print(dataset.shape)

    # head
    print("\nhead 20")
    print(dataset.head(20))

    # descriptions
    print("\ndescribe")
    print(dataset.describe())

    # class distribution
    print("")
    print("groupby('class').size()")
    print(dataset.groupby('class').size())

    # box and whisker plots
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.show()

    # histograms
    dataset.hist()
    plt.show()

    # scatter plot matrix
    pandas.plotting.scatter_matrix(dataset)
    plt.show()