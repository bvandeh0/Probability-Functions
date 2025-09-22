"""
This module provides functions to plot various probability distributions including:
- Binomial Distribution
- Poisson Distribution
- Normal Distribution
- Exponential Distribution
- Geometric Distribution
- Hypergeometric Distribution
- Negative Binomial Distribution

Each function generates a plot representing the probability mass function (PMF) or probability density function (PDF) of the respective distribution. The plots can be closed by pressing the 'Enter' key.

Dependencies:
- numpy
- scipy
- matplotlib

Usage:
Import this module and call the desired plotting function with appropriate parameters to visualize the corresponding probability distribution.
"""

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import sys
from scipy.stats import binom, poisson, norm, expon, geom, hypergeom, uniform, nbinom

def plot_binomial(n, p):
    """
    Plots the binomial distribution probability mass function (PMF) for given parameters.

    Parameters:
    n (int): The number of trials.
    p (float): The probability of success on each trial.

    The function creates a bar plot representing the probability of achieving 
    a certain number of successes in a binomial distribution defined by 
    the parameters n and p.
    """
    
    # Generate an array of possible success counts from 0 to n
    x = np.arange(0, n + 1)
    
    # Calculate the probability mass function values for each success count
    y = binom.pmf(x, n, p)

    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Create a bar plot for the PMF values
    plt.bar(x, y, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Overlay the PMF points with red dots connected by lines
    plt.plot(x, y, 'ro-', markersize=4, label="PMF")
    
    # Set the title and labels for the axes
    plt.title(f"Binomial Distribution (n={n}, p={p})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    
    # Set the x-ticks to show all possible success counts
    plt.xticks(x)
    
    # Add a grid to the y-axis for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Add a legend to the plot
    plt.legend()
    
    # Display the plot
    plt.show()

def plot_poisson(lam, k):
    # Generate an array of possible event counts from 0 to k
    x = np.arange(0, k+1)
    
    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Create a bar plot for the Poisson PMF values
    plt.bar(x, poisson.pmf(x, lam), color='skyblue')
    
    # Set the title and labels for the axes
    plt.title(f"Poisson Distribution (λ={lam})")
    plt.xlabel("Number of Events")
    plt.ylabel("Probability")
    plt.xticks(x)
    plt.grid()

    # Prompt user to press 'Enter' to close the graph
    print("Press 'Enter' on plot screen to close graph and continue.")
    # Define a function to handle key press events
    def on_key(event):
        # Check if the pressed key is 'enter'
        if event.key == 'enter':
            plt.close()  # Close the plot when 'Enter' is pressed
            return

    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)

    # Display the plot
    plt.show()

def plot_normal(mu, sigma):
    """
    Plots the normal distribution based on the given mean (mu) and standard deviation (sigma).
        
    Parameters:
    mu (float): The mean of the normal distribution.
    sigma (float): The standard deviation of the normal distribution.
    """
    # Generate an array of values for the normal distribution
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Plot the normal distribution PDF
    plt.plot(x, norm.pdf(x, mu, sigma), color='skyblue')
    
    # Set the title and labels for the axes
    plt.title(f"Normal Distribution (μ={mu}, σ={sigma})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    plt.grid()

    # Prompt user to press 'Enter' to close the graph
    print("Press 'Enter' on plot screen to close graph and continue.")
    
    # Define a function to handle key press events
    def on_key(event):
        if event.key == 'enter':
            plt.close()  # Close the plot when 'Enter' is pressed
            return
            
    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)
    
    # Display the plot
    plt.show()

def plot_exponential(lam):
    """
    Plots an exponential distribution based on the provided lambda parameter.

    Parameters:
    lam (float): The rate parameter (lambda) of the exponential distribution.

    This function generates a plot of the exponential distribution and waits for the user
    to press 'Enter' to close the plot window.
    """
    # Generate an exponential distribution plot
    print("Press 'Enter' on plot screen to close graph and continue.")
    
    def on_key(event):
        # Close the plot when 'Enter' key is pressed
        if event.key == 'enter':
            plt.close()
            return
            
    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)
    # Display the plot
    plt.show()

def plot_geometric(p):
    """
    Plots the geometric distribution for a given probability of success.

    Parameters:
    p (float): The probability of success on each trial (0 < p < 1).

    The function creates a bar plot representing the probability mass function (PMF)
    of the geometric distribution for the given probability of success. The plot will
    display the number of trials on the x-axis and the corresponding probabilities on
    the y-axis. The plot can be closed by pressing 'Enter'.
    """
    
    # Generate an array of possible trial counts from 1 to 10
    x = np.arange(1, 11)
    
    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Plot the PMF as a bar chart
    plt.bar(x, geom.pmf(x, p), color='skyblue')
    
    # Set the title and labels for the plot
    plt.title(f"Geometric Distribution (p={p})")
    plt.xlabel("Number of Trials")
    plt.ylabel("Probability")
    
    # Set the x-ticks to show all possible trial counts
    plt.xticks(x)
    
    # Add a grid to the plot for better readability
    plt.grid()

    print("Press 'Enter' on plot screen to close graph and continue.")
    
    # Define a function to handle key press events
    def on_key(event):
        if event.key == 'enter':
            plt.close()  # Close the plot when 'Enter' is pressed
            return
            
    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)
    
    # Display the plot
    plt.show()
    
def plot_hypergeometric(N, K, n):
    """
    Plots the hypergeometric distribution.

    Parameters:
    N (int): The total number of items.
    K (int): The total number of successful items.
    n (int): The number of draws.

    The function creates a bar plot representing the probability mass function (PMF)
    of the hypergeometric distribution for the given parameters. The plot will display
    the number of successes on the x-axis and the corresponding probabilities on the y-axis.
    The plot can be closed by pressing 'Enter'.
    """
    
    # Calculate the range of x values for the number of successes
    x = np.arange(max(0, n-K), min(n, K)+1)
    
    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Plot the PMF as a bar chart
    plt.bar(x, hypergeom.pmf(x, N, K, n), color='skyblue')
    
    # Set the title and labels for the plot
    plt.title(f"Hypergeometric Distribution (N={N}, K={K}, n={n})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    
    # Set the x-ticks to the values of x
    plt.xticks(x)
    
    # Add a grid to the plot for better readability
    plt.grid()

    print("Press 'Enter' on plot screen to close graph and continue.")
    
    # Define a function to handle key press events
    def on_key(event):
        if event.key == 'enter':
            plt.close()  # Close the plot when 'Enter' is pressed
            return
            
    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)
    
    # Display the plot
    plt.show()

def plot_negative_binomial(p, r):
    """
    Plots the Negative Binomial Distribution for given parameters.

    Parameters:
    p (float): The probability of success on each trial (0 < p < 1).
    r (int): The number of successes until the experiment is stopped (r > 0).

    The function creates a bar plot representing the probability mass function (PMF)
    of the Negative Binomial Distribution and waits for the user to press 'Enter'
    to close the plot.
    """
    
    x = np.arange(0, 11)  # Define the range of successes to plot
    plt.figure(figsize=(8, 5))  # Set the figure size for the plot
    plt.bar(x, nbinom.pmf(x, r, p), color='skyblue')  # Create a bar plot of the PMF
    plt.title(f"Negative Binomial Distribution (r={r}, p={p})")  # Set the title of the plot
    plt.xlabel("Number of Successes")  # Label for the x-axis
    plt.ylabel("Probability")  # Label for the y-axis
    plt.xticks(x)  # Set the ticks on the x-axis
    plt.grid()  # Enable grid on the plot

    print("Press 'Enter' on plot screen to close graph and continue.")
    
    def on_key(event):
        if event.key == 'enter':  # Check if the 'Enter' key is pressed
            plt.close()  # Close the plot
            return
    plt.connect('key_press_event', on_key)  # Connect the key press event to the function
    plt.show()  # Display the plot

    
def plot_negative_binomial(p, r):
    x = np.arange(0, 11)
    plt.figure(figsize=(8, 5))
    plt.bar(x, nbinom.pmf(x, r, p), color='skyblue')
    plt.title(f"Negative Binomial Distribution (r={r}, p={p})")
    plt.xlabel("Number of Successes")
    plt.ylabel("Probability")
    plt.xticks(x)
    plt.grid()

    print("Press 'Enter' on plot screen to close graph and continue.")
    
    def on_key(event):
        if event.key == 'enter':
            plt.close()
            return
    plt.connect('key_press_event', on_key)
    plt.show()

def plot_uniform(a, b):
    """
    Plots the uniform distribution based on the provided parameters.

    Parameters:
    a (float): The lower bound of the uniform distribution.
    b (float): The upper bound of the uniform distribution.

    This function generates a plot of the probability density function (PDF) 
    for a uniform distribution defined by the parameters a and b. The plot 
    will display the probability density on the y-axis and the values on the x-axis.
    The plot can be closed by pressing 'Enter'.
    """
    
    # Generate an array of 1000 evenly spaced values between a and b
    x = np.linspace(a, b, 1000)
    
    # Create a figure for the plot
    plt.figure(figsize=(8, 5))
    
    # Plot the PDF of the uniform distribution
    plt.plot(x, uniform.pdf(x, a, b), color='skyblue')
    
    # Set the title and labels for the axes
    plt.title(f"Uniform Distribution (a={a}, b={b})")
    plt.xlabel("Value")
    plt.ylabel("Probability Density")
    
    # Add a grid to the plot for better readability
    plt.grid()

    # Prompt user to press 'Enter' to close the graph
    print("Press 'Enter' on plot screen to close graph and continue.")
    
    # Define a function to handle key press events
    def on_key(event):
        # Close the plot if 'Enter' is pressed
        if event.key == 'enter':
            plt.close()
            return
    
    # Connect the key press event to the on_key function
    plt.connect('key_press_event', on_key)
    
    # Display the plot
    plt.show()


def binomial():
    """
    Calculate and display the probabilities associated with a binomial distribution.

    This function prompts the user for the number of trials, the probability of success,
    and the number of successes. It then calculates the probability of getting exactly,
    less than or equal to, or greater than or equal to the specified number of successes.
    Additionally, it provides the mean, variance, and skew of the distribution and offers
    to display a graph of the distribution.

    Returns:
        None
    """
    
    # Prompt user for the number of trials
    n = input("Enter the number of trials or press enter to return to the main menu: ")
    if n == '':
        main()  # Return to main menu if no input is provided
    n = int(n)  # Convert input to integer

    while True:
        # Prompt user for the probability of success
        p = float(input("Enter the probability of success: "))
        if p < 0 or p > 1:
            print("Invalid probability. Please enter a value between 0 and 1.")
        else:
            while True:
                # Prompt user for the number of successes
                k = int(input("Enter the number of successes: "))
                if k < 0 or k > n:
                    print(f"Invalid number of successes. Please enter a value between 0 and {n}.")
                else:
                    break  # Valid input for number of successes
            break  # Valid input for probability of success

    # Prompt user for the type of probability calculation
    option = input('''Would you like the probability of getting exactly k successes, the probability of getting less than or equal to k successes, or \
                   the probability of getting greater than or equal to k successes? (Enter '>=', '=', or '<='): ''')
    
    # Calculate and display the requested probability
    if(option == '='):
        print("\nThe probability of getting exactly", k, "successes in", n, "trials is", binom.pmf(k, n, p))
    elif(option == '<='):
        print("\nThe probability of getting less than or equal to", k, "successes in", n, "trials is", binom.cdf(k, n, p))
    elif(option == '>='):
        print("\nThe probability of getting greater than or equal to", k, "successes in", n, "trials is", 1 - binom.cdf(k-1, n, p))
    else:
        print("\nInvalid choice. Please try again.\n\n")
        binomial()  # Retry if the input is invalid

    # Calculate mean, variance, and skew of the binomial distribution
    mean, var, skew = binom.stats(n, p, moments='mvs')
    
    # Display the calculated statistics
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")
    
    # Ask user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")
    if graph == 'y':
        plot_binomial(n, p)  # Call function to plot the distribution
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    
    main()  # Return to main menu
    
def pois():
    """
    Calculate and display probabilities for a Poisson distribution based on user input.

    The function prompts the user for the average number of events (lambda), the number of events (k),
    and an optional time period to adjust lambda. It then calculates the probability of getting exactly k
    successes, less than or equal to k successes, or greater than or equal to k successes based on user choice.
    Additionally, it provides the mean, variance, and skew of the distribution and offers to display a graph.

    Returns:
        None
    """
    
    # Prompt user for the average number of events (lambda)
    lam = (input("Enter the average number of events or press enter to return to main menu: "))
    if lam == '':
        main()  # Return to main menu if input is empty
    lam = float(lam)  # Convert lambda to float

    while True:
        try:
            # Prompt user for the number of events (k)
            k = int(input("Enter the number of events: "))
            if k < 0:
                print(f"Invalid number of events. Please enter a value greater than 0.")
            else:  
                break  # Exit loop if valid input is provided
        except ValueError:
            print("Invalid input. Please enter an integer.")  # Handle non-integer input

    # Prompt user for an optional time period to adjust lambda
    t = input("Enter the time period (optional, press enter to skip): ")
    if t != '':
        try:
            lam *= float(t)  # Adjust lambda based on time period
            print(f"Adjusted lambda for time period {t} is {lam}")
        except ValueError:
            print("Invalid time period. Skipping adjustment.")  # Handle invalid time period input

    # Prompt user for the type of probability calculation
    option = input('''Would you like the probability of getting exactly k successes, the probability of getting less than or equal to k successes, or \
                   the probability of getting greater than or equal to k successes? (Enter '>=', '=', or '<='): ''')
    if(option == '='):
        print("\nThe probability of getting exactly", k, "successes is", poisson.pmf(k, lam))
    elif(option == '<='):
        print("\nThe probability of getting less than or equal to", k, "successes is", poisson.cdf(k, lam))
    elif(option == '>='):
        print("\nThe probability of getting greater than or equal to", k, "successes is", 1 - poisson.cdf(k-1, lam))
    else:
        print("\nInvalid choice. Please try again.\n\n")
        pois()  # Retry if the input is invalid

    # Calculate and display mean, variance, and skew of the distribution
    mean, var, skew = poisson.stats(lam, moments='mvs')
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")
    
    # Prompt user to display a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")
    if graph == 'y':
        plot_poisson(lam, k)  # Call function to plot the distribution
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    main()  # Return to main menu


def normal():
    """
    Calculate and display the probability density function for a normal distribution.
    
    This function prompts the user for the mean (mu), standard deviation (sigma), 
    and a value (x) to calculate the probability of getting exactly x, 
    less than or equal to x, or greater than or equal to x. 
    It also provides the mean, variance, and skew of the distribution, 
    and offers to display a graph of the distribution.

    Returns:
        None
    """
    
    # Prompt user for the mean and handle the case where they want to return to the main menu
    mu = input("Enter the mean or press enter to return to main menu: ")
    if mu == '':
        main()  # Return to main menu if no input is provided
    mu = float(mu)  # Convert mean to float

    # Prompt user for the standard deviation
    sigma = float(input("Enter the standard deviation: "))
    
    # Prompt user for the value of x
    x = float(input("Enter the value of x: "))

    # Ask user for the type of probability calculation they want
    option = input('''Would you like the probability of getting exactly x, the probability of getting less than or equal to x, or \
                   the probability of getting greater than or equal to x? (Enter '>=', '=', or '<='): ''')
    
    # Calculate and display the requested probability
    if(option == '='):
        print("\nThe probability of getting exactly", x, "is", norm.pdf(x, mu, sigma))
    elif(option == '<='):
        print("\nThe probability of getting less than or equal to", x, "is", norm.cdf(x, mu, sigma))
    elif(option == '>='):
        print("\nThe probability of getting greater than or equal to", x, "is", 1 - norm.cdf(x, mu, sigma))
    else:
        print("\nInvalid choice. Please try again.\n\n")
        normal()  # Retry if the input is invalid

    # Calculate mean, variance, and skew of the normal distribution
    mean, var, skew = norm.stats(mu, sigma, moments='mvs')

    # Display the calculated statistics
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")
    if graph == 'y':
        plot_normal(mu, sigma)  # Call function to plot the normal distribution
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    
    main()  # Return to main menu

def exponential():
    """
    Calculate and display the cumulative distribution function (CDF) of an exponential distribution
    based on a user-provided rate parameter (lambda) and a value of x. It also provides the mean,
    variance, and skewness of the distribution, and offers an option to display a graph of the distribution.

    The function prompts the user for input and handles cases where the user may want to return to the main menu.
    """
    from scipy.stats import expon 

    # Prompt the user for the rate parameter (lambda)
    lam = (input("Enter the rate parameter or press enter to return to main menu: "))
    if lam == '':
        main()  # Return to the main menu if no input is provided

    lam = float(lam)  # Convert the input to a float
    x = float(input("Enter the value of x: "))  # Prompt for the value of x

    # Calculate and print the cumulative distribution function (CDF) for the given x
    print("The probability of getting a value less than", x, "is", stats.expon.cdf(x, scale=1/lam))

    # Calculate the mean, variance, and skewness of the exponential distribution
    mean, var, skew = expon.stats(scale=1/lam, moments='mvs')
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask the user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")

    if graph == 'y':
        plot_exponential(lam)  # Call the function to plot the exponential distribution
    elif graph != 'n':
        print("Invalid choice. Please try again.")  # Handle invalid input for graph option

    main()  # Return to the main menu

def geometric():
    """
    Calculate and display the probability of success on the k-th trial 
    of a geometric distribution, along with its mean, variance, and skew.
    
    The function prompts the user for the probability of success and 
    the number of trials, then computes and displays the probability 
    of success on the specified trial. It also provides the option 
    to visualize the distribution graphically.
    """
    
    # Prompt the user for the probability of success
    p = (input("Enter the probability of success or press enter to return to main menu: "))
    if p == '':
        main()  # Return to the main menu if no input is provided

    p = float(p)  # Convert the input probability to a float
    k = int(input("Enter the number of trials: "))  # Prompt for the number of trials

    # Calculate and display the probability of getting a success on the k-th trial
    print("The probability of getting a success on the", k, "th trial is", stats.geom.pmf(k, p))

    # Calculate mean, variance, and skew of the geometric distribution
    mean, var, skew = geom.stats(p, moments='mvs')

    # Display the calculated statistics
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask the user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")

    if graph == 'y':
        plot_geometric(p)  # Call the function to plot the geometric distribution
    elif graph != 'n':
        print("Invalid choice. Please try again.")  # Handle invalid input
    main()  # Return to the main menu

def hypergeometric():
    """
    Calculate and display the probability of getting a certain number of successes
    in a hypergeometric distribution, along with its mean, variance, and skewness.
    
    The function prompts the user for the total number of items, the total number of 
    successes, the number of items drawn, and the number of successes. It then calculates 
    the probability mass function (PMF) for the hypergeometric distribution and displays 
    the mean, variance, and skewness. Optionally, it can also display a graph of the 
    distribution.
    """
    
    # Prompt the user for the total number of items
    N = (input("Enter the total number of items or press enter to return to main menu: "))
    if N == '':
        main()  # Return to the main menu if no input is provided
    N = int(N)  # Convert input to integer

    # Prompt the user for the total number of successes
    K = int(input("Enter the total number of successes: "))
    
    # Prompt the user for the number of items drawn
    n = int(input("Enter the number of items drawn: "))
    
    # Prompt the user for the number of successes
    k = int(input("Enter the number of successes: "))
    
    # Calculate and display the probability of getting k successes in n draws
    print("The probability of getting", k, "successes in", n, "draws is", stats.hypergeom.pmf(k, N, K, n))

    # Calculate the mean, variance, and skewness of the distribution
    mean, var, skew = stats.hypergeom.stats(N, K, n, moments='mvs')

    # Display the calculated mean, variance, and skewness
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask the user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")

    # If the user wants to see the graph, call the plotting function
    if graph == 'y':
        plot_hypergeometric(N, K, n)
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    
    # Return to the main menu
    main()

def negative_binomial():
    """
    Calculate and display the probability of getting a certain number of successes
    before a specified number of failures in a negative binomial distribution.
    
    The function prompts the user for the probability of success, the number of successes,
    and the number of failures. It then computes and displays the probability mass function (PMF)
    for the negative binomial distribution, along with its mean, variance, and skewness.
    Optionally, it can also display a graph of the distribution.
    """
    
    # Prompt the user for the probability of success
    p = (input("Enter the probability of success or press enter to return to main menu: "))
    if p == '':
        main()  # Return to the main menu if the user presses enter
    p = float(p)  # Convert the input probability to a float
    
    # Prompt the user for the number of successes
    k = int(input("Enter the number of successes: "))
    
    # Prompt the user for the number of failures
    r = int(input("Enter the number of failures: "))
    
    # Calculate and display the probability of getting k successes before r failures
    print("The probability of getting", k, "successes before", r, "failures is", stats.nbinom.pmf(k, r, p))

    # Calculate the mean, variance, and skewness of the negative binomial distribution
    mean, var, skew = stats.nbinom.stats(r, p, moments='mvs')

    # Display the calculated mean, variance, and skewness
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask the user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")

    # If the user wants to see the graph, call the plotting function
    if graph == 'y':
        plot_negative_binomial(p, r)
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    
    # Return to the main menu
    main()

def unif():
    """
    Calculate and display the cumulative distribution function (CDF) for a uniform distribution
    given lower and upper bounds, and a specific value of x. It also computes and displays the 
    mean, variance, and skew of the distribution. Optionally, it can display a graph of the 
    distribution.

    The function prompts the user for input and handles invalid choices by returning to the main menu.
    """
    a = (input("Enter the lower bound or press enter to return to main menu: "))  # Prompt for lower bound
    if a == 0.000:  # Check if the user wants to return to the main menu
        main()
    a = float(a)  # Convert lower bound to float
    b = float(input("Enter the upper bound: "))  # Prompt for upper bound
    x = float(input("Enter the value of x: "))  # Prompt for value of x
    print("The probability of getting a value less than", x, "is", stats.uniform.cdf(x, a, b))  # Calculate and display CDF

    mean, var, skew = stats.uniform.stats(loc=a, scale=b-a, moments='mvs')  # Calculate mean, variance, and skew

    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")  # Display statistics

    graph = input("Would you like to see a graph of the distribution? (y/n): ")  # Prompt for graph display

    if graph == 'y':  # If user wants to see the graph
        plot_uniform(a, b)  # Call function to plot the uniform distribution
    elif graph != 'n':  # If the input is invalid
        print("Invalid choice. Please try again.")  # Inform user of invalid choice
    main()  # Return to main menu

def unif():
    """
    Calculate the cumulative distribution function (CDF) for a uniform distribution
    and display its mean, variance, and skewness. Optionally, plot the distribution.

    The function prompts the user for the lower and upper bounds of the uniform distribution,
    as well as a value of x to calculate the probability of getting a value less than x.
    It also provides an option to display a graph of the distribution.

    Returns:
        None
    """
    
    # Prompt user for the lower bound
    a = (input("Enter the lower bound or press enter to return to main menu: "))
    
    # Check if the user wants to return to the main menu
    if a == 0.000:
        main()
    
    # Convert lower bound to float
    a = float(a)
    
    # Prompt user for the upper bound
    b = float(input("Enter the upper bound: "))
    
    # Prompt user for the value of x
    x = float(input("Enter the value of x: "))
    
    # Calculate and print the probability of getting a value less than x
    print("The probability of getting a value less than", x, "is", stats.uniform.cdf(x, a, b))

    # Calculate mean, variance, and skewness of the uniform distribution
    mean, var, skew = stats.uniform.stats(loc=a, scale=b-a, moments='mvs')

    # Print the mean, variance, and skewness
    print(f"\nThe mean, variance, and skew of the data is {mean}, {var}, and {skew}\n")

    # Ask user if they want to see a graph of the distribution
    graph = input("Would you like to see a graph of the distribution? (y/n): ")

    # If user wants to see the graph, call the plotting function
    if graph == 'y':
        plot_uniform(a, b)
    # Handle invalid input
    elif graph != 'n':
        print("Invalid choice. Please try again.")
    
    # Return to the main menu
    main()

def main():
    """Main function to display the probability calculator menu and handle user input.

    This function presents a list of probability functions to the user, 
    accepts their choice, and calls the corresponding function based on the input.
    The user can also exit the program by typing 'exit'.
    """
    # Display welcome message and options to the user
    print('''\n\nWelcome to the Probability Calculator! Please select one of the following probability functions:\n
          1. Binomial Distribution\n
          2. Poisson Distribution\n
          3. Normal Distribution\n
          4. Exponential Distribution\n
          5. Geometric Distribution\n
          6. Hypergeometric Distribution\n
          7. Negative Binomial Distribution\n
          8. Uniform Distribution\n
          Type 'exit' to quit the program.\n''')
    
    # Get user input for the desired probability function
    choice = input("Enter the number of the function you would like to use: ")
    
    # Call the appropriate function based on user choice
    if choice == "1":
        binomial()
    elif choice == "2":
        pois()
    elif choice == "3":
        normal()
    elif choice == "4":
        exponential()
    elif choice == "5":
        geometric()
    elif choice == "6":
        hypergeometric()
    elif choice == "7":
        negative_binomial()
    elif choice == "8":
        unif()
    elif choice == "exit":
        # Exit the program
        print("Goodbye!")
        sys.exit()
    else:
        # Handle invalid input
        print("Invalid choice. Please try again.")
    
    # Recursively call main to display the menu again
    main()
        



if __name__ == "__main__":
    main()