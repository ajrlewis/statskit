import numpy as np
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from typing import Optional


def fit_powerlaw(x: np.array) -> tuple[float, float]:
    """
    Fit a power law distribution to the input data.

    This function fits a power law distribution to the given data using the
    `scipy.stats.powerlaw` module. It returns the estimated parameters of the
    power law distribution.

    Args:
        x (np.array): The input data.

    Returns:
        tuple[float, float]: A tuple containing the estimated parameters of the
        power law distribution. The first element is the alpha parameter, and the
        second element is the x_min parameter.
    """
    fit = stats.powerlaw.fit(x, floc=0)
    alpha, x_min = fit[0], fit[2]
    return alpha, x_min


def random_powerlaw(alpha: float, x_min: float, n_x: int = 100) -> np.array:
    """
    Generate random realizations from a power law distribution.

    This function generates random realizations from a power law distribution
    defined by the given parameters using the `scipy.stats.powerlaw` module.

    Args:
        alpha (float): The alpha parameter of the power law distribution.
        x_min (float): The x_min parameter of the power law distribution.
        n_x (int, optional): The number of random realizations to generate. Defaults to 100.

    Returns:
        np.array: An array containing the generated random realizations from the
        power law distribution.
    """
    random_samples = stats.powerlaw.rvs(alpha, loc=0, scale=x_min, size=n_x)
    return random_samples


def sigma_percentiles(array: np.ndarray) -> tuple[float, float]:
    """
    Calculate the sigma values using percentiles.

    The sigma values represent the spread of data around the median. This method
    calculates the upper and lower sigma values based on the percentiles of the input array.

    Args:
        array (np.ndarray): The input array.

    Returns:
        tuple[float, float]: A tuple containing the upper and lower sigma values.
    """
    array_median = np.median(array)
    sigma_upper = np.percentile(array, 84.135) - array_median
    sigma_lower = array_median - np.percentile(array, 15.865)
    return sigma_upper, sigma_lower


def sigma_mad(array: np.ndarray) -> float:
    """
    Calculate the sigma value using the Median Absolute Deviation (MAD).

    The sigma value represents the spread of data around the median. This method
    calculates the sigma value based on the Median Absolute Deviation of the input array.

    Args:
        array (np.ndarray): The input array.

    Returns:
        float: The calculated sigma value.
    """
    return 1.4826 * np.median(np.abs(array - np.median(array)))


def sigma_gehrels(n: int) -> tuple[float, float]:
    """
    Calculate the sigma values using Gehrels' approximation.

    The sigma values represent the uncertainty in counting experiments. This method
    calculates the upper and lower sigma values based on Gehrels' approximation.

    Args:
        n (int): The count value.

    Returns:
        tuple[float, float]: A tuple containing the upper and lower sigma values.
    """
    integers_to_limits = {
        0: (1.841, 2e-16),
        1: (3.3, 0.173),
        2: (4.638, 0.708),
        3: (5.918, 1.367),
        4: (7.163, 2.086),
        5: (8.382, 2.84),
        6: (9.584, 3.62),
        7: (10.77, 4.419),
        8: (11.95, 5.232),
        9: (13.11, 6.057),
        10: (14.27, 6.891),
        11: (15.42, 7.734),
        12: (16.56, 8.585),
        13: (17.7, 9.441),
        14: (18.83, 10.3),
        15: (19.96, 11.17),
        16: (21.08, 12.04),
        17: (22.2, 12.92),
        18: (23.32, 13.8),
        19: (24.44, 14.68),
        20: (25.55, 15.57),
        21: (26.66, 16.45),
        22: (27.76, 17.35),
        23: (28.87, 18.24),
        24: (29.97, 19.14),
        25: (31.07, 20.03),
        26: (32.16, 20.93),
        27: (33.26, 21.84),
        28: (34.35, 22.74),
        29: (35.45, 23.65),
        30: (36.54, 24.55),
        31: (37.63, 25.46),
        32: (38.72, 26.37),
        33: (39.8, 27.28),
        34: (40.89, 28.2),
        35: (41.97, 29.11),
        36: (43.06, 30.03),
        37: (44.14, 30.94),
        38: (45.22, 31.86),
        39: (46.3, 32.78),
        40: (47.38, 33.7),
        41: (48.46, 34.62),
        42: (49.53, 35.55),
        43: (50.61, 36.47),
        44: (51.68, 37.39),
        45: (52.76, 38.32),
        46: (53.83, 39.24),
        47: (54.9, 40.17),
        48: (55.98, 41.1),
        49: (57.05, 42.02),
        50: (58.12, 42.95),
        60: (68.79, 52.28),
        70: (79.41, 61.65),
        80: (89.98, 71.07),
        90: (100.5, 80.53),
        100: (111.0, 90.02),
    }
    try:
        limits = integers_to_limits[n]
    except KeyError:
        upper_limit = (n + 1.0) + np.sqrt(n + 1.0)
        lower_limit = n * (1.0 - (1.0 / (9.0 * n)) - (1.0 / (3.0 * np.sqrt(n)))) ** 3.0
    else:
        upper_limit, lower_limit = limits
    sigma_upper, sigma_lower = upper_limit - n, n - lower_limit
    return sigma_upper, sigma_lower


def gaussian(x: np.ndarray, x_peak: float, x_width: float) -> np.ndarray:
    """
    Calculate the Gaussian distribution centered at x_peak.

    The Gaussian distribution is a probability distribution that is symmetric and bell-shaped.
    This function calculates the values of the Gaussian distribution at the given x values.

    Args:
        x (np.ndarray): The x values at which to calculate the Gaussian distribution.
        x_peak (float): The center of the Gaussian distribution.
        x_width (float): The width of the Gaussian distribution.

    Returns:
        np.ndarray: The values of the Gaussian distribution at the given x values.
    """
    cyclic_distance = np.mod(x - x_peak + len(x) / 2, len(x)) - len(x) / 2
    return np.exp(-0.5 * (cyclic_distance / x_width) ** 2)


def split_normal(
    x: np.ndarray, x_peak: float, x_upper_width: float, x_lower_width: float
) -> np.ndarray:
    """
    Calculate a split normal distribution centered at x_peak.

    The split normal distribution is a probability distribution that is composed of two halves,
    each with a different width. The upper half has x_upper_width and the lower half has x_lower_width.
    This function calculates the values of the split normal distribution at the given x values.

    Args:
        x (np.ndarray): The x values at which to calculate the split normal distribution.
        x_peak (float): The center of the split normal distribution.
        x_upper_width (float): The width of the upper half of the split normal distribution.
        x_lower_width (float): The width of the lower half of the split normal distribution.

    Returns:
        np.ndarray: The values of the split normal distribution at the given x values.
    """
    cyclic_distance = np.mod(x - x_peak + len(x) / 2, len(x)) - len(x) / 2
    upper_half = np.exp(-0.5 * (cyclic_distance / x_upper_width) ** 2)
    lower_half = np.exp(-0.5 * (cyclic_distance / x_lower_width) ** 2)
    return np.where(cyclic_distance >= 0, upper_half, lower_half)


def random_gaussian(
    mu: np.ndarray,
    sigma: np.ndarray,
    number_mc: int = 500,
    bounds: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random samples from a multivariate normal distribution.

    This method generates random samples from a multivariate normal distribution
    with the specified mean and covariance matrix. The number of samples can be
    controlled using the `number_mc` parameter. If bounds are provided, the samples
    are truncated within the specified bounds.

    Args:
        mu (np.ndarray): The mean vector.
        sigma (np.ndarray): The covariance matrix.
        number_mc (int, optional): The number of samples. Defaults to 500.
        bounds (Optional[tuple[np.ndarray, np.ndarray]], optional): The lower and upper bounds. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the generated samples.
    """
    if bounds is None:
        return np.random.multivariate_normal(mu, sigma, size=number_mc).T
    indices = np.where(sigma == 0.0)[0]
    if indices.size > 0:
        sigma[indices] = np.finfo(float).eps
    lower_bounds, upper_bounds = np.transpose(bounds)
    a, b = (lower_bounds - mu) / sigma, (upper_bounds - mu) / sigma
    x = stats.truncnorm(a, b, loc=mu, scale=sigma)
    return x.rvs((mu.size, number_mc)).T


def random_split_normal(mu: float, sigma_upper: float, sigma_lower: float) -> float:
    """
    Generate random samples from a split normal distribution.

    This method generates random samples from a split normal distribution with the
    specified mean, upper standard deviation, and lower standard deviation.

    Args:
        mu (float): The mean.
        sigma_upper (float): The standard deviation for values greater than the mean.
        sigma_lower (float): The standard deviation for values less than the mean.

    Returns:
        float: A random sample from the split normal distribution.
    """
    gauss = np.random.normal(0.0, 1.0)
    return mu + gauss * (sigma_upper if gauss > 0.0 else sigma_lower)


def gaussian_smooth(array: np.array, sigma: float, mode: str = "wrap") -> np.array:
    """
    Apply Gaussian smoothing to the input data.

    This function applies Gaussian smoothing to the given 1D input data using
    the `scipy.ndimage.gaussian_filter1d` function.

    Args:
        array (np.array): The input data to be smoothed.
        sigma (float): The standard deviation of the Gaussian kernel.
        mode (str): Method to handle edge values.

    Returns:
        np.array: The Gaussian smoothed version of the input data.
    """
    smoothed_data = gaussian_filter1d(array, sigma, mode=mode)
    return smoothed_data


def histogram(
    array: np.ndarray,
    bin_min: Optional[float] = None,
    bin_max: Optional[float] = None,
    bin_size: Optional[float] = None,
    **kwargs
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the histogram of an array.

    The histogram is a graphical representation of the distribution of data. This method
    calculates the histogram of the input array using the specified bin range and size.

    Args:
        array (np.ndarray): The input array.
        bin_min (float, optional): The minimum value of the histogram bins. Defaults to None.
        bin_max (float, optional): The maximum value of the histogram bins. Defaults to None.
        bin_size (float, optional): The size of the histogram bins. Defaults to None.
        **kwargs: Additional keyword arguments to be passed to np.histogram.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the bin centers and frequencies of the histogram.
    """
    bin_min = np.floor(array.min()) if bin_min is None else bin_min
    bin_max = np.ceil(array.max()) if bin_max is None else bin_max
    bin_size = (
        sturges_rule(bin_min, bin_max, array.size) if bin_size is None else bin_size
    )
    bin_edges = np.arange(bin_min, bin_max + bin_size, bin_size)
    bin_centers = bin_edges[:-1] + 0.5 * bin_size
    frequencies, _ = np.histogram(array, bins=bin_edges, **kwargs)
    return bin_centers, frequencies


def sturges_rule(bin_min: float, bin_max: float, number_of_elements: int) -> float:
    """
    Calculate the bin size using Sturges' rule.

    Sturges' rule is a heuristic method for determining the number of bins in a histogram.
    This method calculates the bin size based on the minimum and maximum values and the
    number of elements in the data.

    Args:
        bin_min (float): The minimum value of the histogram bins.
        bin_max (float): The maximum value of the histogram bins.
        number_of_elements (int): The number of elements in the data.

    Returns:
        float: The calculated bin size.
    """
    return (bin_max - bin_min) / (1.0 + 3.322 * np.log10(number_of_elements))


def bhattacharyya_coefficient(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Bhattacharyya coefficient between two probability distributions.

    The Bhattacharyya coefficient measures the similarity between two probability distributions.
    It is defined as the sum of the square root of the element-wise product of the distributions.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The Bhattacharyya coefficient between the distributions.
    """
    return np.sum(np.sqrt(p * q))


def bhattacharyya_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Bhattacharyya distance between two probability distributions.

    The Bhattacharyya distance is a measure of dissimilarity between two probability distributions.
    It is defined as the negative logarithm of the Bhattacharyya coefficient.

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The Bhattacharyya distance between the distributions.
    """
    return -np.log(bhattacharyya_coefficient(p, q))


def euclidean_distance(array_a: np.ndarray, array_b: np.ndarray) -> float:
    """
    Calculate the Euclidean distance between two arrays.

    The Euclidean distance is a measure of the straight-line distance between two points
    in a multidimensional space. This method calculates the Euclidean distance between
    two input arrays.

    Args:
        array_a (np.ndarray): The first input array.
        array_b (np.ndarray): The second input array.

    Returns:
        float: The Euclidean distance between the two arrays.
    """
    r = np.array(array_a) - np.array(array_b)
    return np.linalg.norm(r)


def overlap_coefficient(
    p: np.ndarray, q: np.ndarray, method: str = "weitzman"
) -> float:
    """
    Calculate the overlap coefficient between two arrays.

    The overlap coefficient is a measure of the similarity between two sets. This method
    calculates the overlap coefficient between two input arrays using different methods.

    Args:
        p (np.ndarray): The first input array.
        q (np.ndarray): The second input array.
        method (str, optional): The method to use for calculating the overlap coefficient.
            Valid options are 'weitzman', 'matusita', 'morisita', and 'kullback-leibler'.
            Defaults to 'weitzman'.

    Returns:
        float: The overlap coefficient between the two arrays.
    """
    if method == "weitzman":
        return np.sum(np.minimum(p, q))
    elif method == "matusita":
        return np.sum(np.sqrt(p * q))
    elif method == "morisita":
        return 2.0 * np.sum(p * q) / (np.sum(p * p) + np.sum(q * q))
    elif method == "kullback-leibler":
        r = (p - q) * np.log(p / q)
        indices = np.isfinite(r)
        return 1.0 / (1.0 + np.sum(r[indices]))
    else:
        raise ValueError(
            "Invalid method. Valid options are 'weitzman', 'matusita', 'morisita', and 'kullback-leibler'."
        )


def bresenham_normalize(array: np.ndarray, to_value: int = 100) -> np.ndarray:
    """
    Normalize an array of values using the Bresenham algorithm.

    The Bresenham algorithm is used to distribute a total value (`to_value`) among the elements
    of the input array based on their relative magnitudes. The resulting normalized array
    will have the same shape as the input array.

    Args:
        array (np.ndarray): The input array of values to be normalized.
        to_value (int, optional): The total value to be distributed among the elements. Defaults to 100.

    Returns:
        np.ndarray: The normalized array with values distributed according to the Bresenham algorithm.
    """
    array_size = array.size
    m = np.zeros(array_size, dtype=int)
    indices = np.argsort(array)[
        ::-1
    ]  # Sort indices in descending order of array values
    s = np.round(array.sum())
    remainder = 0.0
    for i in indices:
        c = array[i] * float(to_value) + remainder
        m[i] = int(c // s)
        remainder = c % s
    m[indices[0]] += int(
        np.round(remainder)
    )  # In case there is any remainder left over.
    return m


def bootstrap(
    array_observed: np.ndarray,
    number_of_bootstraps: int = 1499,
    bootstrap_method=np.sum,
) -> np.ndarray:
    """
    Perform bootstrap resampling on an observed array.

    Bootstrap resampling is a statistical technique used to estimate the sampling
    distribution of a statistic. This method performs bootstrap resampling on the
    observed array by randomly sampling with replacement.

    Args:
        array_observed (np.ndarray): The observed array.
        number_of_bootstraps (int, optional): The number of bootstrap samples. Defaults to 1499.

    Returns:
        np.ndarray: An array containing the bootstrap estimates.
    """
    number_of_elements = array_observed.size
    bootstrap_estimates = np.empty(number_of_bootstraps)
    for i in range(number_of_bootstraps):
        bootstrap_indices = np.random.choice(
            number_of_elements, size=number_of_elements, replace=True
        )
        bootstrap_sample = array_observed[bootstrap_indices]
        bootstrap_estimate = bootstrap_method(bootstrap_sample)
        bootstrap_estimates[i] = bootstrap_estimate
    return bootstrap_estimates


def kappa_clipped_indices(array: np.ndarray, kappa: float = 5.0) -> np.ndarray:
    """
    Calculate the indices of values within a certain kappa range.

    This method calculates the indices of values in the array that fall within a
    certain range around the median. The range is determined by multiplying the
    Median Absolute Deviation (MAD) by the kappa value.

    Args:
        array (np.ndarray): The input array.
        kappa (float, optional): The kappa value. Defaults to 5.0.

    Returns:
        np.ndarray: An array containing the indices of the clipped values.
    """
    threshold = kappa * sigma_mad(array)
    return np.where(np.abs(array - np.median(array)) < threshold)[0]


def shuffled_indices_for(array: np.ndarray) -> np.ndarray:
    """
    Generate shuffled indices for an array.

    This method generates shuffled indices for the given array. The indices can be
    used to shuffle the array in the same order.

    Args:
        array (np.ndarray): The input array.

    Returns:
        np.ndarray: An array containing the shuffled indices.
    """
    indices = np.arange(len(array))
    np.random.shuffle(indices)
    return indices
