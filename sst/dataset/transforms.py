import torch


class ScaleTemperature:
    """
    This transform class will scale temperature to [0, 1] range
    based on the given minimum and maximum temperatures.
    """
    def __init__(self, min_temp: float, max_temp: float) -> None:
        self.min_temp = min_temp
        self.max_temp = max_temp

    def __call__(self, sample: tuple[torch.Tensor, torch.Tensor]):
        X, y = sample
        return self.scale_temperature(X), self.scale_temperature(y)

    def scale_temperature(self, t):
        mintemp = self.min_temp
        maxtemp = self.max_temp

        # Scale to 0 and 1.
        t = (t - mintemp) / (maxtemp - mintemp)
        return t

    def inverse(self, t):
        mintemp, maxtemp = self.min_temp, self.max_temp
        return t * (maxtemp - mintemp) + mintemp


class StandardScaler:
    def __init__(self, mean: float, std: float) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, sample: tuple[torch.Tensor, torch.Tensor]):
        X, y = sample
        return self.scale(X), self.scale(y)

    def scale(self, t):
        return (t - self.mean) / self.std

    def inverse(self, t):
        return t * self.std + self.mean
