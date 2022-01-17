"""HFVC - Utility Objects."""

import pandas as pd

class DatedValue:
    def __init__(self, name, value, date):
        self.name = name
        self.value = value
        self.date = date

    def __repr__(self):
        string = (f'{self.__class__.__name__}(name="{self.name}", value={self.value}, '
                  f'date={repr(self.date)})')
        return string

    def __str__(self):
        string = f"{self.name}({self.value}, {self.date})"
        return string


class NamedDate(DatedValue):
    def __init__(self, name, date):
        super().__init__(name=name, value=pd.NA, date=date)
