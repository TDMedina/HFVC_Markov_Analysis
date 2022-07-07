"""Heart Failure Virtual Clinic - Helper Objects and Functions.

@author: T.D. Medina
"""

import pandas as pd
from datetime import datetime


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


def rollback_week_to_monday(date):
    iso = date.isocalendar()
    date = datetime.fromisocalendar(iso.year, iso.week, 1).date()
    return date


def rollback_month_to_first(date):
    date = datetime(date.year, date.month, 1).date()
    return date


def rollback_year_to_first(date):
    date = datetime(date.year, 1, 1).date()
    return date


def advance_month(date):
    if date.month == 12:
        date = datetime(date.year + 1, 1, 1).date()
    else:
        date = datetime(date.year, date.month + 1, 1).date()
    return date


def advance_year(date):
    date = datetime(date.year + 1, 1, 1).date()
    return date
