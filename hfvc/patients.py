"""Heart Failure Virtual Clinic - Patient Objects.

@author: T.D. Medina
 """

from datetime import datetime, timedelta
from warnings import warn

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from hfvc.admissions import AdmissionList, Admission
import hfvc.utilities as ut

pio.renderers.default = "browser"


class PatientDatabase:
    def __init__(self, patients=None):
        self.patients = patients
        if self.patients is None:
            self.patients = {}
        self._index = list(self.patients.keys())

    def __str__(self):
        string = f"PatientDatabase(size={self.size})"
        return string

    def __len__(self):
        size = len(self.patients)
        return size

    def __contains__(self, item):
        return item in self.patients

    def __getitem__(self, key):
        return self.patients[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Patient):
            raise TypeError("Value to add is not Patient object.")
        self.patients[key] = value

    def __iter__(self):
        self.__iteri__ = 0
        return self

    def __next__(self):
        if self.__iteri__ == self.size:
            raise StopIteration
        result = self.patients[self._index[self.__iteri__]]
        self.__iteri__ += 1
        return result

    def keys(self):
        return self.patients.keys()

    def values(self):
        return self.patients.values()

    def items(self):
        return self.patients.items()

    @property
    def size(self):
        return self.__len__()

    def make_admission_chains(self, interval="year", start=None, stop=None,
                              admit_types="EM", shift_death=True, add_min_clinic_date=True,
                              add_follow_up_date=True, values_only=False, min_length=0):
        chains = {
            patient.id: patient.make_admission_chain(interval, start, stop, admit_types,
                                                     shift_death, add_min_clinic_date,
                                                     add_follow_up_date, values_only)
            for patient in self
            }
        chains = {patient_id: chain for patient_id, chain in chains.items()
                  if len(chain) >= min_length}
        if values_only:
            chains = list(chains.values())
        return chains

    # def make_stage_chains(self, interval="year",admit_types="EM", date_range=None):
    #     chains = {patient.id: chain for patient in self
    #               if (chain := patient.make_stage_chain(interval, admit_types))}
    #     return chains

    def plot_patient_chains2(self, interval="year", admit_types="EM", date_range=None,
                             normalize=False):
        fig = go.Figure()
        for i, patient in enumerate(self, start=1):
            chain = patient.make_admission_chain(interval, False, admit_types, date_range)
            if not chain:
                continue
            if normalize:
                chain = {i: admit for i, (date, admit)
                         in enumerate(chain, start=1)}
            admits = [date for date, admit in chain if admit == 1]
            death = [date for date, admit in chain if admit == 2]
            fig.add_trace(go.Scatter(
                x=[date for date, admit in chain], y=[i]*len(chain), name=patient.id,
                mode="lines", line={"color": "black"},
                legendgroup=patient.id,
                hovertemplate="No admission: %{x}")
                )
            fig.add_trace(go.Scatter(
                x=admits, y=[i]*len(admits), name=patient.id,
                mode="markers", marker={"color": "blue", "symbol": "circle"},
                legendgroup=patient.id, showlegend=False,
                hovertemplate="Admissions: %{x}")
                )
            fig.add_trace(go.Scatter(
                x=death, y=[i]*len(death), name=patient.id,
                mode="markers", marker={"color": "red", "symbol": "x"},
                legendgroup=patient.id, showlegend=False,
                hovertemplate="Death: %{x}")
                )
        fig.show()
        return


class Patient:
    def __init__(self, patient_id, MRN, patient_type, sex, date_of_birth,
                 min_clinic_date, follow_up_date, follow_up_duration,
                 deceased=False, date_of_death=pd.NaT, prescriptions=None,
                 metrics=None, dgn_flags=None, other_flags=None, admissions=None,
                 **kwargs):
        self.id = patient_id
        self.MRN = MRN
        self.type = patient_type
        self.sex = sex

        self.min_clinic_date = min_clinic_date
        self.follow_up_date = follow_up_date
        self.follow_up_duration = follow_up_duration
        self.admissions = admissions
        if admissions is None:
            self.admissions = AdmissionList()

        self.prescriptions = prescriptions
        self.metrics = metrics
        self.dgn_flags = dgn_flags
        self.other_flags = other_flags

        self.date_of_birth = date_of_birth
        self.deceased = deceased
        self.date_of_death = date_of_death

        # self.stage = self.determine_stage()

    def __repr__(self):
        string = f"Patient(ID={self.id})"
        return string

    @property
    def age(self):
        if self.deceased is True:
            warn("Warning: Patient is deceased. Age is age at death.")
            end = self.date_of_death
        else:
            end = datetime.today().date()
        age = round((end - self.date_of_birth).days / 365, 2)
        return age

    @staticmethod
    def _patient_from_dict(p_dict):
        return Patient(**p_dict)

    def make_timeline(self, filter_admission_type=""):
        events = [
            ["Birth", ut.NamedDate("Birth", self.date_of_birth)],
            ["Min Clinic Date", ut.NamedDate("Min Clinic Date", self.min_clinic_date)],
            ["Follow-Up Date", ut.NamedDate("Follow-Up Date", self.follow_up_date)],
            ["Death", ut.NamedDate("Death", self.date_of_death)]
            ]
        admissions = self.admissions.filter_admissions(filter_admission_type, as_AdmissionList=True)
        events += admissions.get_admit_type_tuples()
        events = [x for x in events if not pd.isna(x[1].date)]
        events.sort(key=lambda x: x[1].date)
        return events

    def show_timeline(self, filter_admission_type=""):
        timeline = self.make_timeline(filter_admission_type)
        timeline = [(x[0], x[1].date) if isinstance(x[1], ut.NamedDate)
                    else (x[0], x[1]) for x in timeline]
        for entry in timeline:
            print(f"{entry[0]}: {entry[1]}")

    def determine_stage(self, reference_date=datetime.today().date()):
        start_date = reference_date - timedelta(365)
        stop_date = reference_date
        stage_b = self.other_flags["StageB_FLAG_BL"]
        if self.deceased and self.date_of_death <= reference_date:
            return 5
        if pd.isna(stage_b.value) or not stage_b.value or reference_date < stage_b.date:
            if not self.admissions.filter_admissions("EM", start_date, stop_date):
                return 1
            return 2
        if stage_b.value and stage_b.date <= reference_date:
            if not self.admissions.filter_admissions("EM", start_date, stop_date):
                return 3
            return 4

    def make_stage_chain(self, interval="year", admit_types="EM", values_only=False):
        match interval.lower():
            case "d" | "day":
                rollback = lambda date: date - timedelta(1)
                advance_date = lambda date: date + timedelta(1)
            case "w" | "week":
                rollback = lambda date: ut.rollback_week_to_monday(date)
                advance_date = lambda date: date + timedelta(7)
            case "m" | "month":
                rollback = lambda date: ut.rollback_month_to_first(date)
                advance_date = lambda date: ut.advance_month(date)
            case "y" | "year":
                rollback = lambda date: ut.rollback_year_to_first(date)
                advance_date = lambda date: ut.advance_year(date)
            case _:
                raise ValueError("Unknown interval type.")
        admissions = self.admissions.filter_admissions(admit_types)
        admissions.sort(key=lambda admit: admit.date)

        if self.deceased is True and not pd.isna(self.date_of_death):
            death = Admission(self.id, "Death", -1, self.date_of_death, 1)
            admissions.append(death)

        if not admissions:
            if values_only:
                return []
            return {}

        position = rollback(admissions[0].date)
        last = advance_date(admissions[-1].date + timedelta(admissions[-1].length_of_stay))
        stage_chain = {}
        min_stage = 1
        while position <= last:
            stage = max(min_stage, self.determine_stage(position))
            min_stage = stage
            stage_chain[position] = stage
            position = advance_date(position)
        if values_only:
            stage_chain = list(stage_chain.values())
        return stage_chain

    def make_staged_timeline(self, filter_admission_type=""):
        timeline = self.make_timeline(filter_admission_type)
        stage = 1
        for i, event in enumerate(timeline):
            stage = max(stage, self.determine_stage(event[1].date))
            timeline[i].append(stage)
        return timeline

    def show_staged_timeline(self, filter_admission_type=""):
        timeline = self.make_staged_timeline(filter_admission_type)
        timeline = [(x[0], x[1].date, x[2]) if isinstance(x[1], ut.NamedDate)
                    else (x[0], x[1], x[2]) for x in timeline]
        for entry in timeline:
            print(f"{entry[0]}: {entry[1]} - Stage {entry[2]}")

    def make_admission_chain(self, interval="year", start=None, stop=None,
                             admit_types="EM", shift_death=True, add_min_clinic_date=True,
                             add_follow_up_date=True, values_only=False):
        # Make appropriate time interval functions.
        match interval.lower():
            case "d" | "day":
                split_admit = lambda admit: admit.split_into_days()
                advance_date = lambda date: date + timedelta(1)
            case "w" | "week":
                split_admit = lambda admit: admit.split_into_isoweeks()
                advance_date = lambda date: date + timedelta(7)
            case "m" | "month":
                split_admit = lambda admit: admit.split_into_months()
                advance_date = lambda date: ut.advance_month(date)
            case "y" | "year":
                split_admit = lambda admit: admit.split_into_years()
                advance_date = lambda date: ut.advance_year(date)
            case _:
                raise ValueError("Unknown interval type.")

        # Add admission dates to chain.
        admissions = self.admissions.filter_admissions(admit_types, start, stop)
        admissions = {date: 1 for admit in admissions for date in split_admit(admit)}

        # Add death to chain.
        if self.deceased is True and not pd.isna(self.date_of_death):
            death_date = Admission(self.id, "Death", 0, self.date_of_death, 1)
            death_date = split_admit(death_date)[0]
            # Advance death date to next time period if masking admissions.
            if shift_death and death_date in admissions:
                death_date = advance_date(death_date)
            admissions[death_date] = 2

        # Add follow-up date to chain.
        if (add_follow_up_date and not pd.isna(self.follow_up_date)
                and self.follow_up_date not in admissions):
            follow_up = Admission(self.id, "Follow-Up", 0, self.follow_up_date, 1)
            follow_up = split_admit(follow_up)[0]
            admissions[follow_up] = 0

        # Add min clinic date to chain.
        if (add_min_clinic_date and not pd.isna(self.min_clinic_date)
                and self.min_clinic_date not in admissions):
            min_clin = Admission(self.id, "Min-Clin", 0, self.min_clinic_date, 1)
            min_clin = split_admit(min_clin)[0]
            admissions[min_clin] = 0

        # Fill non-admission dates.
        admit_dates = sorted(admissions.keys())
        for i, date in enumerate(admit_dates, start=0):
            if i == len(admit_dates)-1:
                break
            diff = admit_dates[i+1] - admit_dates[i]
            non_admissions = Admission(self.id, "Non-admission", 0,
                                       date+timedelta(days=1), diff.days-1)
            non_admissions = split_admit(non_admissions)
            non_admissions = set(non_admissions) - set(admit_dates)
            non_admissions = {non_admit: 0 for non_admit in non_admissions}
            admissions.update(non_admissions)

        if values_only:
            admissions = [x[1] for x in sorted(admissions.items())]
            return admissions
        return list(admissions.items())
