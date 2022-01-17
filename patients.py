"""HFVC - Patient Objects."""

from datetime import datetime, timedelta
from warnings import warn
import pandas as pd
from admissions import AdmissionList
from utilities import NamedDate


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
            ["Birth", NamedDate("Birth", self.date_of_birth)],
            ["Min Clinic Date", NamedDate("Min Clinic Date", self.min_clinic_date)],
            ["Follow-Up Date", NamedDate("Follow-Up Date", self.follow_up_date)],
            ["Death", NamedDate("Death", self.date_of_death)]
            ]
        admissions = self.admissions.filter_admissions(filter_admission_type, None, True)
        events += admissions.get_all("lol")
        events = [x for x in events if not pd.isna(x[1].date)]
        events.sort(key=lambda x: x[1].date)
        return events

    def show_timeline(self, filter_admission_type=""):
        timeline = self.make_timeline(filter_admission_type)
        timeline = [(x[0], x[1].date) if isinstance(x[1], NamedDate)
                    else (x[0], x[1]) for x in timeline]
        for entry in timeline:
            print(f"{entry[0]}: {entry[1]}")

    def determine_stage(self, reference_date=datetime.today().date()):
        date_range = [reference_date - timedelta(365), reference_date + timedelta(1)]
        stage_b = self.other_flags["StageB_FLAG_BL"]
        if self.deceased and self.date_of_death <= reference_date:
            return 5
        if pd.isna(stage_b.value) or not stage_b.value or reference_date < stage_b.date:
            if not self.admissions.filter_admissions("EM", date_range):
                return 1
            return 2
        if stage_b.value and stage_b.date <= reference_date:
            if not self.admissions.filter_admissions("EM", date_range):
                return 3
            return 4

    def make_staged_timeline(self, filter_admission_type=""):
        timeline = self.make_timeline(filter_admission_type)
        stage = 1
        for i, event in enumerate(timeline):
            stage = max(stage, self.determine_stage(event[1].date))
            timeline[i].append(stage)
        return timeline

    def show_staged_timeline(self):
        timeline = self.make_staged_timeline()
        timeline = [(x[0], x[1].date, x[2]) if isinstance(x[1], NamedDate)
                    else (x[0], x[1], x[2]) for x in timeline]
        for entry in timeline:
            print(f"{entry[0]}: {entry[1]} - Stage {entry[2]}")
