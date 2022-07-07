"""Heart Failure Virtual Clinic - Hospital Admission Objects.

@author: T.D. Medina
"""

from datetime import datetime, timedelta


ADMISSION_TYPES = {
    "ELCV": "elective CV",
    "ELXX": "elective misc.",
    "EMCV": "emergency CV",
    "EMHF": "emergency HF",
    "EMXX": "emergency misc."
    }


class Admission:
    def __init__(self, patient_id, admission_type, index, date, length_of_stay):
        self.patient_id = patient_id
        self.index = index
        self.type = admission_type
        self.date = date
        self.length_of_stay = length_of_stay

    def __repr__(self):
        string = (f'Admission(patient_id="{self.patient_id}", type="{self.type}", '
                  f'index={self.index}, date={repr(self.date)}, '
                  f'length_of_stay={repr(self.length_of_stay)})')
        return string

    def __str__(self):
        string = (f"Patient {self.patient_id} {ADMISSION_TYPES[self.type]} "
                  f"admission #{self.index}: {self.date.strftime('%Y-%m-%d')}, "
                  f"{self.length_of_stay} day(s)")
        return string

    def __hash__(self):
        return hash(f"{self.patient_id},{self.type},{self.index}")

    @staticmethod
    def convert_from_DatedValue(pid, dated_value):
        """DatedValue(name, value, date)"""
        _, code, _, index = dated_value.name.replace("LOS", "_LOS").split("_")
        index = int(index[1:])
        return Admission(pid, code, index, dated_value.date, int(dated_value.value))

    def split_into_days(self):
        days = [self.date + timedelta(days=i) for i in range(self.length_of_stay)]
        return days

    def split_into_isoweeks(self):
        days = self.split_into_days()
        weeks = sorted(
            {datetime.fromisocalendar(day.isocalendar().year, day.isocalendar().week, 1).date()
             for day in days}
             )
        return weeks

    def split_into_months(self):
        days = self.split_into_days()
        months = sorted({datetime(day.year, day.month, 1).date() for day in days})
        return months

    def split_into_years(self):
        days = self.split_into_days()
        years = sorted({datetime(day.year, 1, 1).date() for day in days})
        return years

    def in_date_range(self, start_date, end_date):
        if start_date is None and end_date is None:
            return True
        first = self.date
        last = self.date + timedelta(self.length_of_stay)
        if end_date is None:
            return start_date <= last
        if start_date is None:
            return first < end_date
        return start_date <= last and first < end_date


class AdmissionList:
    def __init__(self, admissions=None):
        if admissions is None:
            self.admissions = []
        else:
            self.admissions = sorted(admissions, key=lambda admit: admit.date)

        self.ELCV = []
        self.ELXX = []
        self.EMCV = []
        self.EMHF = []
        self.EMXX = []
        if admissions is not None:
            self.assign_admissions(admissions)

    def __len__(self):
        size = len(self.admissions)
        return size

    @property
    def _size(self):
        size = len(self.admissions)
        return size

    def __str__(self):
        return f"AdmissionList(size={len(self)}"

    def assign_admissions(self, admissions):
        for admission in admissions:
            self.__getattribute__(admission.type).append(admission)

    def get_admit_dict(self):
        admit_dict = {"ELCV": self.ELCV,
                      "ELXX": self.ELXX,
                      "EMCV": self.EMCV,
                      "EMHF": self.EMHF,
                      "EMXX": self.EMXX}
        return admit_dict

    def get_admit_type_tuples(self):
        admits = [(admit.type, admit) for admit in self.admissions]
        return admits

    def show_all(self):
        string = ""
        for admit_type, admits in self.get_admit_dict().items():
            if not admits:
                continue
            string += f"{admit_type}:\n"
            string += "\n".join([f"  {str(admit)}" for admit in admits])
            string += "\n"
        print(string)

    def get_counts(self):
        counts = {admit_type: len(admits)
                  for admit_type, admits in self.get_admit_dict().items()}
        return counts

    def show_counts(self):
        string = ""
        for admit_type, count in self.get_counts().items():
            if count == 0:
                continue
            string += f"{admit_type}: {count}\n"
        print(string)
        # return string

    def show_timeline(self):
        for entry in self.get_admit_type_tuples():
            print(f"{entry[0]}: {entry[1]}")

    def filter_admissions(self, admit_type="", start_date=None, end_date=None,
                          as_AdmissionList=False):
        match admit_type.upper():
            case "" | "ALL":
                admit_list = self.admissions
            case "EMHF" | "EMCV" | "EMXX" | "ELCV" | "ELXX":
                admit_list = self.__getattribute__(admit_type.upper())
            case "EM" | "EMERGENCY":
                admit_list = [x for x in self.admissions if x.type.startswith("EM")]
            case "EL" | "ELECTIVE":
                admit_list = [x for x in self.admissions if x.type.startswith("EL")]
            case "HF" | "HEART FAILURE":
                admit_list = [x for x in self.admissions if x.type.endswith("HF")]
            case "CV" | "CARDIOVASCULAR":
                admit_list = [x for x in self.admissions if x.type.endswith("CV")]
            case "XX" | "OTHER" | "MISC":
                admit_list = [x for x in self.admissions if x.type.endswith("XX")]
            case _:
                raise ValueError("Unknown admission type.")
        if not (start_date is None and end_date is None):
            admit_list = [x for x in admit_list if x.in_date_range(start_date, end_date)]
        if as_AdmissionList:
            admit_list = AdmissionList(admit_list)
        return admit_list

    def admission_date_range(self, admit_type=""):
        admits = self.filter_admissions(admit_type)
        if not admits:
            return None
        min_date = min([admit.date for admit in admits])
        max_date = max([admit.date + timedelta(admit.length_of_stay-1) for admit in admits])
        return min_date, max_date
