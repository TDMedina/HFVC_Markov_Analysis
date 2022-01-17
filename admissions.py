"""HFVC - Admission Objects."""

from datetime import timedelta
import pandas as pd


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
                  f"{self.length_of_stay.days} day(s)")
        return string

    def __hash__(self):
        return hash(f"{self.patient_id},{self.type},{self.index}")

    @staticmethod
    def convert_from_DatedValue(pid, dated_value):
        """DatedValue(name, value, date)"""
        _, code, _, index = dated_value.name.replace("LOS", "_LOS").split("_")
        index = int(index[1:])
        return Admission(pid, code, index, dated_value.date, timedelta(dated_value.value))


class AdmissionList:
    def __init__(self, admissions=None):
        self.ELCV = []
        self.ELXX = []
        self.EMCV = []
        self.EMHF = []
        self.EMXX = []

        if admissions is not None:
            self.assign_admissions(admissions)

    def __len__(self):
        size = self.get_counts(False)
        return size

    def assign_admissions(self, admissions):
        for admission in admissions:
            self.__getattribute__(admission.type).append(admission)

    def get_all(self, container="dict"):
        admits = {"ELCV": self.ELCV,
                  "ELXX": self.ELXX,
                  "EMCV": self.EMCV,
                  "EMHF": self.EMHF,
                  "EMXX": self.EMXX}
        match container.lower():
            case "list":
                admits = [admit for admit_list in admits.values()
                          for admit in admit_list]
                admits.sort(key=lambda admit: admit.date)
            case "lol":
                admits = [[admit_type, admit] for admit_type in admits
                          for admit in admits[admit_type]]
            case "tuples" | "tuple" | "tup":
                admits = [(admit_type, admit) for admit_type in admits
                          for admit in admits[admit_type]]
            case _:
                pass
        return admits

    # def list_all(self):
    #     admit_list = self.get_all()
    #     admit_list = [(key, admit) for key in admit_list
    #                   for admit in admit_list[key]]
    #     admit_list.sort(key=lambda x: x[1].date)
    #     return admit_list

    def show_all(self):
        string = ""
        for admit_type, admits in self.get_all().items():
            if not admits:
                continue
            string += f"{admit_type}:\n"
            string += "\n".join([f"  {str(admit)}" for admit in admits])
            string += "\n"
        print(string)

    def get_counts(self, per_type=True):
        if per_type:
            counts = {admit_type: len(admits)
                      for admit_type, admits in self.get_all().items()}
        else:
            counts = len(self.get_all("list"))
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
        for entry in sorted(self.get_all("tuples"), key=lambda x: x[1].date):
            print(f"{entry[0]}: {entry[1]}")

    def filter_admissions(self, admit_type="", date_range=None, as_AdmissionList=False):
        admit_list = self.get_all("list")
        match admit_type.upper():
            case "":
                pass
            case "EMHF" | "EMCV" | "EMXX" | "ELCV" | "ELXX":
                admit_list = self.__getattribute__(admit_type.upper())
            case "EM" | "EMERGENCY":
                admit_list = [x for x in admit_list if x.type.startswith("EM")]
            case "EL" | "ELECTIVE":
                admit_list = [x for x in admit_list if x.type.startswith("EL")]
            case "HF" | "HEART FAILURE":
                admit_list = [x for x in admit_list if x.type.endswith("HF")]
            case "CV" | "CARDIOVASCULAR":
                admit_list = [x for x in admit_list if x.type.endswith("CV")]
            case "XX" | "OTHER" | "MISC":
                admit_list = [x for x in admit_list if x.type.endswith("XX")]
            case _:
                raise ValueError("Unknown admission type.")
        if date_range is not None:
            admit_list = [x for x in admit_list
                          if date_range[0] <= x.date < date_range[1]]
        if as_AdmissionList:
            admit_list = AdmissionList(admit_list)
        return admit_list
