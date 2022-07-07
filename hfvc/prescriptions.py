"""Heart Failure Virtual Clinic - Medication Prescription Objects.

@author: T.D. Medina
"""

from datetime import datetime
import json


class Prescription:
    def __init__(self, trade, generic, dose, freq, daily_dose, unit,
                 drug_classes=None, **kwargs):
        self.trade_name = trade
        self.generic_name = generic
        self.dose = dose
        self.frequency = freq
        self.daily_dose = daily_dose
        self.unit = unit
        self.drug_classes = drug_classes
        if "class" in kwargs:
            self.drug_classes = kwargs["class"]

    def __repr__(self):
        string = "Prescription("
        string += ", ".join([f"{name}={attr}" for name, attr in self.__dict__.items()])
        string += ")"
        return string

    def __str__(self):
        if self.trade_name:
            string = f"Prescription({self.trade_name})"
        else:
            string = f"Prescription({self.generic_name})"
        return string


class PrescriptionList:
    def __init__(self, patient_id, ID, date, drug_classes, script_type, prescriptions):
        self.patient_ID = patient_id
        self.ID = ID
        self.date = date
        self.drug_classes = drug_classes
        self.type = script_type
        self.prescriptions = prescriptions

    def __str__(self):
        string = (f"PrescriptionList(patient_ID={self.patient_ID}, "
                  f"drug_classes={self.drug_classes}")
        return string

    def __len__(self):
        return len(self.prescriptions)

    @staticmethod
    def import_from_json_string(patient_id, json_string):
        json_dict = json.loads(json_string)
        prescription_list = PrescriptionList(
            patient_id=patient_id,
            ID=json_dict["id"],
            date=datetime.strptime(json_dict["date"], "%Y-%m-%d").date(),
            drug_classes=sorted([med for med in json_dict["class"]]),
            script_type=json_dict["type"],
            prescriptions=[Prescription(**med) for med in json_dict["meds"]]
            )
        return prescription_list

    def all_medications(self):
        trades = [med.trade_name for med in self.prescriptions]
        generics = [med.generic_name for med in self.prescriptions]
        meds = list(zip(trades, generics))
        return meds
