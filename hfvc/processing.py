#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heart Failure Virtual Clinic - Patient Data Processing.

Created on Mon Jan 10 16:25:26 2022

@author: T.D. Medina
"""

from datetime import datetime
import importlib.resources as pkg_resources
import sys

import pandas as pd

from hfvc.admissions import Admission, AdmissionList
from hfvc.patients import Patient, PatientDatabase
from hfvc.prescriptions import PrescriptionList
from hfvc.utilities import DatedValue
from hfvc import resources


class HFVCDataManager:
    def __init__(self):
        pass

    @staticmethod
    def read_data(data_file):
        table = pd.read_csv(data_file, delimiter="\t")
        return table

    @staticmethod
    def convert_unix_dates(table):
        for col in ["CHLDate_BL", "LDLDate_BL", "HDLDate_BL", "TGSDate_BL", "GLCDate_BL"]:
            table[col] = [datetime.fromtimestamp(stamp * 86400) if not pd.isna(stamp) else pd.NaT
                          for stamp in table[col]]

    @staticmethod
    def convert_dates(table):
        for col in table:
            if "date" not in col.lower():
                continue
            table[col] = pd.to_datetime(table[col], dayfirst=True)
            table[col] = [date.to_pydatetime().date() for date in table[col]]

    @staticmethod
    def convert_bools(table):
        for col in table:
            if "flag" not in col.lower() or "date" in col.lower():
                continue
            table[col] = table[col].astype("boolean")

    @staticmethod
    def rename_columns(table):
        mapping = {
            "Patient ID": "patient_id",
            "male": "sex",
            "birth_date": "date_of_birth",
            "RIP_flag": "deceased",
            "death_date": "date_of_death",
            "DaysFU": "follow_up_duration",
            "FollowUpDate": "follow_up_date",
            }
        table.rename(columns=mapping, inplace=True)

    @staticmethod
    def convert_sex(table):
        conversion = {0: "female", 1: "male"}
        table.rename(columns={"male": "sex"}, inplace=True)
        table["sex"] = [conversion[x] for x in table["sex"]]

    @staticmethod
    def make_dated_values(patient_data, fields=None):
        if fields is None:
            fields = list(
                next((patient for patient in patient_data.values())).keys()
                )
        for field in sorted(fields):
            if "LOS_n" in field:
                date = field.replace("LOS", "Date")
            elif field.endswith("_BL") and not field.endswith("Date_BL"):
                date = field.replace("_BL", "Date_BL")
            else:
                continue
            if date not in fields:
                continue

            for patient_id in patient_data:
                patient_data[patient_id][field] = DatedValue(
                    field,
                    patient_data[patient_id][field],
                    patient_data[patient_id][date]
                    )
                del patient_data[patient_id][date]
        return

    @staticmethod
    def separate_MED_scripts(patient_data):
        for patient_id, patient in patient_data.items():
            if pd.isna(patient["MED_script_BL"].value):
                prescriptions = pd.NA
            else:
                prescriptions = PrescriptionList.import_from_json_string(
                    patient_id,
                    patient["MED_script_BL"].value
                    )
            patient_data[patient_id]["prescriptions"] = prescriptions
            del patient_data[patient_id]["MED_script_BL"]
            del patient_data[patient_id]["MED_classes_BL"]

    @staticmethod
    def group_admissions(patient_data):
        for patient, fields in patient_data.items():

            admissions = []
            remove = []
            for field, value in fields.items():
                if not field.startswith("ADM_"):
                    continue
                if not (pd.isna(value.value) and pd.isna(value.date)):
                    admission = Admission.convert_from_DatedValue(patient, value)
                    admissions.append(admission)
                remove.append(field)
            for field in remove:
                del patient_data[patient][field]

            patient_data[patient]["admissions"] = AdmissionList(admissions)

    @staticmethod
    def group_DGN_flags(patient_data):
        for patient, fields in patient_data.items():
            dgns = {field: value for field, value in fields.items() if field.startswith("DGN_")}
            for dgn in dgns:
                del patient_data[patient][dgn]
            patient_data[patient]["dgn_flags"] = dgns

    @staticmethod
    def group_flags(patient_data):
        for patient, fields in patient_data.items():
            flags = {field: value for field, value in fields.items() if field.lower().endswith("flag_bl")}
            for flag in flags:
                del patient_data[patient][flag]
            patient_data[patient]["other_flags"] = flags

    @staticmethod
    def group_bl_metrics(patient_data):
        for patient, fields in patient_data.items():
            bls = {field: value for field, value in fields.items() if field.lower().endswith("_bl")}
            for bl in bls:
                del patient_data[patient][bl]
            patient_data[patient]["metrics"] = bls

    @classmethod
    def import_data(cls, filepath):
        data = cls.read_data(filepath)
        cls.convert_unix_dates(data)
        cls.convert_dates(data)
        cls.convert_bools(data)
        cls.rename_columns(data)
        cls.convert_sex(data)
        data = data.to_dict(orient="records")
        data = {patient["patient_id"]: patient for patient in data}
        cls.make_dated_values(data)
        cls.separate_MED_scripts(data)
        cls.group_admissions(data)
        cls.group_DGN_flags(data)
        cls.group_flags(data)
        cls.group_bl_metrics(data)
        data = PatientDatabase({patient_id: Patient.patient_from_dict(patient)
                               for patient_id, patient in data.items()})
        return data


def read_default_data():
    data = pkg_resources.path(resources, "data.csv")
    return data


def main(data=None):
    data = data or read_default_data()
    data = HFVCDataManager.import_data(data)
    return data


if __name__ == "__main__":
    if len(sys.argv) == 2:
        dataset = main(sys.argv[1])
    else:
        dataset = main()
    # chains = dataset.make_admission_chains("y", True, include_empties=False)
    # init_params = hfvc_hmm.make_random_init_params(5, 3)
    # test_model = hfvc_hmm.TestModel(*hfvc_hmm.multichain_baum_welch(chains, **init_params))
    # hmmlearn_model = hfvc_hmm.make_hmmlearn_model(chains, 5, **init_params)
