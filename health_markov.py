#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Heart Failure Virtual Consultation - Financial Analysis.

Created on Mon Jan 10 16:25:26 2022

@author: T.D. Medina
"""

from datetime import datetime
from hmmlearn import hmm
import numpy as np
import pandas as pd
from admissions import Admission, AdmissionList
from markov import make_markov_model
from patients import Patient, PatientDatabase
from prescriptions import PrescriptionList
from utilities import DatedValue


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

    # @staticmethod
    # def separate_MED_classes(patient_data):
    #     for patient, fields in patient_data.items():
    #         og_meds = fields["MED_classes_BL"]
    #         if pd.isna(og_meds.value):
    #             meds = [pd.NA]
    #         else:
    #             meds = og_meds.value.strip("[]").split(",")
    #             meds = [med.strip('"') for med in meds]
    #             meds = sorted([med for med in meds if med])
    #         patient_data[patient]["medication_classes"] = DatedValue(
    #             name="medication_classes",
    #             value=meds,
    #             date=og_meds.date
    #         )
    #         del patient_data[patient]["MED_classes_BL"]

    @staticmethod
    def separate_MED_scripts(patient_data):
        for patient_id, patient in patient_data.items():
            if pd.isna(patient["MED_script_BL"].value):
                prescriptions = pd.NA
            else:
                prescriptions = PrescriptionList._import_from_json_string(
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
        data = PatientDatabase({patient_id: Patient._patient_from_dict(patient)
                               for patient_id, patient in data.items()})
        return data


def make_random_left_right_matrix(n_states):
    array = []
    i = n_states
    while i:
        array.append([0]*(n_states-i) + [1/(i)]*i)
        i-=1
    array = np.array(array)
    return array


def make_markov_model(chains, n_components, initial_tpm,
                      extend_death_state=0):
    chains = [chain for chain in chains if chain]
    if extend_death_state > 0:
        for i, chain in enumerate(chains):
            if chain[-1] == 2:
                chains[i] += [2]*extend_death_state
    chain_lens = [len(chain) for chain in chains]
    chain_array = [[val] for chain in chains for val in chain]
    model = hmm.MultinomialHMM(n_components, init_params="se")
    model.transmat_ = initial_tpm
    model.fit(chain_array, chain_lens)
    return model


def main():
    data = HFVCDataManager.import_data("data.csv")
    return data


if __name__ == "__main__":
    dataset = main()
    chains = dataset.make_admission_chains("y", True)
    model = make_markov_model(chains, 5, make_random_left_right_matrix(5), 1000)
