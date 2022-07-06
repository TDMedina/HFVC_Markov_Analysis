"""Heart Failure Virtual Clinic - Admission Markov Modelling.

@author: T.D. Medina
"""

from markov_modelling import simple_markov as smm


def build_simple_markov_model(patient_database, n_states, interval="month",
                              add_min_clinic_date=True, add_follow_up_date=True):
    chains = patient_database.make_stage_chains(
        interval=interval,
        values_only=True,
        add_min_clinic_date=add_min_clinic_date,
        add_follow_up_date=add_follow_up_date
        )
    init_dist, tpm = smm.train_tpm(chains, n_states)
    return init_dist, tpm


def smm_by_patient_type(patient_database, n_states=5, interval="month",
                        add_min_clinic_date=True, add_follow_up_date=True):
    model_params = dict(n_states=n_states, interval=interval,
                        add_min_clinic_date=add_min_clinic_date,
                        add_follow_up_date=add_follow_up_date)
    models = {}
    for patient_type in {patient.type for patient in patient_database}:
        patients = patient_database.subset_type(patient_type)
        models[patient_type] = build_simple_markov_model(patients, **model_params)
    return models


def smm_patient_type_against_others(patient_database, patient_type, n_states=5,
                                    interval="month", add_min_clinic_date=True,
                                    add_follow_up_date=True):
    model_params = dict(n_states=n_states, interval=interval,
                        add_min_clinic_date=add_min_clinic_date,
                        add_follow_up_date=add_follow_up_date)
    patients, others = patient_database.separate_type(patient_type)
    models = dict()
    models[patient_type] = build_simple_markov_model(patients, **model_params)
    models["Others"] = build_simple_markov_model(others, **model_params)
    return models


def make_all_models(patient_database, n_states=5, interval="month",
                    add_min_clinic_date=True, add_follow_up_date=True):
    model_params = dict(n_states=n_states,
                        interval=interval,
                        add_min_clinic_date=add_min_clinic_date,
                        add_follow_up_date=add_follow_up_date)
    models = dict()
    models["All"] = build_simple_markov_model(patient_database, **model_params)
    models |= smm_by_patient_type(patient_database, **model_params)
    return models
