import pandas as pd

from src.config import FILES, DATA_INTERIM, DATA_PROCESSED
from src.data.load import load_multiple_datasets



def clean_admissions(df):
    """
    Limpieza básica de la tabla admissions.

    En esta función se realizan las primeras operaciones de limpieza
    sobre la tabla de admisiones hospitalarias.

    Operaciones realizadas:
    - Eliminación de duplicados
    - Conversión de columnas de fechas a formato datetime
   
    """

    # Creamos una copia del dataframe para evitar modificar el original
    df = df.copy()

    # Eliminamos registros duplicados en caso de que existan
    df = df.drop_duplicates()

    # Convertimos las columnas de fecha a formato datetime
    # errors="coerce" convierte valores inválidos en NaT
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")

    # Calculamos la duración de la estancia hospitalaria en días -> podria ser interesante en muchos papers se hace.
    df["length_of_stay"] = (df["dischtime"] - df["admittime"]).dt.days

    return df


def clean_patients(df):
    """
    Limpieza básica de la tabla patients.

    Esta tabla contiene información del paciente.

    Operaciones realizadas:
    - Eliminación de duplicados
    - Conversión de columnas numéricas al tipo adecuado
    """

    # Copia del dataframe
    df = df.copy()

    # Eliminamos duplicados si existieran
    df = df.drop_duplicates()

    # Convertimos anchor_year a valor numérico
    # Si hay errores de formato se convierten en NaN
    df["anchor_year"] = pd.to_numeric(df["anchor_year"], errors="coerce")

    return df


def merge_datasets(patients, admissions):
    """
    Unión de los datasets principales. 
    A PRIORI solo se integran las tablas de pacientes y admisiones.TODO

    En este caso se combinan:
    - la tabla de pacientes (patients)
    - la tabla de admisiones hospitalarias (admissions)

    La unión se realiza mediante la clave 'subject_id',
    que identifica de forma única a cada paciente.

    Se utiliza un LEFT JOIN para mantener todas las admisiones
    incluso si faltara alguna información del paciente.
    """

    df = admissions.merge(
        patients,
        on="subject_id",
        how="left"
    )

    return df


def save_interim(df):
    """
    Guarda el dataset limpio en la carpeta 'interim'.

    La carpeta interim contiene datasets intermedios que ya han sido
    limpiados pero todavía no están completamente preparados para el modelado.
    """

    path = DATA_INTERIM / "cleaned_dataset.csv"

    df.to_csv(path, index=False)


def save_processed(df):
    """
    Guarda el dataset final en la carpeta 'processed'.

    Esta versión del dataset es la que se utilizará posteriormente
    para el entrenamiento de modelos.
    """

    path = DATA_PROCESSED / "model_dataset.csv"

    df.to_csv(path, index=False)


def run_preprocessing_part1():
    """
    Primera etapa del pipeline de preprocesamiento.

    En esta fase se realiza la limpieza básica de los datasets
    y su integración en un único dataframe.

    """

    print("Loading datasets...")

    # Cargamos únicamente las tablas necesarias para esta fase.
    # Aunque en el EDA se analizaron más tablas (procedures, prescriptions),
    # en el pipeline de preprocesamiento inicial solo se utilizan las
    # tablas más relevantes para reducir el consumo de memoria.

    datasets = load_multiple_datasets({
        "patients": FILES["patients"],
        "admissions": FILES["admissions"],
        "diagnoses": FILES["diagnoses"] #esta de momento no se está utilizandoTODO 
    })

    # Extraemos los datasets del diccionario
    patients = datasets["patients"]
    admissions = datasets["admissions"]

    print("Cleaning datasets...")

    # Aplicamos las funciones de limpieza definidas anteriormente
    patients = clean_patients(patients)
    admissions = clean_admissions(admissions)

    print("Merging datasets...")

    # Integramos la información de pacientes y admisiones
    df = merge_datasets(patients, admissions)

    print("Saving interim dataset...")

    # Guardamos una versión intermedia del dataset
    save_interim(df)

    print("Preprocessing completed.")

    return df

def run_preprocessing_part2(df):
    """
    Segunda etapa del pipeline.

    - creación de la variable objetivo readmission_30_days
    - eliminación de variables irrelevantes
    - tratamiento de valores faltantes
    - codificación de variables categóricas

    Genera el dataset final en la carpeta 'processed'.
    """

    print("Preparing dataset for modeling...")
    print("Creating readmission target variable...")
    df = create_readmission_target(df)
    print("Columns before processing:", len(df.columns))
    # eliminar columnas irrelevantes
    df = df.drop(columns=[
        "deathtime",
        "dod",
        "edregtime",
        "edouttime",
        "admittime",
        "dischtime",
        "anchor_year_group",
        "admit_provider_id",
        "subject_id",
        "hadm_id",
        "hospital_expire_flag",
        "anchor_year"
    ], errors="ignore")

    # imputación de variables categóricas con valor "Unknown"
    #se imputan aquellas que presentas una cantidad significativa de valores faltantes.

    df["insurance"] = df["insurance"].fillna("Unknown")
    df["marital_status"] = df["marital_status"].fillna("Unknown")
    df["language"] = df["language"].fillna("Unknown")
    df["discharge_location"] = df["discharge_location"].fillna("Unknown")
    df["admission_location"] = df["admission_location"].fillna("Unknown")
    df["admission_type"] = df["admission_type"].fillna("Unknown")
    
    #en el caso de gender y race no existen valores faltantes, 
    # por lo que no se realiza imputación.

    # variables categóricas para encoding
    categorical_cols = [
        "gender", 
        "race",
        "insurance",
        "marital_status",
        "language",
        "admission_type",
        "admission_location",
        "discharge_location"
    ]

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("Columns after encoding:", len(df.columns))
    print("Saving processed dataset...")

    save_processed(df)

    print("Model dataset created.")

    return df

#La variable objetivo del modelo corresponde a la readmisión hospitalaria dentro de
#  los 30 días posteriores al alta. Para su construcción se ordenaron cronológicamente las admisiones de cada
#  paciente y se calculó el intervalo temporal entre el alta hospitalaria y el siguiente ingreso. Cuando este intervalo
#  fue inferior o igual a 30 días se consideró que se había producido una readmisión.

def create_readmission_target(df):
    """
    Crea la variable objetivo readmission_30_days.

    La variable indica si un paciente vuelve a ingresar
    en el hospital dentro de los 30 días posteriores al alta.
    """

    df = df.copy()
    
    # ordenar admisiones por paciente y fecha
    df = df.sort_values(["subject_id", "admittime"])

    
    #NUEVA VARIABLE
    # número de ingresos previos
    # Se incorporó una variable adicional que representa el número de ingresos hospitalarios previos 
    # del paciente. Esta variable captura la frecuencia de hospitalización previa y se ha identificado 
    #  como un factor relevante para la predicción de readmisiones.
    df["previous_admissions"] = df.groupby("subject_id").cumcount()
    
    #NUEVA VARIABLE QUE SIRVE PARA ELIMINAR
    # obtener la siguiente admisión del mismo paciente
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)

    # calcular días hasta la siguiente admisión
    df["days_to_next_admission"] = (
        df["next_admittime"] - df["dischtime"]
    ).dt.days

    #NUEVA VARIABLE
    # crear variable objetivo
    df["readmission_30_days"] = (
        df["days_to_next_admission"] <= 30
    ).astype(int)

    # eliminar últimas admisiones (no sabemos si hubo readmisión)
    #las últimas admisiones de cada paciente no deberían usarse para entrenamiento, porque no sabemos si hubo readmisión después. -> sesgo
    df = df[df["next_admittime"].notna()]

    # eliminar columnas auxiliares
    df = df.drop(columns=[
        "next_admittime",
        "days_to_next_admission"
    ])

    return df





    
    