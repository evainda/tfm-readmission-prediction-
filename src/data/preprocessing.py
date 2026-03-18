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
    - Cálculo de length_of_stay (duración de la estancia en días)
    - Filtrado de registros con duración negativa
    """

    # Creamos una copia del dataframe para evitar modificar el original
    df = df.copy()

    # Eliminamos registros duplicados en caso de que existan
    df = df.drop_duplicates()

    # Convertimos las columnas de fecha a formato datetime
    # errors="coerce" convierte valores inválidos en NaT
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["dischtime"] = pd.to_datetime(df["dischtime"], errors="coerce")

    # Duración de la estancia en días. Variable muy utilizada en la literatura
    # de predicción de readmisión como indicador de gravedad del ingreso.
    df["length_of_stay"] = (df["dischtime"] - df["admittime"]).dt.days

    # Eliminamos registros con duración negativa (error en los datos de admisión/alta)
    df = df[df["length_of_stay"] >= 0]

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
    Unión de patients y admissions mediante subject_id (LEFT JOIN).

    Las features procedentes de diagnósticos se añaden en un paso posterior
    mediante add_diagnosis_features, para mantener cada transformación separada.
    """

    df = admissions.merge(
        patients,
        on="subject_id",
        how="left"
    )

    return df


def add_diagnosis_features(df, diagnoses):
    """
    Añade features derivadas de la tabla de diagnósticos.

    Variables generadas:
    - n_diagnoses: número de diagnósticos registrados por hospitalización.
      Es un indicador de la complejidad clínica del ingreso y está relacionado
      con mayor riesgo de readmisión en la literatura.

    La unión se realiza mediante 'hadm_id'. Los ingresos sin diagnósticos
    registrados reciben el valor 0.
    """

    # Contamos el número de diagnósticos por ingreso
    diag_count = (
        diagnoses.groupby("hadm_id")
        .size()
        .reset_index(name="n_diagnoses")
    )

    df = df.merge(diag_count, on="hadm_id", how="left")

    # Los ingresos sin diagnósticos registrados se imputan con 0
    df["n_diagnoses"] = df["n_diagnoses"].fillna(0).astype(int)

    return df


def group_rare_categories(df, col, threshold=0.01):
    """
    Agrupa las categorías poco frecuentes de una variable categórica en 'Other'.

    Las categorías con una frecuencia relativa inferior al umbral (por defecto 1%)
    se reemplazan por la etiqueta 'Other'. Esto reduce la dimensionalidad del
    one-hot encoding y evita columnas casi vacías que añaden ruido al modelo.

    Parámetros
    ----------
    df : pd.DataFrame
    col : str — nombre de la columna categórica
    threshold : float — proporción mínima para conservar una categoría (default 0.01)
    """

    min_count = threshold * len(df)
    counts = df[col].value_counts()
    rare_categories = counts[counts < min_count].index
    df[col] = df[col].replace({cat: "Other" for cat in rare_categories})

    return df


def create_readmission_target(df):
    """
    Crea la variable objetivo readmission_30_days.

    La variable indica si un paciente vuelve a ingresar en el hospital
    dentro de los 30 días posteriores al alta. Se excluyen las últimas
    admisiones de cada paciente (sin seguimiento conocido) para evitar sesgo.

    Variables intermedias eliminadas al final: next_admittime, days_to_next_admission.
    """

    df = df.copy()

    # Ordenar admisiones por paciente y fecha para calcular la siguiente admisión
    df = df.sort_values(["subject_id", "admittime"])

    # Obtener la fecha de la siguiente admisión del mismo paciente
    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)

    # Calcular días entre el alta y el siguiente ingreso.
    # Se usa >= 0 para excluir admisiones solapadas en MIMIC (donde next_admittime
    # puede ser anterior a dischtime), que de otro modo quedarían etiquetadas
    # incorrectamente como readmisión al satisfacer la condición <= 30.
    df["days_to_next_admission"] = (
        df["next_admittime"] - df["dischtime"]
    ).dt.days

    df["readmission_30_days"] = (
        (df["days_to_next_admission"] >= 0) &
        (df["days_to_next_admission"] <= 30)
    ).astype(int)

    # Eliminar últimas admisiones: no se conoce si hubo readmisión posterior → sesgo
    df = df[df["next_admittime"].notna()]

    # Eliminar columnas auxiliares
    df = df.drop(columns=["next_admittime", "days_to_next_admission"])

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
        "diagnoses": FILES["diagnoses"]
    })

    # Extraemos los datasets del diccionario
    patients = datasets["patients"]
    admissions = datasets["admissions"]
    diagnoses = datasets["diagnoses"]

    print("Cleaning datasets...")

    # Aplicamos las funciones de limpieza definidas anteriormente
    patients = clean_patients(patients)
    admissions = clean_admissions(admissions)

    print("Merging datasets...")

    # Integramos la información de pacientes y admisiones
    df = merge_datasets(patients, admissions)

    print("Adding diagnosis features...")

    # Añadimos el número de diagnósticos por ingreso como feature de complejidad clínica
    df = add_diagnosis_features(df, diagnoses)

    print("Computing previous admissions...")

    # previous_admissions se calcula aquí, sobre el historial completo del paciente,
    # incluyendo ingresos con hospital_expire_flag=1. Si se calculara después de
    # filtrar fallecidos, el contador quedaría incorrecto para pacientes que tuvieron
    # una muerte previa registrada en MIMIC.
    df = df.sort_values(["subject_id", "admittime"])
    df["previous_admissions"] = df.groupby("subject_id").cumcount()

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

    # Filtramos los pacientes fallecidos durante la hospitalización.
    # Estos registros no pueden tener readmisión, por lo que incluirlos
    # sesgaría la variable objetivo hacia 0 de forma artificiosa.
    print("Filtering in-hospital deaths...")
    df = df[df["hospital_expire_flag"] == 0].copy()

    print("Creating readmission target variable...")
    df = create_readmission_target(df)

    # Calculamos la edad real en el momento del ingreso.
    # anchor_age representa la edad del paciente en anchor_year (año de referencia
    # desplazado por MIMIC), no en la fecha real de admisión. La diferencia entre
    # el año de admisión y anchor_year permite aproximar la edad en cada ingreso.
    # Re-parse de admittime necesario si df procede de la lectura del CSV interim
    # (donde las fechas se guardan como string).
    df["admittime"] = pd.to_datetime(df["admittime"], errors="coerce")
    df["age_at_admission"] = df["anchor_age"] + (
        df["admittime"].dt.year - df["anchor_year"]
    )

    print("Columns before processing:", len(df.columns))

    # Eliminamos columnas que no aportan información útil para la predicción:
    # - variables temporales ya resumidas en length_of_stay y age_at_admission
    # - identificadores únicos que no son features
    # - variables con riesgo de data leakage (deathtime, dod)
    # - anchor_age y anchor_year sustituidas por age_at_admission
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
        "anchor_year",
        "anchor_age"
    ], errors="ignore")

    # Imputación de variables categóricas con valor "Unknown".
    # Se imputan aquellas que presentan valores faltantes significativos.
    # En el caso de gender y race no existen valores faltantes,
    # por lo que no se realiza imputación.
    df["insurance"] = df["insurance"].fillna("Unknown")
    df["marital_status"] = df["marital_status"].fillna("Unknown")
    df["language"] = df["language"].fillna("Unknown")
    df["discharge_location"] = df["discharge_location"].fillna("Unknown")
    df["admission_location"] = df["admission_location"].fillna("Unknown")
    df["admission_type"] = df["admission_type"].fillna("Unknown")

    # Agrupamos categorías poco frecuentes (< 1%) en 'Other' para reducir
    # la dimensionalidad del one-hot encoding y evitar columnas casi vacías.
    # race y language presentan alta cardinalidad con muchas categorías raras.
    # discharge_location incluye "DIED" como categoría residual en MIMIC tras
    # filtrar hospital_expire_flag; se agrupa en 'Other' para evitar columnas espurias.
    df = group_rare_categories(df, "race", threshold=0.01)
    df = group_rare_categories(df, "language", threshold=0.01)
    df = group_rare_categories(df, "discharge_location", threshold=0.01)

    # Codificación one-hot de variables categóricas.
    # drop_first=True elimina una categoría por variable para evitar
    # multicolinearidad perfecta (dummy variable trap).
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
