import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from numpy.random import random
import time
from sklearn.metrics import mean_squared_error




COLS_NAS_FALSOS = ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 
                   'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType', 
                   'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond', 
                   'PoolQC', 'Fence', 'MiscFeature']

COLS_STR_CON_ORDEN = ["LotShape", "Utilities", "LandSlope", "ExterQual", 
                      "ExterCond", "BsmtQual", "BsmtCond", "BsmtExposure", 
                      "BsmtFinType1", "BsmtFinType2", "HeatingQC", "CentralAir", 
                      "KitchenQual", "Functional", "FireplaceQu", 
                      "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", 
                      "PoolQC"]

DICT_LOTSHAPE = {"Reg": "0", "IR1":"1", "IR2":"2", "IR3":"3"}
DICT_UTILITIES = {"AllPub": "3", "NoSewr": "2", "NoSeWa": "1","ELO": "0"}
DICT_LANDSLOPE = {"Gtl":"0","Mod":"1","Sev":"2"}
DICT_EXTERQUAL = {"Ex":"4","Gd":"3","TA":"2","Fa":"1", "Po":"0"}
DICT_EXTERCOND = DICT_EXTERQUAL
DICT_BSMTQUAL = {"Ex":"5","Gd":"4","TA":"3","Fa":"2", "Po":"1", 
                 "No Basement": "0"}
DICT_BSMTCOND = DICT_BSMTQUAL
DICT_BSMTEXPOSURE = {"Gd":"4", "Av": "3", "Mn": "2", "No": "1", 
                     "No Basement": "0"}
DICT_BSMTFINTYPE1 = {"GLQ":"6","ALQ":"5","BLQ":"4","Rec":"3","LwQ":"2",
                     "Unf":"1","No Basement": "0"}
DICT_BSMTFINTYPE2 = DICT_BSMTFINTYPE1
DICT_HEATINGQC = DICT_EXTERQUAL
DICT_CENTRALAIR = {"Y": "1", "N":"0"}
DICT_KITCHENQUAL = DICT_EXTERQUAL
DICT_FUNCTIONAL ={"Typ":"7","Min1":"6","Min2":"5","Mod":"4","Maj1":"3",
                  "Maj2":"2","Sev":"1","Sal":"0"}
DICT_FIREPLACEQU = {"Ex":"5","Gd":"4","TA":"3","Fa":"2", 
                    "Po":"1", "No Fireplace":"0"}
DICT_GARAGEFINISH = {"Fin":"3","RFn":"2","Unf":"1","No Garage":"0"}
DICT_GARAGEQUAL = {"Ex":"5","Gd":"4","TA":"3","Fa":"2", "Po":"1", 
                   "No Garage": "0"}
DICT_GARAGECOND = DICT_GARAGEQUAL
DICT_PAVEDDRIVE = {"Y": "2", "P":"1", "N": "0"}
DICT_POOLQC = {"Ex":"4","Gd":"3","TA":"2","Fa":"1", "No Pool": "0"}



DICT_LIST = [DICT_LOTSHAPE,DICT_UTILITIES,DICT_LANDSLOPE,DICT_EXTERQUAL,
             DICT_EXTERCOND,DICT_BSMTQUAL,DICT_BSMTCOND,DICT_BSMTEXPOSURE,
             DICT_BSMTFINTYPE1,DICT_BSMTFINTYPE2,DICT_HEATINGQC,
             DICT_CENTRALAIR,DICT_KITCHENQUAL,DICT_FUNCTIONAL,DICT_FIREPLACEQU,
             DICT_GARAGEFINISH,DICT_GARAGEQUAL,DICT_GARAGECOND,DICT_PAVEDDRIVE,
             DICT_POOLQC]



def proporcion_nas(df):
    cols_con_nas = df.isnull().any()
    cols_con_nas = cols_con_nas.index[cols_con_nas == True]
    nrows = df.values.shape[0]
    proporcion = pd.Series([sum(df[column].isna())/nrows 
                            for column in cols_con_nas], 
                            index = [column for column in cols_con_nas])
    numero_nas = pd.Series([sum(df[column].isna()) for column in cols_con_nas], 
                            index = [column for column in cols_con_nas])
    df_out = pd.concat([proporcion, numero_nas], axis = 1)
    df_out.columns = ["Proporcion_NAs", "Num_NAs"]
    return df_out.sort_values("Num_NAs")


def corrige_nas_falsos(df):
    """
    Corrige todos los NAs falsos excepto el GarageYrBlt
    """
    
    cols_nas_falsos = COLS_NAS_FALSOS.copy()
    cols_nas_falsos.remove("GarageYrBlt")
    
    df2 = df.copy()
    nuevos_valores = ["No Alley Access","No Basement","No Basement",
                      "No Basement","No Basement","No Basement",
                      "No Fireplace","No Garage", "No Garage", "No Garage", 
                      "No Garage", "No Pool", "No Fence", "No Misc"]
    
    for col, valor in zip(cols_nas_falsos, nuevos_valores):
        df2.loc[:,col] = df2.loc[:,col].fillna(valor)
    return df2
        
def añade_features(df):
    df2 = df.copy()
    df2["TotalBath"] = df2.FullBath + df2.BsmtFullBath + df2.HalfBath + df2.BsmtHalfBath
    df2["TotalFullBath"] = df2.FullBath + df2.BsmtFullBath
    df2["Bedroom_TotalFullBath"] = df2.BedroomAbvGr / df2.TotalFullBath
    
    # Ahora las columnas con booleanos indicando si hay alley access, sótano,
    # chimenea, garaje, piscina, valla, elementos varios (misc) y si ha sido
    # reformado. El nombre de todas estas columnas acaba en "Bool", pero 
    # lo pongo como int de todos modos.
    
    df2["AlleyBool"] = (df2.Alley == "No Alley Access").astype(int)
    df2["BsmtBool"] = (df2.BsmtQual == "No Basement").astype(int)
    df2["FireplaceBool"] = (df2.FireplaceQu == "No Fireplace").astype(int)
    df2["GarageBool"] = (df2.GarageType == "No Garage").astype(int)
    df2["PoolBool"] = (df2.PoolQC == "No Pool").astype(int)
    df2["FenceBool"] = (df2.Fence == "No Fence").astype(int)
    df2["MiscBool"] = (df2.MiscFeature == "No Misc").astype(int)
    df2["ReformedBool"] = (df2.YearRemodAdd - df2.YearBuilt > 0).astype(int)
    
    return df2

def selec_no_numeric(df):
    """
    Devuelve un dataframe solo con las columnas no numéricas
    """
    numericos = ["int16", "int32","int64", "float16", "float32","float64"]
    return df.select_dtypes(exclude = numericos)

def selec_numeric(df):
    """
    Devuelve un dataframe solo con las columnas numéricas
    """
    numericos = ["int16", "int32","int64", "float16", "float32","float64"]
    return df.select_dtypes(include = numericos)

def str_a_num(df):
    df_niveles = selec_no_numeric(df)
    for col in df_niveles:
        df_niveles[col] = pd.factorize(df_niveles[col])


def factoriza_col(serie, dic):
    """
    Factoriza una columna string que tenía orden (por ejemplo, las 
    columnas que hacen referencia a calidades están ordenadas aunque 
    sean strings).
    """
    for key in dic:
        serie[key == serie] = dic[key]
    return serie.astype(int)

def factoriza_cols_con_orden(df):
    """
    Factoriza las columnas string que tenían orden (por ejemplo, las que
    hacen referencia a calidades están ordenadas aunque sean strings).
    """
    df2 = df.copy()
    for i, col in zip(range(df2.shape[1]),COLS_STR_CON_ORDEN):
        df2[col] = factoriza_col(df2[col], DICT_LIST[i])
    return df2
    
def factoriza_cols_categoricas(df):
    """
    Factoriza las columnas de strings que no tienen un orden concreto. 
    """
    df_1hot = df.copy()
    
    label_bin = preprocessing.LabelBinarizer()
            
    for col in selec_no_numeric(df_1hot):
        if len(np.unique(df_1hot[col])) > 2:
            cols_binarias = label_bin.fit_transform(df_1hot[col])
            colnames = [col+str(i) for i in range(cols_binarias.shape[1])]
            for i in range(len(colnames)):
                df_1hot[colnames[i]] = cols_binarias[:,i]
            df_1hot = df_1hot.drop(col, axis = 1)
        else:
            df_1hot[col] = df_1hot[col].factorize()[0]
    return df_1hot
    
def escalar_numericas(df):
    """
    Usa el MinMaxScaler de Scikit Learn para convertir todas las variables
    numéricas con más de 10 elementos diferentes a escala 0-1.
    """
    scaler = MinMaxScaler()
    df1 = df.copy()
    for column in df1.columns:
        if len(np.unique(df1[column])) > 10:
            arr = np.array(df1[column].dropna())
            arr1 = scaler.fit_transform(arr.reshape(-1, 1))
            df1[column] = arr1
    else:
            print("la columna" + str(column) + "es categorica; siguiente variable")
            pass
    return df1

"""
     Comprueba qué columnas tienen NAs y devuelve dos listas, una con el nombre
     de las columnas que tienen NAs ("ausentes") y otra con las que no los tienen.
"""

def check_nas(df, show = False):
  
    no_ausentes = []
    ausentes = [] 
   #hacemos dos listas, por si queremos quitarnos todas esas variables de una por ejemplo. 
   
    for col in df.columns:
       suma = df[col].isnull().sum()
       if suma > 0:
            if show:
                print("la columna  " + str(col) + "  tiene   " +
                      str(suma) + "   valores ausentes de un total de   " +
                      str(df.shape[0]) + "   filas   ")
            ausentes.append(col)
       else:
                
            if show:
                print("   la columna   " + str(col) + "   no tiene valores ausentes    ")
            no_ausentes.append(col)

    return ausentes, no_ausentes        




def decision_over_nas(df, imputation_method, threshold = 0.3, seed = 100):
    """
    Automatiza la decisión sobre los NAs. Para ello:
        *Elimina las columnas que tengan mayor proporción de NAs que threshold.
        *Imputa valores a todas las demás observaciones de acuerdo con los 
         métodos dados con imputation_method.
    
    Parámetros
    ----------
    
    df: un dataframe
    
    imputation_method: String indicando el método de imputación para las 
        columnas numéricas.
    
    threshold: La proporción de NAs máxima que puede tener una columna. Cuando
        alguna columna tiene más, se elimina directamente.
    
    seed: el random seed a usar en los métodos que lo requieren.
    """
    
    
    proportions = proporcion_nas(df)
    na_cols = proporcion_nas(df).index
    new_df = df.copy()
    
    for inc_col in na_cols:

        print(inc_col)

        proportion = proportions.Proporcion_NAs[inc_col]

        if proportion >= threshold:

            new_df = new_df.drop(inc_col, axis = 1)

        else:
            print(str(inc_col) + "Imputando valores por el metodo:"+
                  str(imputation_method))
                
            if imputation_method == "randmean":
                media = np.mean(new_df[inc_col].dropna()) #we could also make a proxy for kurtosis or other measure of the relationship
                #between mean and median, in order to design different strategies for each case, trying to bias as less as possible the data.
                np.random.seed(seed)
                rand_n = random(1)
                for i in range(new_df.shape[0]):
    
                    new_df.loc[i, inc_col] = media*rand_n
                        
            elif imputation_method == "mean":
                    
                    media = np.mean(new_df[inc_col].dropna())
                    
                    for j in range(new_df.shape[0]):
                        
                        new_df.loc[j, inc_col]  = media
                        
            else: print("No se reconoce el método")
                        
    print("YA LA TIENES TODA LIMPITA ;))")                    
    return new_df                    


def where_are_infinites(df):
    
    for column in df:
        for row in range(df.shape[0]):
            if np.isfinite(df.loc[row, column]) == False:
                print("el valor no finito esta en la fila  " + str(row) + 
                      "y la columna    " + str(column) + "   y es " +
                      str(df.loc[row, column]))
                
            else:
                next
            
"""
me hice esta funcion de arriba porque al intentar escalar las variables
no me dejaba porque habia valores infinitos; esto era para buscarlos y eliminarlos
"""




def save_cleaned_df(df, name):
    
    df.to_csv(name, sep = ",", encoding = "utf8")

"""
estoy definiendo esta funcion para que el main no quede tan engorroso, y poderselo pasar
directamente al train y al test y que funcione de pm
"""


def todo_en_uno(df, garaje, guardar, verbose = True, train = True): 
    """
    si garaje=2, te da las 2 df, si garaje = 1, te da solo las que tienen garaje
     si garaje = 0, las que no tienen garaje
    el argumento guardar funciona de forma parecida a garaje, de tal forma que le podemos decir si queremos
    que guarde 1, 2 o ningun archivo
    El argumento train esta principalmente para poner los nombres y demás, de esta forma
    en caso de que estemos con los datos de entrenamiento se llamara train_gar, train_no...y asi
    """    

    df2 = corrige_nas_falsos(df)
    df2 = añade_features(df2)
    df2 = factoriza_cols_con_orden(df2)
    df2 = df2.drop("LotFrontage", axis = 1)
    df_gar = df2[-df2.GarageYrBlt.isna()]
    df_no = df2[df2.GarageYrBlt.isna()]
    df_no = df_no.drop("GarageYrBlt", axis = 1)
    df_gar = df_gar.dropna()
    df_gar = factoriza_cols_categoricas(df_gar)
    #df_gar = escalar_numericas(df_gar)
    df_no = factoriza_cols_categoricas(df_no)
    #df_no = escalar_numericas(df_no)
    
    if verbose:
         print("empezamos con {:d} observaciones".format(df.shape[0]))
         time.sleep(5)
         print("luego tenemos {:d} observaciones".format(df2.shape[0]))
         time.sleep(5)
         print("df_gar tiene {:d} observaciones".format(df_gar.shape[0]))
         time.sleep(5)
         print("df_no tiene {:d} observaciones".format(df_no.shape[0]))
    else:
        pass

    if guardar == 2 and train == True:
        save_cleaned_df(df_gar, "train_gar.csv")
        save_cleaned_df(df_no, "train_no.csv")
        print("guardado")
    elif guardar ==2 and train == False:
        save_cleaned_df(df_gar, "test_gar.csv")
        save_cleaned_df(df_no, "test_no.csv")
        print("guardado")
    elif guardar == 1 and train == True:
        save_cleaned_df(df_gar, "train_gar.csv")
        print("guardado")
    elif guardar ==1 and train == False:
        save_cleaned_df(df_gar, "test_gar.csv")
        print("guardado")
    else:
        pass
    if garaje == 2:
        return df_no, df_gar
    elif garaje == 1:
        return df_gar
    elif garaje == 0:
        return df_no
    else:
        print("Se te ha olvidado poner qué df quieres")
 
def rmse(y_test, y_pred):
    
    return np.sqrt(mean_squared_error(y_true = y_test, y_pred = y_pred))
    
       


if __name__ == "__main__":
    train = pd.read_csv(r"./train.csv")
    train_no, train_gar = todo_en_uno(train, garaje = 2, guardar = 2, train = True)
    """
    de mmento esto de abajo no lo tocamos, ya que no nos funciona en el test set
    Si alguien puede echar un cable con este problema ayudando a entender 
    por qué no funciona con el test set igual que con el train set, se agradecería
    
    
    test = pd.read_csv(r"./test.csv")
    test_no, test_gar = todo_en_uno(test, garaje = 2, guardar = 2, train = False)
    print("listo para ponerte a hacer ML, abre el archivo housing_models y a trabajar")
    """
    
   
