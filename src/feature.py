from pandas import DataFrame


def add_cabin_known_col(df: DataFrame):
    '''
    Добавляем новый признак CabinKnown
    return: Новый дата сет с новым признаком
    '''
    data = df.copy()
    data['CabinKnown'] = data['Cabin'].notna()
    return data

def add_fare_log(df: DataFrame):
    pass