import pandas as pd 
import numpy as np
import os
from datetime import datetime,date,time
pd.set_option('display.max_columns', None)
import warnings

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df

def location_north_south(latitude):
    '''
        Positive latitude is above the equator (N), and negative latitude is below the equator (S).
    '''
    if np.float(latitude) <0:
        return 'South'
    elif np.float(latitude) == np.nan:
        return 'Not Available'
    else:
        return 'North'
    
def location_east_west(longitude):
    '''
        Positive longitude is east of the prime meridian, while negative longitude 
        is west of the prime meridian (a north-south line that runs through a point in England).
    '''
    if np.float(longitude)<0:
        return 'West'
    elif np.float(longitude) == np.nan:
        return 'Not Available'
    else:
        return 'East'



def bearing_array(lat, lng):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lat - lng)
    lat, lng = map(np.radians, (lat, lng))
    y = np.sin(lng_delta_rad) * np.cos(lat)
    x = np.cos(lat) * np.sin(lat) - np.sin(lng) * np.cos(lng) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

def haversine_distance(df, lat1, long1, lat2, long2):
    """
    Calculates the haversine distance between 2 sets of GPS coordinates in df
    """
    r = 6371  # average radius of Earth in kilometers
       
    phi1 = np.radians(df[lat1])
    phi2 = np.radians(df[lat2])
    
    delta_phi = np.radians(df[lat2]-df[lat1])
    delta_lambda = np.radians(df[long2]-df[long1])
     
    a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = (r * c) # in kilometers

    return d


def haversine_array(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h


 
def dummy_manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_array(lat1, lng1, lat1, lng2)
    b = haversine_array(lat1, lng1, lat2, lng1)
    return a + b


def manhattan_distance(lat, lon):
    a = np.abs(lat -lon)
    return a


def equirectangular_distances(lat,lon):
    r = 6371
    return r*np.sqrt(lat**2 + lon**2)




def join_customer_location(customer_data,location_data):
    '''
        Here we join both customer and location data for both train and test and we drop some redundant columns.
    '''
    customer_data                        = customer_data.sort_values(by='customer_verified',ascending=False)
    customer_data                        = customer_data.drop_duplicates(subset=['customer_id'])
    customer_location_data               = pd.merge(location_data,customer_data,on='customer_id',how='left')
    
    customer_location_data['created_at'] = pd.to_datetime(customer_location_data['created_at']).dt.date
    customer_location_data['updated_at'] = pd.to_datetime(customer_location_data['updated_at']).dt.date
    return customer_location_data
    
def my_groupby(df,primary_keys,dictionary_ops,renaming_dict):
    '''
        primary_keys is a list of primary keys.
        dictionary_ops is the dictionay having the operations to be performed (example :- {'location_number':'count'})
        renaming_dict is the column to be renamed after joining and resetting index
    '''
    return df.groupby(primary_keys).agg(dictionary_ops).reset_index().rename(columns=renaming_dict)

def data_left_join(df1,df2,primary_key):
    '''
        df1 :- First dataframe
        df2 :- Second Dataframe
        primary_key :- The list of primary keys on which one needs to left join
    '''
    return df1.merge(df2,how='left',on=primary_key)    


def total_hours_open(opening_time):
    '''
        This function will calculate the opening duration of the restaurant 
        based on the columns - 'OpeningTime' and 'OpeningTime2'
    '''
    try:
        time_array   = opening_time.split('-')
        
        start_time   = datetime.strptime(time_array[0], "%I:%M%p")

        end_time     = datetime.strptime(time_array[1], "%I:%M%p")

        time_del     = end_time-start_time
        hours_open   = time_del.seconds/3600
        return hours_open
    except:
        return 0
    

def daily_hours_open(time1,time2):
    '''
        This function will calculate the opening time of a restaurant in each individual days
    '''
    try:
        start_time   = datetime.strptime(time1, "%H:%M:%S")

        end_time     = datetime.strptime(time2, "%H:%M:%S")

        time_del     = end_time-start_time
        hours_open   = time_del.seconds/3600.0
        return hours_open
    except:
        return 0
    
    
def night_service(time1,time2):
    '''
        Function to check is night service post 12:00 am is given or not
    '''
    try:
        time1 = datetime.strptime(time1, "%H:%M:%S")
        time2 = datetime.strptime(time2, "%H:%M:%S")
        
        if (time1.hour<4) or (time2.hour<4):
            return 1
        else:
            return 0
    except:
        return 0
        

def breakfast(time1,time2):
    '''
        Function to check if breakfast is served or not
    '''
    try:
        time1 = datetime.strptime(time1, "%H:%M:%S")
        time2 = datetime.strptime(time2, "%H:%M:%S")
        
        if (time1.hour<12) and (time1.hour>7):
            return 1
        else:
            return 0
    except:
        return 0
    

def lunch_snacks(time1,time2):
    '''
        Function to check if lunch or snacks is served or not.
    '''
    try:
        time1 = datetime.strptime(time1, "%H:%M:%S")
        time2 = datetime.strptime(time2, "%H:%M:%S")
        
        if (time2.hour>12) or (time2.hour<19):
            return 1
        else:
            return 0
    except:
        return 0

def dinner(time1,time2):
    '''
        Function to check if dinner is served or not.
    '''
    try:
        time1 = datetime.strptime(time1, "%H:%M:%S")
        time2 = datetime.strptime(time2, "%H:%M:%S")
        
        if (time2.hour>19):
            return 1
        else:
            return 0
    except:
        return 0
    